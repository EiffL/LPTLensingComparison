import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np
import jax_cosmo as jc
from jax_cosmo.scipy.integrate import simps
import jax_cosmo.constants as constants

import numpyro
import numpyro.distributions as dist

from jaxpm.pm import pm_forces, growth_factor, lpt, make_ode_fn
from jaxpm.kernels import fftk
from jaxpm.painting import cic_paint, compensate_cic, cic_paint_2d
from jaxpm.utils import gaussian_smoothing

import diffrax 
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt

import astropy.units as u

def linear_field(mesh_shape, box_size, pk, field):
  """
    Generate initial conditions.
  """
  kvec = fftk(mesh_shape)
  kmesh = sum(
      (kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
  pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
      box_size[0] * box_size[1] * box_size[2])

  field = jnp.fft.rfftn(field) * pkmesh**0.5
  field = jnp.fft.irfftn(field)
  return field

def lpt_lightcone(cosmo, initial_conditions, positions, a, mesh_shape, box_size):
    """
    Computes first order LPT lightcone
    """
    initial_force = pm_forces(positions, delta=initial_conditions).reshape(mesh_shape+[3])
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a).reshape([1,1,-1,1]) * initial_force
    dx.reshape([-1,3])

    # Paint the particles on a new mesh
    lightcone = cic_paint(jnp.zeros(mesh_shape),  positions+dx)
    # Apply de-cic filter to recover more signal on small scales
    lightcone = compensate_cic(lightcone)

    dx = box_size[0] / mesh_shape[0]
    dz = box_size[-1] / mesh_shape[-1]
    return lightcone, a, dx, dz

def pm_lightcone(cosmo, initial_conditions, positions, a_init, mesh_shape, box_size,
                 density_plane_width, density_plane_npix, density_plane_smoothing):
  """ Computes lensplane lightcone using a PM simulation
  """

  def density_plane_fn(t, y, args):
    """ Callback function to paint density plane
    """
    cosmo, _ , density_plane_width, density_plane_npix  = args
    positions = y[0]
    nx, ny, nz = mesh_shape

    # Converts time t to comoving distance in voxel coordinates
    w = density_plane_width / box_size[2] * mesh_shape[2]
    center = jc.background.radial_comoving_distance(cosmo, t) / box_size[2] * mesh_shape[2]

    xy = positions[..., :2]
    d = positions[..., 2]

    # Apply 2d periodic conditions
    xy = jnp.mod(xy, nx)

    # Rescaling positions to target grid
    xy = xy / nx * density_plane_npix

    # Selecting only particles that fall inside the volume of interest
    weight = jnp.where((d > (center - w / 2)) & (d <= (center + w / 2)), 1., 0.)

    # Painting density plane
    density_plane = cic_paint_2d(jnp.zeros([density_plane_npix, density_plane_npix]), xy, weight)

    # Apply density normalization
    density_plane = density_plane / ((nx / density_plane_npix) *
                                     (ny / density_plane_npix) * w)
    return density_plane

  # LPT initilization 
  dx, p, _ = lpt(cosmo, initial_conditions, positions, a_init)

  # Define parameters of the simulation
  n_lens    = int(box_size[-1] // density_plane_width)
  r         = jnp.linspace(0., box_size[-1], n_lens + 1)
  r_center  = 0.5 * (r[1:] + r[:-1])
  a_center  = jc.background.a_of_chi(cosmo, r_center)

  # Define the ODE solver
  ode_fn = make_ode_fn(mesh_shape)
  term      = ODETerm(lambda a, state, args: ode_fn(state, a, args[0]))
  solver    = Dopri5()
  saveat    = SaveAt(ts=a_center[::-1], fn=density_plane_fn)
  
  solution  = diffeqsolve(term, solver, t0=a_init, t1=1., dt0=0.05,
                          y0        = jnp.stack([positions+dx, p], axis=0),
                          args      = (cosmo,),
                          saveat    = saveat,
                          adjoint   = diffrax.RecursiveCheckpointAdjoint(5),
                          max_steps = 32)

  dx = box_size[0] / density_plane_npix
  dz = density_plane_width

  lightcone = jax.vmap(lambda x: gaussian_smoothing(x, density_plane_smoothing / dx ))(solution.ys)
  # Reorder the ligtcone to have the first lens plane at the end
  lightcone = jnp.transpose(lightcone[::-1],axes=(1, 2, 0))
  a = solution.ts[::-1]

  return lightcone, a, dx, dz

def convergence_Born(cosmo,
                     density_planes,
                     r,
                     a,
                     dx,
                     dz,
                     coords,
                     z_source):
  """
  Compute the Born convergence
  Args:
    cosmo: `Cosmology`, cosmology object.
    density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use
    coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
    z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
    name: `string`, name of the operation.
  Returns:
    `Tensor` of shape [batch_size, N, Nz], of convergence values.
  """
  # Compute constant prefactor:
  constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
  # Compute comoving distance of source galaxies
  r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

  convergence = 0
  n_planes = len(r)

  def scan_fn(carry, i):
    density_planes, a, r = carry

    p = density_planes[:,:,i]
    density_normalization = dz * r[i] / a[i]
    p = (p - p.mean()) * constant_factor * density_normalization

    # Interpolate at the density plane coordinates
    im = map_coordinates(p,
                         coords * r[i] / dx - 0.5,
                         order=1, mode="wrap")

    return carry, im * jnp.clip(1. - (r[i] / r_s), 0, 1000).reshape([-1, 1, 1])

  _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r), jnp.arange(n_planes))

  return convergence.sum(axis=0)

def make_full_field_model(field_size, field_npix, 
                          box_shape, box_size, 
                          pm=False, 
                          pm_density_plane_width=None, 
                          pm_density_plane_npix=None, 
                          pm_density_plane_smoothing=None):

  def forward_model(cosmo, nz_shear, initial_conditions):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Create initial conditions
    lin_field = linear_field(box_shape, box_size, pk_fn, initial_conditions)

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in box_shape]),axis=-1).reshape([-1,3])

    # Compute the scale factor that corresponds to each slice of the volume
    r = (jnp.arange(box_shape[-1]) + 0.5)*box_size[-1]/box_shape[-1]
    a = jc.background.a_of_chi(cosmo, r)

    cosmo = jc.Cosmology(Omega_c=cosmo.Omega_c, sigma8=cosmo.sigma8, Omega_b=cosmo.Omega_b,
                        h=cosmo.h, n_s=cosmo.n_s, w0=cosmo.w0, Omega_k=0., wa=0.)
    
    # Generate a lightcone with either a PM ot LPT simulation
    if pm:
      lightcone, a, dx, dz  = pm_lightcone(cosmo, lin_field, particles, a[-1], box_shape, box_size,
                                           pm_density_plane_width, pm_density_plane_npix, pm_density_plane_smoothing)
    else:
      lightcone, a, dx, dz  = lpt_lightcone(cosmo, lin_field, particles, a, box_shape, box_size)

    # Defining the coordinate grid for lensing map
    xgrid, ygrid = np.meshgrid(np.linspace(0, field_size, box_shape[0], endpoint=False), # range of X coordinates
                                np.linspace(0, field_size, box_shape[1], endpoint=False)) # range of Y coordinates
    coords = jnp.array((np.stack([xgrid, ygrid], axis=0)*u.deg).to(u.rad))

    # Generate convergence maps by integrating over nz and source planes
    convergence_maps = [simps(lambda z: nz(z).reshape([-1,1,1]) *
                                convergence_Born(cosmo, lightcone, r,a, dx, dz, coords, z), 0.01, 3.0, N=32)
                        for nz in nz_shear]

    # Reshape the maps to desired resoluton
    convergence_maps = [kmap.reshape([field_npix, box_shape[0] // field_npix,  field_npix, box_shape[1] // field_npix ]).mean(axis=1).mean(axis=-1)
                        for kmap in convergence_maps]

    return convergence_maps, lightcone

  return forward_model


# Build the probabilistic model
def full_field_probmodel(config):
  forward_model = make_full_field_model(config.field_size, config.field_npix,
                                        config.box_shape, config.box_size,
                                        pm=config.pm,
                                        pm_density_plane_width=config.pm_density_plane_width,
                                        pm_density_plane_npix=config.pm_density_plane_npix,
                                        pm_density_plane_smoothing=config.pm_density_plane_smoothing)
  
  # Sampling the cosmological parameters
  cosmo = config.fiducial_cosmology(**{k: numpyro.sample(k, v) for k, v in config.priors.items()})

  # Sampling the initial conditions
  initial_conditions = numpyro.sample('initial_conditions', dist.Normal(jnp.zeros(config.box_shape),
                                                                        jnp.ones(config.box_shape)))

  # Apply the forward model
  convergence_maps, _ = forward_model(cosmo, config.nz_shear, initial_conditions)

  # Define the likelihood of observations
  observed_maps = [numpyro.sample('kappa_%d'%i,
                                  dist.Normal(k, 
                                              config.sigma_e/jnp.sqrt(config.nz_shear[i].gals_per_arcmin2*
                                                                      (config.field_size*60/config.field_npix)**2)))
                   for i,k in enumerate(convergence_maps)]

  return observed_maps


def pixel_window_function(l, pixel_size_arcmin):
  """
  Calculate the pixel window function W_l for a given angular wave number l and pixel size.

  Parameters:
  - l: Angular wave number (can be a numpy array or a single value).
  - pixel_size_arcmin: Pixel size in arcminutes.

  Returns:
  - W_l: Pixel window function for the given l and pixel size.
  """
  # Convert pixel size from arcminutes to radians
  pixel_size_rad = pixel_size_arcmin * (np.pi / (180.0 * 60.0))

  # Calculate the Fourier transform of the square pixel (sinc function)
  # Note: l should be the magnitude of the angular wave number vector, |l| = sqrt(lx^2 + ly^2) for a general l
  # For simplicity, we assume l is already provided as |l|
  W_l = (jnp.sinc(l * pixel_size_rad / (2 * np.pi)))**2

  return W_l


def make_2pt_model(pixel_scale, ell, sigma_e=0.3):
  """
  Create a function that computes the theoretical 2-point correlation function for a given cosmology and redshift distribution.

  Parameters:
  - pixel_scale: Pixel scale in arcminutes.
  - ell: Angular wave number (numpy array).

  Returns:
  - forward_model: Function that computes the theoretical 2-point correlation function for a given cosmology and redshift distribution.
  """
  
  def forward_model(cosmo, nz_shear):      
    tracer = jc.probes.WeakLensing(nz_shear, sigma_e=sigma_e)
    cell_theory = jc.angular_cl.angular_cl(cosmo, ell, [tracer], nonlinear_fn=jc.power.linear)
    cell_theory = cell_theory * pixel_window_function(ell, pixel_scale)
    cell_noise = jc.angular_cl.noise_cl(ell,[tracer])
    return cell_theory, cell_noise

  return forward_model