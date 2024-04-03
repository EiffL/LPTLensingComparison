import jax
import jax.numpy as jnp
import jax_cosmo as jc
import pickle
import numpyro
import numpyro.distributions as dist
import haiku as hk

from jax_cosmo.scipy.integrate import simps
from jax.scipy.ndimage import map_coordinates
import jax_cosmo.constants as constants

from jaxpm.pm import pm_forces, growth_factor, growth_rate
from jaxpm.kernels import fftk
from jaxpm.painting import cic_paint, compensate_cic, cic_paint_2d, cic_read
from jaxpm.nn import NeuralSplineFourierFilter
from jaxpm.kernels import gradient_kernel, laplace_kernel, longrange_kernel
from jaxpm.utils import gaussian_smoothing

import diffrax 
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt


neural_spline_params = pickle.load( open( "camels_25_64_pkloss.params", "rb" ) )
model = hk.without_apply_rng( hk.transform(lambda x, a: NeuralSplineFourierFilter(n_knots=16, latent_size=32)(x, a)))


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

def lpt_lightcone(cosmo, initial_conditions, positions, a, mesh_shape):
    """ Computes first order LPT displacement """
    initial_force = pm_forces(positions, delta=initial_conditions).reshape(mesh_shape+[3])
    a  = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a).reshape([1,1,-1,1]) * initial_force
    p  = (a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo, a)) * growth_factor(cosmo, a)).reshape([1,1,-1,1]) * initial_force
    return dx.reshape([-1,3]),p.reshape([-1,3])

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
    im = map_coordinates(p, coords * r[i] / dx - 0.5, order=1, mode="wrap")

    return carry, im * jnp.clip(1. - (r[i] / r_s), 0, 1000).reshape([-1, 1, 1])

  # Similar to for loops but using a jaxified approach
  _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r), jnp.arange(n_planes))

  return convergence.sum(axis=0)



def make_full_field_model(field_size, field_npix, box_shape, box_size, method='lpt',density_plane_width=None, density_plane_npix=None,density_plane_smoothing=None):
  

  def density_plane_fn(t, y, args):
    
    cosmo, _ , density_plane_width, density_plane_npix  = args
    positions = y[0]
    nx, ny, nz = box_shape

    # Converts time t to comoving distance in voxel coordinates
    w = density_plane_width / box_size[2] * box_shape[2]
    center = jc.background.radial_comoving_distance(cosmo, t) / box_size[2] * box_shape[2]

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
     
  @jax.jit
  def neural_nbody_ode(a, state, args):
    """
      state is a tuple (position, velocities ) in internal units: [grid units, v=\frac{a^2}{H_0}\dot{x}]
      See this link for conversion rules: https://github.com/fastpm/fastpm#units
      """
    cosmo, params, _, _ = args
    pos = state[0]
    vel = state[1]

    kvec = fftk(box_shape)

    delta = cic_paint(jnp.zeros(box_shape), pos)

    delta_k = jnp.fft.rfftn(delta)

    # Computes gravitational potential
    pot_k = delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)

    # Apply a correction filter
    if params is not None:
      kk = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))
      pot_k = pot_k * (1. + model.apply(params, kk, jnp.atleast_1d(a)))

    # Computes gravitational forces
    forces = jnp.stack([cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k), pos) for i in range(3)], axis=-1)
    forces = forces * 1.5 * cosmo.Omega_m

    # Computes the update of position (drift)
    dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

    # Computes the update of velocity (kick)
    dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

    return jnp.stack([dpos, dvel], axis=0)


  def forward_model(cosmo, nz_shear, initial_conditions):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Create initial conditions
    lin_field = linear_field(box_shape, box_size, pk_fn, initial_conditions)

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in box_shape]),axis=-1).reshape([-1,3])


    cosmo = jc.Cosmology(Omega_c=cosmo.Omega_c, sigma8=cosmo.sigma8, Omega_b=cosmo.Omega_b,
                        h=cosmo.h, n_s=cosmo.n_s, w0=cosmo.w0, Omega_k=0., wa=0.)
    # Temporary fix
    cosmo._workspace = {}
    
    # Initial displacement
    if method=='lpt':
      # Compute the scale factor that corresponds to each slice of the volume
      r_center = (jnp.arange(box_shape[-1]) + 0.5)*box_size[-1]/box_shape[-1]
      a_center = jc.background.a_of_chi(cosmo, r_center)

      # Compute displacement and paint positions of particles onto lightcone
      eps,_     = lpt_lightcone(cosmo, lin_field, particles, a, box_shape)
      lightcone = cic_paint(jnp.zeros(box_shape),  particles+eps) 
      
      # Apply de-cic filter to recover more signal on small scales
      lightcone = compensate_cic(lightcone)
      
      dx = box_size[0]  / box_shape[0]
      dz = box_size[-1] / box_shape[-1]


    elif method=='pm':

      assert density_plane_width is not None
      assert density_plane_npix is not None

      density_plane_smoothing = 0.1
      
      a_init    = 0.01
      n_lens    = int(box_size[-1] // density_plane_width)
      r         = jnp.linspace(0., box_size[-1], n_lens + 1)
      r_center  = 0.5 * (r[1:] + r[:-1])
      a_center  = jc.background.a_of_chi(cosmo, r_center)

      eps, p    = lpt_lightcone(cosmo, lin_field, particles, a_init, box_shape)
      term      = ODETerm(neural_nbody_ode)
      solver    = Dopri5()
      saveat    = SaveAt(ts=a_center[::-1], fn=density_plane_fn)
     
      solution  = diffeqsolve(term, solver, t0=0.01, t1=1., dt0=0.05,
                              y0        = jnp.stack([particles+eps, p], axis=0),
                              args      = (cosmo, neural_spline_params, density_plane_width, density_plane_npix),
                              saveat    = saveat,
                              adjoint   = diffrax.RecursiveCheckpointAdjoint(5),
                              max_steps = 32)

      dx = box_size[0] / density_plane_npix
      dz = density_plane_width

      lightcone = jax.vmap(lambda x: gaussian_smoothing(x, density_plane_smoothing / dx ))(solution.ys)
      lightcone = lightcone[::-1]
      a         = solution.ts[::-1]
      lightcone = jnp.transpose(lightcone,axes=(1, 2, 0))
    
 
    # Defining the coordinate grid for lensing map
    xgrid, ygrid = jnp.meshgrid(jnp.linspace(0, field_size, box_shape[0], endpoint=False), # range of X coordinates
                                jnp.linspace(0, field_size, box_shape[1], endpoint=False)) # range of Y coordinates
    
    #coords       = jnp.array((jnp.stack([xgrid, ygrid], axis=0)*u.deg).to(u.rad))
    coords = jnp.array((jnp.stack([xgrid, ygrid], axis=0))*0.017453292519943295 ) # deg->rad

    # Generate convergence maps by integrating over nz and source planes
    convergence_maps = [simps(lambda z: nz(z).reshape([-1,1,1]) *
                              convergence_Born(cosmo, lightcone, r_center, a, dx, dz, coords, z), 0.01, 3.0, N=32) for nz in nz_shear]

    # Reshape the maps to desired resoluton
    convergence_maps = [kmap.reshape([field_npix, box_shape[0] // field_npix,  field_npix, box_shape[1] // field_npix ]).mean(axis=1).mean(axis=-1) for kmap in convergence_maps]

    return convergence_maps, lightcone

  return forward_model



# Build the probabilistic model
def full_field_probmodel(config):
  forward_model = make_full_field_model(config.field_size, config.field_npix,
                                        config.box_shape, config.box_size)
  
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
  pixel_size_rad = pixel_size_arcmin * (jnp.pi / (180.0 * 60.0))

  # Calculate the Fourier transform of the square pixel (sinc function)
  # Note: l should be the magnitude of the angular wave number vector, |l| = sqrt(lx^2 + ly^2) for a general l
  # For simplicity, we assume l is already provided as |l|
  W_l = (jnp.sinc(l * pixel_size_rad / (2 * jnp.pi)))**2

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
    tracer      = jc.probes.WeakLensing(nz_shear, sigma_e=sigma_e)
    cell_theory = jc.angular_cl.angular_cl(cosmo, ell, [tracer], nonlinear_fn=jc.power.linear)
    cell_theory = cell_theory * pixel_window_function(ell, pixel_scale)
    cell_noise  = jc.angular_cl.noise_cl(ell,[tracer])
    return cell_theory, cell_noise

  return forward_model
