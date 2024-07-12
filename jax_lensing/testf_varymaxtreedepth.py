import pickle
import os,sys
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.scipy.stats import norm
from numpyro.handlers import seed, trace, condition
import diffrax 
import argparse
from pathlib import Path
import numpy as np
from functools import partial
import jax.numpy.fft as fft

parser  = argparse.ArgumentParser()
parser.add_argument('method'      , default=None, type=str, help='')
parser.add_argument('run'         , default=None, type=int, help='')
parser.add_argument('resume_state', default=None, type=int, help='')
parser.add_argument('maxtreedepth', default=None, type=int, help='')
parser.add_argument('--trace'     , default=False, dest='trace',action='store_true')
parser.add_argument('--warmup'     , default=False, dest='warmup',action='store_true')
args         = parser.parse_args()
resume_state = args.resume_state
run          = args.run
method       = args.method
trace        = args.trace
warmup       = args.warmup
maxtreedepth = args.maxtreedepth


def filter(ngrid, reso_rad, cut_off):
    ''' For now, assume square grid'''
    #reso_rad=reso*0.000290888
    '''
    nsub = int(ngrid/2 + 1)
    i,j  = jnp.meshgrid(np.arange(nsub), np.arange(nsub))
    submatrix = 2*jnp.pi*jnp.sqrt(i**2 + j**2)/reso_rad/jnp.float32(ngrid)

    result = jnp.zeros([ngrid, ngrid])
    result[0:nsub, 0:nsub] = submatrix
    result[0:nsub,nsub:]   = jnp.fliplr(submatrix[:,1:-1])
    result[nsub:,:]        = jnp.flipud(result[1:nsub-1,:])
    tmp = jnp.around(result).astype(int)
    print(np.max(tmp))
    mask=np.ones_like(tmp)
    mask[tmp>cut_off]=0
    '''
    nsub = int(ngrid / 2 + 1)
    i, j = jnp.meshgrid(jnp.arange(nsub), jnp.arange(nsub))
    submatrix = 2 * jnp.pi * jnp.sqrt(i**2 + j**2) / reso_rad / jnp.float32(ngrid)

    result = jnp.zeros([ngrid, ngrid])
    result = result.at[0:nsub, 0:nsub].set(submatrix)
    result = result.at[0:nsub, nsub:].set(jnp.fliplr(submatrix[:, 1:-1]))
    result = result.at[nsub:, :].set(jnp.flipud(result[1:nsub-1, :]))
    tmp = jnp.around(result).astype(int)

    mask = jnp.ones_like(tmp)
    mask = mask.at[tmp > cut_off].set(0)
    return mask


dir  = '/pscratch/sd/y/yomori/lssty10/maxtreedepth%d/'%maxtreedepth
name = '%s_paperrun_flatprior_run1_lssty1'%method
Path(dir).mkdir(parents=True, exist_ok=True)


sigma_e  = 0.26
ngal     = jnp.array([2.00,2.00,2.00,2.00,2.00])

tmp      = jnp.load('nz/nz_lssty1_srd.npy'); print(tmp.shape)
zz       = tmp[:,0]
nz1      = tmp[:,1]
nz2      = tmp[:,2]
nz3      = tmp[:,3]
nz4      = tmp[:,4]
nz5      = tmp[:,5]


nz      = [nz1,nz2,nz3,nz4,nz5] 

nz_shear = [jc.redshift.kde_nz(zz,nz[i],bw=0.01, zmax=2.5, gals_per_arcmin2=ngal[i]) for i in range(5)]

nbins    = len(nz_shear)

Omega_b = 0.0492
Omega_c = 0.2664 
sigma8  = 0.831
h       = 0.6727
n_s     = 0.9645
w0      = -1
cosmo   = jc.Cosmology(Omega_c = Omega_c,
                       sigma8  = sigma8,
                       Omega_b = Omega_b,
                       Omega_k = 0.,
                       h   = h,
                       n_s = n_s,
                       w0  = w0,
                       wa  = 0.)


# Now, let's build a full field model
import numpyro
import numpyro.distributions as dist
from jax_lensing.model import make_full_field_model

# High resolution settings
# Note: this low resolution on the los impacts a tiny bit the cross-correlation signal,
# but it's probably worth it in terms of speed gains
#box_size  = [600., 600., 3500.]     # In Mpc/h [RUN2]
box_size  = [400., 400., 4600.]     # In Mpc/h [RUN3]
box_shape = [200, 200,  128]       # Number of voxels/particles per side


# Specify the size and resolution of the patch to simulate
field_size = 5.0    # transverse size in degrees [RUN3]
field_npix = 50     # number of pixels per side
pixel_size = field_size * 60 / field_npix
print("Pixel size in arcmin: ", pixel_size)


lowpass = 500
reso    = (field_size/field_npix)*60 # resolution in arcmin. 
ang     = 0.0166667*(reso)*50  #angle of the fullfield in deg

# Noise covariance
print('Computing filter')
f = filter(field_npix,ang/field_npix*0.0174533,lowpass) #low_pass_filter((50,50), 500, ang/npix*0.0174533)


print('Computing noise covriance matrix')
ret=np.zeros((int(field_npix*field_npix),50000))

for i in range(0,50000):
    tmp      = np.random.normal(np.zeros((50,50)), sigma_e/np.sqrt(ngal[0]*(ang*60/field_npix)**2)) 
    ret[:,i] = fft.ifft2(fft.fft2(tmp)*f).real.flatten()

noisecov = np.cov(ret)
print('Done')

# Generate the forward model given these survey settings
lensing_model = jax.jit(make_full_field_model( field_size  = field_size,
                                                field_npix = field_npix,
                                                box_size   = box_size,
                                                box_shape  = box_shape,
                                                method     = method,
                                                density_plane_width = 100,
                                                density_plane_npix  = 300
                                             ))
# Define the probabilistic model
def model():
  """
  This function defines the top-level forward model for our observations
  """
  # Sampling initial conditions
  initial_conditions = numpyro.sample('initial_conditions', dist.Normal(jnp.zeros(box_shape),
                                                                        jnp.ones(box_shape)))

  #Omega_b = 0.049
  Omega_b = 0.0492 #numpyro.sample("omega_b", dist.Normal(0.0492, 0.006))
  Omega_c = numpyro.sample("omega_c", dist.Uniform(0.05, 1.0))
  sigma8  = numpyro.sample("sigma8" , dist.Uniform(0.1, 2.0))
  h       = 0.6726 #numpyro.sample("h_0"    , dist.Normal(0.6727, 0.063))
  w0      = -1     #numpyro.sample("w_0"    , dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3)) #-1
  n_s     = 0.9645 #0.9624

  #print('############################ %.3f %.3f ############################'%(Omega_c,sigma8) )
  cosmo   = jc.Cosmology(Omega_c = Omega_c,
                         sigma8  = sigma8,
                         Omega_b = Omega_b,
                         Omega_k = 0.,
                         h   = h,
                         n_s = n_s,
                         w0  = w0,
                         wa  = 0.)

  # Generate random convergence maps
  convergence_maps, _ = lensing_model(cosmo, nz_shear, initial_conditions)

  numpyro.deterministic('noiseless_convergence_0', convergence_maps[0])
  numpyro.deterministic('noiseless_convergence_1', convergence_maps[1])
  numpyro.deterministic('noiseless_convergence_2', convergence_maps[2])
  numpyro.deterministic('noiseless_convergence_3', convergence_maps[3])
  numpyro.deterministic('noiseless_convergence_4', convergence_maps[4])
  


  # Apply noise to the maps (this defines the likelihood)
  #observed_maps = [numpyro.sample('kappa_%d'%i,dist.Normal(k, sigma_e/jnp.sqrt(nz_shear[i].gals_per_arcmin2*(field_size*60/field_npix)**2)))for i,k in enumerate(convergence_maps)]
  #noise = [numpyro.sample('noise_%d'%i,dist.Normal(jnp.zeros_like(convergence_maps[0]), sigma_e/jnp.sqrt(nz_shear[i].gals_per_arcmin2*(field_size*60/field_npix)**2)))for i,k in enumerate(convergence_maps)]
  
  #filt = low_pass_filter(noise[0].shape,500,pixel_size)
  #import pdb;pdb.set_trace()
  #obslp = [fft.ifft2(fft.fft2(convergence_maps[i]+noise[i])*filt).real for i in range(len(convergence_maps)) ]
  #import pdb; pdb.set_trace()
  #f = filter(field_npix,ang/field_npix*0.0174533,lowpass) #low_pass_filter((50,50), 500, ang/npix*0.0174533)

  reso_rad=ang/field_npix*0.0174533
  nsub = int(field_npix / 2 + 1)
  i, j = jnp.meshgrid(jnp.arange(nsub), jnp.arange(nsub))
  submatrix = 2 * jnp.pi * jnp.sqrt(i**2 + j**2) / reso_rad / jnp.float32(field_npix)

  result = jnp.zeros([field_npix, field_npix])
  result = result.at[0:nsub, 0:nsub].set(submatrix)
  result = result.at[0:nsub, nsub:].set(jnp.fliplr(submatrix[:, 1:-1]))
  result = result.at[nsub:, :].set(jnp.flipud(result[1:nsub-1, :]))
  tmp = jnp.around(result).astype(int)

  mask = jnp.ones_like(tmp)
  #mask = mask.at[tmp > lowpass].set(0)
  #mask = mask.at[jnp.where(tmp > lowpass)].set(0)
  mask = jnp.where(tmp > lowpass, 0, mask)

  obslp         = [fft.ifft2(fft.fft2(convergence_maps[i])*mask).real for i in range(len(convergence_maps)) ] 
  numpyro.deterministic('observed_maps_2s', obslp[2])
  

  observed_maps = [numpyro.sample('kappa_%d'%i, dist.MultivariateNormal(obslp[i].flatten(), noisecov+1e-10*jnp.eye(int(field_npix*field_npix)))) for i in range(len(convergence_maps)) ]

  numpyro.deterministic('observed_maps_0', observed_maps[0])
  numpyro.deterministic('observed_maps_1', observed_maps[1])
  numpyro.deterministic('observed_maps_2', observed_maps[2])
  numpyro.deterministic('observed_maps_3', observed_maps[1])
  numpyro.deterministic('observed_maps_4', observed_maps[2])
  
  return observed_maps


# Create a random realization of a map with fixed cosmology
gen_model     = condition(model, {"omega_c": 0.2664,
                                  #"omega_b": 0.0492,
                                  "sigma8" : 0.831,
                                  #"h_0"    : 0.6727
                                 })

if trace is False:
    model_tracer = numpyro.handlers.trace(numpyro.handlers.seed(gen_model, jax.random.PRNGKey(1234) ))
    model_trace  = model_tracer.get_trace()

    if run!=-1:
        np.savez(dir+'/model_trace_noiseless_convergence_mainseed_%s_%d.npz'%(method,run),conv1 = model_trace['noiseless_convergence_0']['value']
                                                                                         ,conv2 = model_trace['noiseless_convergence_1']['value']
                                                                                         ,conv3 = model_trace['noiseless_convergence_2']['value']
                                                                                         ,conv4 = model_trace['noiseless_convergence_3']['value']
                                                                                         ,conv5 = model_trace['noiseless_convergence_4']['value']
                                                                                         ,kappa1 = model_trace['kappa_0']['value']
                                                                                         ,kappa2 = model_trace['kappa_1']['value']
                                                                                         ,kappa3 = model_trace['kappa_2']['value']
                                                                                         ,kappa4 = model_trace['kappa_3']['value']
                                                                                         ,kappa5 = model_trace['kappa_4']['value']
                                                                                         ,convs3 = model_trace['observed_maps_2s']['value']
                                                                                         
                                                                                                            
                )
    sys.exit()

else:
    model_tracer = numpyro.handlers.trace(numpyro.handlers.seed(gen_model, run ))
    model_trace  = model_tracer.get_trace()

    np.savez(dir+'/model_trace_noiseless_convergence_differentseed_%s_%d.npz'%(method,run),kappa1 = model_trace['noiseless_convergence_0']['value']
                                                                                          ,kappa2 = model_trace['noiseless_convergence_1']['value']
                                                                                          ,kappa3 = model_trace['noiseless_convergence_2']['value']
                                                                                          ,kappa4 = model_trace['noiseless_convergence_3']['value']
                                                                                          ,kappa5 = model_trace['noiseless_convergence_4']['value']
                                                                                                        
            )
    sys.exit()


# Let's condition the model on the observed maps
observed_model = condition(model, {'kappa_0': model_trace['kappa_0']['value'],
                                   'kappa_1': model_trace['kappa_1']['value'],
                                   'kappa_2': model_trace['kappa_2']['value'],
                                   'kappa_3': model_trace['kappa_3']['value'],
                                   'kappa_4': model_trace['kappa_4']['value'],
                                   })


if warmup:
    nuts_kernel = numpyro.infer.NUTS(
                                 model=observed_model,
                                 init_strategy = partial(numpyro.infer.init_to_value, values={'omega_c': 0.2664,
                                                                                              'sigma8' : 0.831,
                                                                                              #'omega_b': 0.0492,
                                                                                              #'h_0'    : 0.6727,
                                                                                              #'w_0'    : -1,
                                                                                              'initial_conditions': model_trace['initial_conditions']['value']}),
                                 max_tree_depth = maxtreedepth,
                                 step_size      = 1.0e-2
                                )

    mcmc = numpyro.infer.MCMC(
                          nuts_kernel,
                          num_warmup   = 360,
                          num_samples  = 2,
                          num_chains   = 1,
                          thinning     = 2,
                          progress_bar = True
                         )


else:
    #mass_matrix = jnp.load(dir+'mass_matrix.npy',allow_pickle=True)
    #import pdb; pdb.set_trace()
    #with open(dir+'mass_matrix.pkl', 'rb') as f:
    #    mass_matrix = pickle.load(f)

    nuts_kernel = numpyro.infer.NUTS(
                                 model=observed_model,
                                 init_strategy = partial(numpyro.infer.init_to_value, values={'omega_c': 0.2664,
                                                                                              'sigma8' : 0.831,
                                                                                              #'omega_b': 0.0492,
                                                                                              #'h_0'    : 0.6727,
                                                                                              #'w_0'    : -1,
                                                                                              'initial_conditions': model_trace['initial_conditions']['value']}),
                                 max_tree_depth = maxtreedepth,
                                 step_size      = 4.0e-3,
                                 #inverse_mass_matrix = mass_matrix
                                )

    mcmc = numpyro.infer.MCMC(
                          nuts_kernel, 
                          num_warmup   = 0,
                          num_samples  = 100,
                          num_chains   = 1,
                          thinning     = 1,
                          progress_bar = True
                         )


if resume_state<0:

    print("---------------STARTING SAMPLING-------------------")
    mcmc.run( jax.random.PRNGKey(run))
    print("-----------------DONE SAMPLING---------------------")

    if warmup:
        mass_matrix = mcmc._last_state.adapt_state.inverse_mass_matrix
        with open(dir+'mass_matrix.pkl', 'wb') as f:
            pickle.dump(mass_matrix, f)
        sys.exit()

    res = mcmc.get_samples()



    # Saving an intermediate checkpoint
    with open(dir+'%s_%d_0.pickle'%(name,run), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del res

    final_state = mcmc.last_state
    with open(dir+'/state_%s_%d_0.pkl'%(name,run), 'wb') as f:
        pickle.dump(final_state, f)

    #sys.exit()
    # Continue on
    for i in range(1,50):
        print('round',i,'done')
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key)
        res = mcmc.get_samples()
        with open(dir+'%s_%d_%d.pickle'%(name,run,i), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del res

        final_state = mcmc.last_state
        with open(dir+'/state_%s_%d_%d.pkl'%(name,run,i), 'wb') as f:
            pickle.dump(final_state, f)

else:
    # Save
    with open(dir+'state_%s_%d_%d.pkl'%(name,run,resume_state), 'rb') as f:
        mcmc.post_warmup_state = pickle.load(f)

    for i in range(resume_state+1,resume_state+50):
        mcmc.run(mcmc.post_warmup_state.rng_key)
        res = mcmc.get_samples()
        with open(dir+'/%s_%d_%d.pickle'%(name,run,i), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del res

        final_state = mcmc.last_state
        with open(dir+'/state_%s_%d_%d.pkl'%(name,run,i), 'wb') as f:
            pickle.dump(final_state, f)


#mcmc.run(jax.random.PRNGKey(0))

#res = mcmc.get_samples()


#import pickle
#with open('/pscratch/sd/y/yomori/simplesinglebin_model_hmc_chain.pickle', 'wb') as handle:
#    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
