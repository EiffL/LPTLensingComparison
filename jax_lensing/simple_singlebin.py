import pickle
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.scipy.stats import norm
from numpyro.handlers import seed, trace, condition
'''
z = jnp.linspace(0, 2.5, 1000)

cosmo = jc.Planck15()

nz_shear = [jc.redshift.kde_nz(z,
                               norm.pdf(z, loc=z_center, scale=0.12) ,
                               bw=0.01, zmax=2.5, gals_per_arcmin2=g )
                for z_center, g in zip([0.5, 1., 1.5, 2.], [7,8.5, 7.5, 7])]
nbins = len(nz_shear)

sigma_e=0.26
'''

import argparse


parser  = argparse.ArgumentParser()
#parser.add_argument('resume_state', default=False, dest='resume', action='store_true')
parser.add_argument('run'         , default=None, type=int, help='')
parser.add_argument('resume_state', default=None, type=int, help='')
args         = parser.parse_args()
resume_state = args.resume_state
run          =  args.run




sigma_e=0.26
ngal   = 6
a,b,z0 = 22,11.5,0.75
zz     = jnp.linspace(0,1.2,1000)
nz     = zz**a*jnp.exp(-((zz / z0) ** b))
nz     = nz/jnp.sum(nz)/(zz[1]-zz[0])

nz_shear = [jc.redshift.kde_nz(zz,nz,bw=0.01, zmax=2.5, gals_per_arcmin2=ngal) for i in range(1)]

nbins    = len(nz_shear)


Omega_b = 0.049
Omega_c = 0.315 - Omega_b
sigma_8 = 0.8
h       = 0.677
n_s     = 0.9624
w0      = -1
cosmo   = jc.Cosmology(Omega_c = Omega_c,
                       sigma8  = sigma_8,
                       Omega_b = Omega_b,
                       Omega_k = 0.,
                       h   = h,
                       n_s = n_s,
                       w0  = w0,
                       wa  = 0.)

# Specify the size and resolution of the patch to simulate
field_size = 5.17   # transverse size in degrees
field_npix = 60   # number of pixels per side
print("Pixel size in arcmin: ", field_size * 60 / field_npix)


# Now, let's build a full field model
import numpyro
import numpyro.distributions as dist
from jax_lensing.model import make_full_field_model

# High resolution settings
# Note: this low resolution on the los impacts a tiny bit the cross-correlation signal,
# but it's probably worth it in terms of speed gains
box_size  = [400., 400., 4000.]     # In Mpc/h
box_shape = [300,  300,  256]       # Number of voxels/particles per side

# Generate the forward model given these survey settings
lensing_model = jax.jit(make_full_field_model( field_size=field_size,
                                            field_npix=field_npix,
                                            box_size=box_size,
                                            box_shape=box_shape))
# Define the probabilistic model
def model():
  """
  This function defines the top-level forward model for our observations
  """
  # Sampling initial conditions
  initial_conditions = numpyro.sample('initial_conditions', dist.Normal(jnp.zeros(box_shape),
                                                                        jnp.ones(box_shape)))
  # Generate random convergence maps
  convergence_maps, _ = lensing_model(cosmo, nz_shear, initial_conditions)

  # Apply noise to the maps (this defines the likelihood)
  observed_maps = [numpyro.sample('kappa_%d'%i,
                                  dist.Normal(k, sigma_e/jnp.sqrt(nz_shear[i].gals_per_arcmin2*(field_size*60/field_npix)**2)))
                   for i,k in enumerate(convergence_maps)]

  return observed_maps


model_tracer = numpyro.handlers.trace(numpyro.handlers.seed(model, jax.random.PRNGKey(1234)))
model_trace = model_tracer.get_trace()

from functools import partial

# Let's condition the model on the observed maps
observed_model = condition(model, {'kappa_0': model_trace['kappa_0']['value']
                                   #'kappa_1': model_trace['kappa_1']['value'],
                                   #'kappa_2': model_trace['kappa_2']['value'],
                                   #'kappa_3': model_trace['kappa_3']['value']
                                   })

# Building the sampling kernel
nuts_kernel = numpyro.infer.NUTS(
    model=observed_model,
    init_strategy=partial(numpyro.infer.init_to_value, values={'omega_c': cosmo.Omega_c,
                                                               'sigma8': cosmo.sigma8,
                                                               'initial_conditions': model_trace['initial_conditions']['value']}),
    max_tree_depth=3,
    step_size=0.05)

mcmc = numpyro.infer.MCMC(
       nuts_kernel,
       num_warmup=0,
       num_samples=1000,
       num_chains=1,
       #chain_method='parallel',
       thinning=10,
       progress_bar=True
    )


if resume_state<0:

    print("---------------STARTING SAMPLING-------------------")
    mcmc.run( jax.random.PRNGKey(run))
    print("-----------------DONE SAMPLING---------------------")

    res = mcmc.get_samples()

    dir  = '/pscratch/sd/y/yomori/' 
    name = 'lpt_singlebin_flatprior'

    # Saving an intermediate checkpoint
    with open(dir+'%s_%d_0.pickle'%(name,run), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del res

    final_state = mcmc.last_state
    with open(dir+'/state_%s_%d_0.pkl'%(name,run), 'wb') as f:
        pickle.dump(final_state, f)


    # Continue on
    for i in range(1,500):
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

    for i in range(resume_state+1,resume_state+500):
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
