import argparse
import logging

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
from jax.lib import xla_bridge
from numpyro import distributions as dist
from sbi_lens.config import config_lsst_y_10
from functools import partial
from numpyro.handlers import condition, reparam, seed, trace
from sbi_lens.simulator.redshift import subdivide
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
from lenstools import ConvergenceMap
import astropy.units as u
import itertools
from functools import partial
from sbi_lens.simulator.Lpt_field import lensingLpt
import numpy as np

print(xla_bridge.get_backend().platform)

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
"unset XLA_FLAGS"


######### SCRIPT ARGS ##########
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--filename", type=str, default="res")
args = parser.parse_args()


######### RANDOM SEED ##########
key = jax.random.PRNGKey(3)
subkey = jax.random.split(key, 200)
key_par = subkey[args.seed]


######### LPT MODEL ##########
# setting

field_npix = N = 60
field_size = map_size = 5
sigma_e = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
n_nz_bins = nbins = config_lsst_y_10.nbins
a = config_lsst_y_10.a
b = config_lsst_y_10.b
z0 = config_lsst_y_10.z0

truth = config_lsst_y_10.truth
omega_c, omega_b, sigma_8, h_0, n_s, w_0 = truth
cosmo = jc.Planck15(
    Omega_c=omega_c, Omega_b=omega_b, sigma8=sigma_8, h=h_0, n_s=n_s, w0=w_0
)

params_name = ["omega_c", "omega_b", "sigma_8", "h_0", "w_0"]

nz = jc.redshift.smail_nz(a, b, z0, gals_per_arcmin2=gals_per_arcmin2, zmax=2.6)
nz_shear = subdivide(nz, nbins=nbins, zphot_sigma=0.05)

# ficucial observation
model_lpt = partial(
    lensingLpt,
    N=60,
    map_size=5,
    box_size=[400.0, 400.0, 4000.0],
    box_shape=[300, 300, 128],
    gal_per_arcmin2=gals_per_arcmin2,
    sigma_e=sigma_e,
    nbins=nbins,
    a=a,
    b=b,
    z0=z0,
    with_noise=True,
)

fiducial_model = condition(
    model_lpt,
    {
        "omega_c": omega_c,
        "omega_b": omega_b,
        "sigma_8": sigma_8,
        "h_0": h_0,
        "n_s": n_s,
        "w_0": w_0,
    },
)

# sample a mass map and save corresponding true parameters
model_trace = trace(seed(fiducial_model, jax.random.PRNGKey(42))).get_trace()

m_data = model_trace["y"]["value"]

l_edges_kmap = np.arange(100.0, 2000.0, 50.0)

ell = ConvergenceMap(m_data[:, :, 0], angle=field_size * u.deg).cross(
    ConvergenceMap(m_data[:, :, 0], angle=field_size * u.deg),
    l_edges=l_edges_kmap,
)[0]

ps = []
for i, j in itertools.combinations_with_replacement(range(nbins), 2):
    ps_ij = ConvergenceMap(m_data[:, :, i], angle=field_size * u.deg).cross(
        ConvergenceMap(m_data[:, :, j], angle=field_size * u.deg),
        l_edges=l_edges_kmap,
    )[1]
    ps.append(ps_ij)
ps = np.array(ps)


# Let's build a theory model
from jax_lensing.model import make_2pt_model

# Generate the forward model given these survey settings
theory_model = jax.jit(
    make_2pt_model(pixel_scale=field_size / field_npix * 60, ell=ell, sigma_e=sigma_e)
)

cell_theory, cell_noise = theory_model(cosmo, nz_shear)

# Generating the covariance and precision matrix at the fiducial cosmology
tracer = jc.probes.WeakLensing(nz_shear, sigma_e=sigma_e)
C = jc.angular_cl.gaussian_cl_covariance(
    ell, [tracer], cell_theory, cell_noise, f_sky=field_size**2 / (41_253)
)
P = jc.sparse.to_dense(jc.sparse.inv(C))
C = jc.sparse.to_dense(C)


def theory_prob_model():
    """
    This function defines the top-level forward model for our observations
    """
    omega_c = numpyro.sample("omega_c", dist.TruncatedNormal(0.2664, 0.2, low=0))
    omega_b = numpyro.sample("omega_b", dist.Normal(0.0492, 0.006))
    sigma_8 = numpyro.sample("sigma_8", dist.Normal(0.831, 0.14))
    h_0 = numpyro.sample("h_0", dist.Normal(0.6727, 0.063))
    n_s = 0.9645  # numpyro.sample("n_s", dist.Normal(0.9645, 0.08))
    w_0 = numpyro.sample("w_0", dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3))

    cosmo = jc.Cosmology(
        Omega_c=omega_c,
        Omega_b=omega_b,
        sigma8=sigma_8,
        h=h_0,
        n_s=n_s,
        w0=w_0,
        wa=0.0,
        Omega_k=0.0,
    )

    # Generate signal
    cell_theory, cell_noise = theory_model(cosmo, nz_shear)

    cl = numpyro.sample(
        "cl",
        dist.MultivariateNormal(
            cell_theory.flatten() + cell_noise.flatten(),
            precision_matrix=P,
            covariance_matrix=C,
        ),
    )

    return cl


######### RUN FULL-FIELD MCMC ##########
num_samples = 4000
num_chains = 1
thinning = 5
num_warmup = 300
chain_method = "vectorized"
nb_loop = 3
step_size = 1e-2
max_tree_depth = 4


# Let's condition the model on observations and sample from it
def config(x):
    if type(x["fn"]) is dist.TransformedDistribution:
        return TransformReparam()
    elif (
        type(x["fn"]) is dist.Normal
        and ("decentered" not in x["name"])
        and ("cl" not in x["name"])
    ):
        return LocScaleReparam(centered=0)
    else:
        return None


# Let's condition the model on the observed maps
observed_cl_model = condition(theory_prob_model, {"cl": np.stack(ps).flatten()})

# And reparametrize the variables to standardize their scale
observed_cl_model_reparam = reparam(observed_cl_model, config=config)

# Building the sampling kernel
nuts_kernel = numpyro.infer.NUTS(
    model=observed_cl_model_reparam,
    init_strategy=numpyro.infer.init_to_median,
    max_tree_depth=max_tree_depth,
    step_size=step_size,
)

mcmc = numpyro.infer.MCMC(
    nuts_kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    chain_method=chain_method,
    thinning=thinning,
    progress_bar=True,
)

samples_ff_store = []
nb_of_log_prob_evaluation = []
mcmc.run(key_par, extra_fields=("num_steps",))
samples_ = mcmc.get_samples()
nb_of_log_prob_evaluation.append(mcmc.get_extra_fields()["num_steps"])
mcmc.post_warmup_state = mcmc.last_state

# save only sample of interest
samples_ = jnp.stack(
    [
        samples_["omega_c"],
        samples_["omega_b"],
        samples_["sigma_8"],
        samples_["h_0"],
        samples_["w_0"],
    ],
    axis=-1,
)

samples_ff_store.append(samples_)

for i in range(1, nb_loop):
    mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("num_steps",))
    samples_ = mcmc.get_samples()
    nb_of_log_prob_evaluation.append(mcmc.get_extra_fields()["num_steps"])
    mcmc.post_warmup_state = mcmc.last_state

    # save only sample of interest
    samples_ = jnp.stack(
        [
            samples_["omega_c"],
            samples_["omega_b"],
            samples_["sigma_8"],
            samples_["h_0"],
            samples_["w_0"],
        ],
        axis=-1,
    )
    samples_ff_store.append(samples_)


######### SAVE CHAINS ##########
jnp.save(
    f"chains/posterior_PS_lpt_lsst_y10_{args.filename}.npy",
    jnp.array(samples_ff_store).reshape([-1, 5]),
)

jnp.save(
    f"diagnostic/diagnostic_PS_lsst_y10_{args.filename}.npy",
    jnp.array(nb_of_log_prob_evaluation),
)
