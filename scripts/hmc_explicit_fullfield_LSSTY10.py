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
from sbi_lens.simulator.Lpt_field import lensingLpt
from numpyro.infer.reparam import LocScaleReparam, TransformReparam

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
N = 60
map_size = 5
sigma_e = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins = config_lsst_y_10.nbins
a = config_lsst_y_10.a
b = config_lsst_y_10.b
z0 = config_lsst_y_10.z0

truth = config_lsst_y_10.truth
omega_c, omega_b, sigma_8, h_0, n_s, w_0 = truth

params_name = ["omega_c", "omega_b", "sigma_8", "h_0", "w_0"]

model_lpt = partial(
    lensingLpt,
    N=N,
    map_size=map_size,
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


######### HACK TO INIT MCMC ON FID ##########
# condition numpyro model on fid

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

# the fid values to init the mcmc
init_values = {
    k: model_trace[k]["value"]
    for k in ["z", "omega_c", "sigma_8", "omega_b", "h_0", "n_s", "w_0"]
}


######### RUN FULL-FIELD MCMC ##########
num_samples = 300
num_chains = 1
thinning = 2
num_warmup = 300
chain_method = "vectorized"
nb_loop = 2
step_size = 1e-2
max_tree_depth = 5


def config(x):
    if type(x["fn"]) is dist.TransformedDistribution:
        return TransformReparam()
    elif (
        (type(x["fn"]) is dist.Normal or type(x["fn"]) is dist.TruncatedNormal)
        and ("decentered" not in x["name"])
        and ("n_s" not in x["name"])
    ):
        return LocScaleReparam(centered=0)
    else:
        return None


model = condition(model_lpt, {"n_s": 0.9645})

observed_model = condition(
    model,
    {
        "y": model_trace["y"]["value"],
    },
)

observed_model_reparam = reparam(observed_model, config=config)

# Building the sampling kernel
nuts_kernel = numpyro.infer.NUTS(
    model=observed_model_reparam,
    init_strategy=numpyro.infer.init_to_value(values=init_values),
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
    f"chains/posterior_full_field_lpt_lsst_y10_{args.filename}.npy",
    jnp.array(samples_ff_store).reshape([-1, 5]),
)

jnp.save(
    f"diagnostic/diagnostic_full_field_lsst_y10_{args.filename}.npy",
    jnp.array(nb_of_log_prob_evaluation),
)
