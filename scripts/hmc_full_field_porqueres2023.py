import argparse
import logging

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
from jax.lib import xla_bridge
from jax_lensing.configs import Porqueres_2023
from jax_lensing.model import make_full_field_model
from numpyro import distributions as dist
from numpyro.handlers import condition, seed, trace

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
fiducial_cosmology = Porqueres_2023.fiducial_cosmology
box_size = Porqueres_2023.box_size
box_shape = Porqueres_2023.box_shape
field_size = Porqueres_2023.field_size
field_npix = Porqueres_2023.field_npix
nz_shear = Porqueres_2023.nz_shear
sigma_e = Porqueres_2023.sigma_e
priors = Porqueres_2023.priors


lensing_model = jax.jit(
    make_full_field_model(
        field_size=field_size,
        field_npix=field_npix,
        box_size=box_size,
        box_shape=box_shape,
    )
)


# Define the probabilistic model
def model():
    """
    This function defines the top-level forward model for our observations
    """
    # Sampling initial conditions
    initial_conditions = numpyro.sample(
        "initial_conditions", dist.Normal(jnp.zeros(box_shape), jnp.ones(box_shape))
    )

    Omega_b = fiducial_cosmology.Omega_b
    h = fiducial_cosmology.h
    n_s = fiducial_cosmology.n_s
    w0 = fiducial_cosmology.w0
    Omega_c = numpyro.sample("Omega_c", priors["Omega_c"])
    sigma8 = numpyro.sample("sigma8", priors["sigma8"])

    cosmo = jc.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        sigma8=sigma8,
        h=h,
        n_s=n_s,
        w0=w0,
        wa=0.0,
        Omega_k=0.0,
    )

    # Generate random convergence maps
    convergence_maps, _ = lensing_model(cosmo, nz_shear, initial_conditions)

    # Apply noise to the maps (this defines the likelihood)
    observed_maps = [
        numpyro.sample(
            "kappa_%d" % i,
            dist.Normal(
                k,
                sigma_e
                / jnp.sqrt(
                    nz_shear[i].gals_per_arcmin2 * (field_size * 60 / field_npix) ** 2
                ),
            ),
        )
        for i, k in enumerate(convergence_maps)
    ]

    return observed_maps


######### HACK TO INIT MCMC ON FID ##########
# condition numpyro model on fid

fiducial_model = condition(
    model, {"Omega_c": fiducial_cosmology.Omega_c, "sigma8": fiducial_cosmology.sigma8}
)
# same seed we used to generate our fixed fid map
model_trace = trace(seed(fiducial_model, jax.random.PRNGKey(1234))).get_trace()

# the fid values to init the mcmc
init_values = {
    k: model_trace[k]["value"] for k in ["initial_conditions", "Omega_c", "sigma8"]
}


######### RUN FULL-FIELD MCMC ##########
num_samples = 3_000
num_chains = 1
thinning = 5
num_warmup = 300
chain_method = "vectorized"
nb_loop = 7
step_size = 1e-2
max_tree_depth = 4

observed_model = condition(
    model,
    {
        "kappa_0": model_trace["kappa_0"]["value"],
        "kappa_1": model_trace["kappa_1"]["value"],
        "kappa_2": model_trace["kappa_2"]["value"],
        "kappa_3": model_trace["kappa_3"]["value"],
    },
)

# Building the sampling kernel
nuts_kernel = numpyro.infer.NUTS(
    model=observed_model,
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
        samples_["Omega_c"],
        samples_["sigma8"],
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
            samples_["Omega_c"],
            samples_["sigma8"],
        ],
        axis=-1,
    )
    samples_ff_store.append(samples_)


######### SAVE CHAINS ##########
jnp.save(
    f"chains/posterior_full_field_lpt_porqueres_2023_{args.filename}.npy",
    jnp.array(samples_ff_store).reshape([-1, 2]),
)

jnp.save(
    f"diagnostic/diagnostic_full_field_lpt_porqueres_2023_{args.filename}.npy",
    jnp.array(nb_of_log_prob_evaluation),
)
