import argparse
import csv
import os
import pickle
from functools import partial
from pickle import HIGHEST_PROTOCOL, load

import GPUtil
import haiku as hk
import jax
import jax.numpy as jnp
import jax.numpy.fft as fft
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import psutil
import tensorflow as tf
import tensorflow_probability as tfp
from chainconsumer import ChainConsumer
from jax.lib import xla_bridge
from numpyro.handlers import condition
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP
from sbi_lens.normflow.train_model import TrainModel
from tqdm import tqdm

from jax_lensing.model import make_full_field_model

print(xla_bridge.get_backend().platform)

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Get CPU memory usage
cpu_memory_usage = psutil.virtual_memory().percent
print("CPU Memory Usage:", cpu_memory_usage, "%")


# Get available GPUs
gpus = GPUtil.getGPUs()
for gpu in gpus:
    memory_usage_percentage = (gpu.memoryUsed / gpu.memoryTotal) * 100
    print(
        f"GPU ID: {gpu.id}, GPU Name: {gpu.name}, GPU Memory Usage: {memory_usage_percentage:.2f}%"
    )


######### SCRIPT ARGS ##########
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_nb", type=int, default=1)
parser.add_argument(
    "--kind_of_analysis", type=str, default="full-field"
)  #'full-field or two-points'
parser.add_argument("--forward_model", type=str, default="lpt")  #'lpt or pm'
parser.add_argument("--lowpass", type=int, default=500)
parser.add_argument("--nb_sim_for_compression", type=int, default=80_000)
parser.add_argument("--nb_sim_for_inference", type=int, default=10_000)
parser.add_argument("--total_steps_train_compressor", type=int, default=10_000)
parser.add_argument("--total_steps_train_nde", type=int, default=30_000)
parser.add_argument(
    "--loss_function_compressor", type=str, default="train_compressor_vmim"
)  #'train_compressor_vmim or train_compressor_mse'
args = parser.parse_args()


print("######## SCRIPT CONGIG ########")
nb_sim_for_compression = args.nb_sim_for_compression
nb_sim_for_inference = args.nb_sim_for_inference
loss_function_compressor = args.loss_function_compressor
kind_of_analysis = args.kind_of_analysis
forward_model = args.forward_model
total_steps_train_compressor = args.total_steps_train_compressor
total_steps_train_nde = args.total_steps_train_nde
lowpass = args.lowpass
experiment_nb = args.experiment_nb

PATH_experiment = f"{forward_model}_{kind_of_analysis}_{lowpass}_{experiment_nb}"
os.makedirs(f"./{PATH_experiment}/fig")
os.makedirs(f"./{PATH_experiment}/save_params")


data = [
    [
        "forward_model",
        "kind_of_analysis",
        "lowpass",
        "total_steps_train_nde",
        "nb_sim_for_compression",
        "nb_sim_for_inference",
        "total_steps_train_compressor",
        "total_steps_train_nde",
        "experiment_nb",
    ],
    [
        forward_model,
        kind_of_analysis,
        lowpass,
        total_steps_train_nde,
        nb_sim_for_compression,
        nb_sim_for_inference,
        total_steps_train_compressor,
        total_steps_train_nde,
        experiment_nb,
    ],
]

with open(f"{PATH_experiment}/script_config.csv", "w", newline="") as file:
    writer = csv.writer(file)
    # Write each row of data to the CSV file
    writer.writerows(data)

print("######## FORWARD MODEL AND CONFIG ########")
print("Utils function used to cut some scales")


def filter(ngrid, reso_rad, cut_off):
    nsub = int(ngrid / 2 + 1)
    i, j = jnp.meshgrid(jnp.arange(nsub), jnp.arange(nsub))
    submatrix = 2 * jnp.pi * jnp.sqrt(i**2 + j**2) / reso_rad / jnp.float32(ngrid)

    result = jnp.zeros([ngrid, ngrid])
    result = result.at[0:nsub, 0:nsub].set(submatrix)
    result = result.at[0:nsub, nsub:].set(jnp.fliplr(submatrix[:, 1:-1]))
    result = result.at[nsub:, :].set(jnp.flipud(result[1 : nsub - 1, :]))
    tmp = jnp.around(result).astype(int)

    mask = jnp.ones_like(tmp)
    mask = mask.at[tmp > cut_off].set(0)
    return mask


print("Forward model setting")

sigma_e = 0.26
ngal = jnp.array([2.00, 2.00, 2.00, 2.00, 2.00])

tmp = jnp.load("nz/nz_lssty1_srd.npy")
print(tmp.shape)
zz = tmp[:, 0]
nz1 = tmp[:, 1]
nz2 = tmp[:, 2]
nz3 = tmp[:, 3]
nz4 = tmp[:, 4]
nz5 = tmp[:, 5]


nz = [nz1, nz2, nz3, nz4, nz5]

nz_shear = [
    jc.redshift.kde_nz(zz, nz[i], bw=0.01, zmax=2.5, gals_per_arcmin2=ngal[i])
    for i in range(5)
]

nbins = int(len(nz_shear) - 1)

# High resolution settings
# Note: this low resolution on the los impacts a tiny bit the cross-correlation signal,
# but it's probably worth it in terms of speed gains
# box_size  = [600., 600., 3500.]     # In Mpc/h [RUN2]
box_size = [400.0, 400.0, 4600.0]  # In Mpc/h [RUN3]
box_shape = [200, 200, 128]  # Number of voxels/particles per side

# Specify the size and resolution of the patch to simulate
field_size = 5.0  # transverse size in degrees [RUN3]
field_npix = 50  # number of pixels per side
pixel_size = field_size * 60 / field_npix
print("Pixel size in arcmin: ", pixel_size)

reso = (field_size / field_npix) * 60  # resolution in arcmin.
ang = 0.0166667 * (reso) * 50  # angle of the fullfield in deg

# Noise covariance
print("Computing filter")
f = filter(field_npix, ang / field_npix * 0.0174533, lowpass)

print("Computing noise covriance matrix")
ret = np.zeros((int(field_npix * field_npix), 50000))

for i in range(0, 50000):
    tmp = np.random.normal(
        np.zeros((50, 50)), sigma_e / np.sqrt(ngal[0] * (ang * 60 / field_npix) ** 2)
    )
    ret[:, i] = fft.ifft2(fft.fft2(tmp) * f).real.flatten()

noisecov = np.cov(ret)
print("Done")

# Generate the forward model given these survey settings
lensing_model = jax.jit(
    make_full_field_model(
        field_size=field_size,
        field_npix=field_npix,
        box_size=box_size,
        box_shape=box_shape,
        method=forward_model,
        density_plane_width=100,
        density_plane_npix=300,
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

    Omega_b = 0.0492
    Omega_c = numpyro.sample("omega_c", dist.Uniform(0.05, 1.0))
    sigma8 = numpyro.sample("sigma8", dist.Uniform(0.1, 2.0))
    h = 0.6726
    w0 = -1
    n_s = 0.9645

    cosmo = jc.Cosmology(
        Omega_c=Omega_c,
        sigma8=sigma8,
        Omega_b=Omega_b,
        Omega_k=0.0,
        h=h,
        n_s=n_s,
        w0=w0,
        wa=0.0,
    )

    # Generate random convergence maps
    convergence_maps, _ = lensing_model(cosmo, nz_shear, initial_conditions)

    reso_rad = ang / field_npix * 0.0174533
    nsub = int(field_npix / 2 + 1)
    i, j = jnp.meshgrid(jnp.arange(nsub), jnp.arange(nsub))
    submatrix = (
        2 * jnp.pi * jnp.sqrt(i**2 + j**2) / reso_rad / jnp.float32(field_npix)
    )

    result = jnp.zeros([field_npix, field_npix])
    result = result.at[0:nsub, 0:nsub].set(submatrix)
    result = result.at[0:nsub, nsub:].set(jnp.fliplr(submatrix[:, 1:-1]))
    result = result.at[nsub:, :].set(jnp.flipud(result[1 : nsub - 1, :]))
    tmp = jnp.around(result).astype(int)

    mask = jnp.ones_like(tmp)
    mask = jnp.where(tmp > lowpass, 0, mask)

    obslp = [
        fft.ifft2(fft.fft2(convergence_maps[i]) * mask).real
        for i in range(len(convergence_maps))
    ]

    observed_maps = [
        numpyro.sample(
            "kappa_%d" % i,
            dist.MultivariateNormal(
                obslp[i].flatten(),
                noisecov + 1e-10 * jnp.eye(int(field_npix * field_npix)),
            ),
        )
        for i in range(len(convergence_maps))
    ]

    return observed_maps


print("Generate fiducial map")
# Create a random realization of a map with fixed cosmology
gen_model = condition(
    model,
    {
        "omega_c": 0.2664,
        "sigma8": 0.831,
    },
)


model_tracer = numpyro.handlers.trace(
    numpyro.handlers.seed(gen_model, jax.random.PRNGKey(1234))
)
model_trace = model_tracer.get_trace()

m_data = jnp.stack(
    [model_trace["kappa_%d" % i]["value"] for i in range(nbins)], axis=-1
).reshape([field_npix, field_npix, nbins])

truth = [0.2664, 0.831]
nb_of_params_to_infer = 2
params_name = ["$\Omega_c$", "$\sigma_8$"]

plt.figure(figsize=(15, 5))

for i in range(nbins):
    plt.subplot(1, nbins, i + 1)
    plt.imshow(m_data[..., i], cmap="cividis")
    plt.title("Bin %d" % (i + 1))
    plt.axis("off")

plt.savefig(f"{PATH_experiment}/fig/check_fiducial_data.png")

print("######## DONE ########")


print("######## DATASET GENERATION AND DATA AUGMENTATION ########")

print("Utils fun")


@jax.jit
@jax.vmap
def augmentation_noise(x, key):
    keys = jax.random.split(key, len(x))
    observed_maps = [
        dist.MultivariateNormal(
            x[i],
            noisecov + 1e-10 * jnp.eye(int(field_npix * field_npix)),
        ).sample(keys[i])
        for i in range(len(x))
    ]
    noisy_map = jnp.stack(observed_maps, axis=-1)
    noisy_map = noisy_map.reshape([field_npix, field_npix, nbins])

    return noisy_map


@jax.jit
@jax.vmap
def get_data(key):
    seeded_model = numpyro.handlers.seed(model, key)
    model_trace = numpyro.handlers.trace(seeded_model).get_trace()
    kmap_obs = jnp.stack(
        [model_trace["kappa_%d" % i]["value"] for i in range(nbins)], axis=-1
    )
    kmap_obs = kmap_obs.reshape([field_npix, field_npix, nbins])

    kmap = jnp.stack(
        [model_trace["kappa_%d" % i]["fn"].mean for i in range(nbins)], axis=0
    )
    theta = jnp.stack(
        [
            model_trace["omega_c"]["value"],
            model_trace["sigma8"]["value"],
        ]
    )
    return theta, kmap_obs, kmap


if os.path.exists(
    f"{PATH_experiment}/datasets/train_dataset_{nb_sim_for_compression}_{nb_sim_for_inference}.pkl"
):
    print("File exists.")

    def pickle_load(path):
        with open(path, "rb") as file:
            return load(file)

    dataset_train = pickle_load(
        f"{PATH_experiment}/train_dataset_{nb_sim_for_compression}_{nb_sim_for_inference}.pkl"
    )

    dataset_test = pickle_load(
        f"{PATH_experiment}/test_dataset_{nb_sim_for_compression}_{nb_sim_for_inference}.pkl"
    )
else:
    print("Dataset does not exist -> lets build it")

    if forward_model == "pm":
        bs = 10
    else:
        bs = 15

    print("Data train genration...")
    params_train = []
    maps_train = []
    masterkey = jax.random.PRNGKey(42)

    for i in tqdm(range(nb_sim_for_compression // bs)):
        key, masterkey = jax.random.split(masterkey)
        params, kmap_obs, kmap = get_data(jax.random.split(key, bs))
        params_train.append(params)
        maps_train.append(kmap)

    params_train = np.stack(params_train).reshape([-1, nb_of_params_to_infer])
    maps_train = np.stack(maps_train).reshape([-1, nbins, field_npix * field_npix])
    dataset_train = {"theta": params_train, "maps": maps_train}

    del maps_train, params_train

    print("Done")

    print("Data test genration...")
    params_test = []
    maps_test_obs = []
    masterkey = jax.random.PRNGKey(20)

    for i in tqdm(range(nb_sim_for_inference // bs)):
        key, masterkey = jax.random.split(masterkey)
        params, kmap_obs, kmap = get_data(jax.random.split(key, bs))
        params_test.append(params)
        maps_test_obs.append(kmap_obs)

    params_test = np.stack(params_test).reshape([-1, nb_of_params_to_infer])
    maps_test_obs = np.stack(maps_test_obs).reshape([-1, field_npix, field_npix, nbins])
    dataset_test = {"theta": params_test, "maps_obs": maps_test_obs}

    del maps_test_obs, params_test

    print("Done")

    print("Saving dataset")

    def pickle_dump(obj, path):
        with open(path, "wb") as file:
            pickle.dump(obj, file, protocol=HIGHEST_PROTOCOL)

    # Save the train dataset
    pickle_dump(
        dataset_train,
        f"{PATH_experiment}/train_dataset_{nb_sim_for_compression}_{nb_sim_for_inference}.pkl",
    )

    # Save the test dataset
    pickle_dump(
        dataset_test,
        f"{PATH_experiment}/test_dataset_{nb_sim_for_compression}_{nb_sim_for_inference}.pkl",
    )

print("Done")

print("######## DONE ########")


print("######## COMPRESSION ########")

print("Build compresso")
bijector_layers_compressor = [128] * 6

bijector_compressor = partial(
    AffineCoupling, layers=bijector_layers_compressor, activation=jax.nn.silu
)

NF_compressor = partial(ConditionalRealNVP, n_layers=4, bijector_fn=bijector_compressor)


class Flow_nd_Compressor(hk.Module):
    def __call__(self, y):
        nvp = NF_compressor(nb_of_params_to_infer)(y)
        return nvp


nf = hk.without_apply_rng(
    hk.transform(lambda theta, y: Flow_nd_Compressor()(y).log_prob(theta).squeeze())
)


# compressor
class CompressorCNN2D(hk.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def __call__(self, x):
        net_x = hk.Conv2D(32, 3, 2)(x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.Conv2D(64, 3, 2)(net_x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.Conv2D(128, 3, 2)(net_x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.AvgPool(16, 8, "SAME")(net_x)
        net_x = hk.Flatten()(net_x)

        net_x = hk.Linear(self.output_dim)(net_x)

        return net_x.squeeze()


compressor = hk.transform_with_state(
    lambda y: CompressorCNN2D(nb_of_params_to_infer)(y)
)


print("Train compressor...")
# init compressor
parameters_resnet, opt_state_compressor = compressor.init(
    jax.random.PRNGKey(0), y=0.5 * jnp.ones([1, field_npix, field_npix, nbins])
)
# init nf
params_nf = nf.init(
    jax.random.PRNGKey(0),
    theta=0.5 * jnp.ones([1, nb_of_params_to_infer]),
    y=0.5 * jnp.ones([1, nb_of_params_to_infer]),
)

parameters_compressor = hk.data_structures.merge(parameters_resnet, params_nf)

del parameters_resnet, params_nf

# define optimizer
lr_scheduler = optax.piecewise_constant_schedule(
    init_value=0.001,
    boundaries_and_scales={
        int(total_steps_train_compressor * 0.1): 0.7,
        int(total_steps_train_compressor * 0.2): 0.7,
        int(total_steps_train_compressor * 0.3): 0.7,
        int(total_steps_train_compressor * 0.4): 0.7,
        int(total_steps_train_compressor * 0.5): 0.7,
        int(total_steps_train_compressor * 0.6): 0.7,
        int(total_steps_train_compressor * 0.7): 0.7,
        int(total_steps_train_compressor * 0.8): 0.7,
        int(total_steps_train_compressor * 0.9): 0.7,
    },
)

optimizer_c = optax.adam(learning_rate=lr_scheduler)
opt_state_c = optimizer_c.init(parameters_compressor)

model_compressor = TrainModel(
    compressor=compressor,
    nf=nf,
    optimizer=optimizer_c,
    loss_name=loss_function_compressor,
)

update = jax.jit(model_compressor.update)

batch_size = 128
masterkey = jax.random.PRNGKey(0)
store_loss = []
loss_train = []
loss_test = []
for batch in tqdm(range(total_steps_train_compressor)):
    key, key2, masterkey = jax.random.split(masterkey, 3)
    inds = np.random.randint(0, len(dataset_train["theta"]), batch_size)
    if not jnp.isnan(dataset_train["theta"][inds]).any():
        data_map = augmentation_noise(
            dataset_train["maps"][inds], jax.random.split(key2, batch_size)
        )
        b_loss, parameters_compressor, opt_state_c, opt_state_compressor = update(
            model_params=parameters_compressor,
            opt_state=opt_state_c,
            theta=dataset_train["theta"][inds],
            x=data_map,
            state_resnet=opt_state_compressor,
        )

        store_loss.append(b_loss)

        if jnp.isnan(b_loss):
            print("NaN Loss")
            break

    if batch % 1000 == 0:
        # save params
        with open(
            f"{PATH_experiment}/save_params/COMPRESSION_params_batch{batch}.pkl",
            "wb",
        ) as fp:
            pickle.dump(parameters_compressor, fp)

        with open(
            f"{PATH_experiment}/save_params/COMPRESSION_opt_state_batch{batch}.pkl",
            "wb",
        ) as fp:
            pickle.dump(opt_state_compressor, fp)

        inds2 = np.random.randint(0, len(dataset_test["theta"]), batch_size)
        key, masterkey = jax.random.split(masterkey)
        data_map_test = dataset_test["maps_obs"][inds2]
        b_loss_test, _, _, _ = update(
            model_params=parameters_compressor,
            opt_state=opt_state_c,
            theta=dataset_test["theta"][inds2],
            x=data_map_test,
            state_resnet=opt_state_compressor,
        )

        loss_train.append(b_loss)
        loss_test.append(b_loss_test)

        jnp.save(f"{PATH_experiment}/save_params/loss_train.npy", loss_train)
        jnp.save(f"{PATH_experiment}/save_params/loss_test.npy", loss_test)


with open(f"{PATH_experiment}/save_params/COMPRESSIONS_params_FINAL.pkl", "wb") as fp:
    pickle.dump(parameters_compressor, fp)

with open(f"{PATH_experiment}/save_params/COMPRESSION_opt_state_FINAL.pkl", "wb") as fp:
    pickle.dump(opt_state_compressor, fp)

print("Done")

print("Intermediate check plots")
plt.figure()
plt.plot(store_loss[100:])
plt.savefig(f"{PATH_experiment}/fig/COMPRESSION_loss.png")

plt.figure()
plt.plot(loss_train[1:], label="train loss")
plt.plot(loss_test[1:], label="test loss")
plt.legend()
plt.title("Batch Loss")
plt.savefig(f"{PATH_experiment}/fig/COMPRESSION_loss_train_and_test.png")


compressed_dataset_train, _ = compressor.apply(
    parameters_compressor,
    opt_state_compressor,
    None,
    augmentation_noise(dataset_train["maps"][:1000], jax.random.split(key2, 1000)),
)

compressed_dataset_test, _ = compressor.apply(
    parameters_compressor,
    opt_state_compressor,
    None,
    dataset_test["maps_obs"][:1000].reshape([-1, field_npix, field_npix, nbins]),
)


plt.figure(figsize=(35, 5))
for i in range(nb_of_params_to_infer):
    plt.subplot(1, nbins, i + 1)
    plt.scatter(
        dataset_test["theta"][:1000, i],
        compressed_dataset_test[:, i],
        label="on test data",
    )
    plt.scatter(
        dataset_train["theta"][:1000, i],
        compressed_dataset_train[:, i],
        label="on train data",
    )
    plt.xlabel("truth")
    plt.ylabel("prediction")
    plt.title(params_name[i])
    plt.legend()
    plt.savefig(f"{PATH_experiment}/fig/COMPRESSION_regression_plot.png")

y, _ = compressor.apply(
    parameters_compressor,
    opt_state_compressor,
    None,
    m_data.reshape([1, field_npix, field_npix, nbins]),
)

nvp_sample_nd = hk.transform(
    lambda x: Flow_nd_Compressor()(x).sample(100000, seed=hk.next_rng_key())
)
sample_nd = nvp_sample_nd.apply(
    parameters_compressor,
    rng=jax.random.PRNGKey(43),
    x=y * jnp.ones([100000, nb_of_params_to_infer]),
)
idx = jnp.where(jnp.isnan(sample_nd))[0]
sample_nd = jnp.delete(sample_nd, idx, axis=0)

plt.figure()
c = ChainConsumer()
c.add_chain(sample_nd, name="SBI")
fig = c.plotter.plot(figsize=1.2, truth=truth)
plt.savefig(f"{PATH_experiment}/fig/COMPRESSION_contour_plot.png")

print("######## DONE ########")


print("######## INFERENCE ########")
print("Create compressed dataset..")

data_compressed, _ = compressor.apply(
    parameters_compressor,
    opt_state_compressor,
    None,
    dataset_test["maps_obs"].reshape([-1, field_npix, field_npix, nbins]),
)

train_dataset_compressed = {"theta": dataset_test["theta"], "x": data_compressed}


data_map = augmentation_noise(
    dataset_train["maps"][:2000], jax.random.split(jax.random.PRNGKey(760), 2000)
)

data_compressed, _ = compressor.apply(
    parameters_compressor,
    opt_state_compressor,
    None,
    data_map.reshape([-1, field_npix, field_npix, nbins]),
)

test_dataset_compressed = {"theta": dataset_train["theta"][:2000], "x": data_compressed}

print("Done")

print("** Create NDE **")
# Create neural density estimator (NDE) to approximate p(theta | y)

summary_stat_dim = nb_of_params_to_infer
nb_params_to_infer = nb_of_params_to_infer
batch_size = 128


# Affine bijection used in the RealNVP coupling
bijector_ff = partial(AffineCoupling, layers=[128] * 2, activation=jax.nn.silu)

# Normalizing Flow with 4 RealNVP coupling layers
NF_ff = partial(ConditionalRealNVP, n_layers=4, bijector_fn=bijector_ff)


# log probability of the NDE
nf_logp_ff = hk.without_apply_rng(
    hk.transform(
        lambda theta, y: NF_ff(nb_params_to_infer)(y).log_prob(theta).squeeze()
    )
)

# sampling of the NDE
nf_sample_ff = hk.transform(
    lambda y: NF_ff(nb_params_to_infer)(y).sample(100_000, seed=hk.next_rng_key())
)

print("** Utils function **")


# negative log likelihood
def loss_nll(params, mu, batch):
    return -jnp.mean(nf_logp_ff.apply(params, mu, batch))


@jax.jit
def update(params, opt_state, mu, batch):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_nll)(
        params,
        mu,
        batch,
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return loss, new_params, new_opt_state


print("** Training **")
# init nf params
params_ff = nf_logp_ff.init(
    jax.random.PRNGKey(42),
    0.5 * jnp.zeros([1, summary_stat_dim]),
    0.5 * jnp.zeros([1, nb_params_to_infer]),
)

total_steps = total_steps_train_nde
nb_steps = total_steps - total_steps * 0.2

lr_scheduler = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=nb_steps // 50,
    decay_rate=0.9,
    end_value=1e-6,
)

# define optimizer
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state_ff = optimizer.init(params_ff)


# training

batch_size = 128
batch_loss = []
loss_train = []
loss_test = []
pbar = tqdm(range(total_steps))

for batch in pbar:
    inds = np.random.randint(0, len(train_dataset_compressed["theta"]), batch_size)

    l, params_ff, opt_state_ff = update(
        params_ff,
        opt_state_ff,
        train_dataset_compressed["theta"][inds],
        train_dataset_compressed["x"][inds],
    )
    batch_loss.append(l)
    pbar.set_description(f"loss {l:.3f}")

    if batch % 1_000 == 0:
        # save params
        with open(
            f"{PATH_experiment}/save_params/INFERENCE_params_nd_flow_batch{batch}.pkl",
            "wb",
        ) as fp:
            pickle.dump(params_ff, fp)

        inds2 = np.random.randint(0, len(test_dataset_compressed["theta"]), batch_size)
        b_loss_test, _, _ = update(
            params_ff,
            opt_state_ff,
            test_dataset_compressed["theta"][inds2],
            test_dataset_compressed["x"][inds2],
        )

        loss_train.append(l)
        loss_test.append(b_loss_test)

with open(
    f"{PATH_experiment}/save_params/INFERENCE_params_nd_flow_FINAL.pkl", "wb"
) as fp:
    pickle.dump(params_ff, fp)

print("** Intermediate check plots **")
plt.figure()
plt.plot(batch_loss[100:])
plt.savefig(f"{PATH_experiment}/fig/INFERENCE_loss.png")

plt.figure()
plt.plot(loss_train[1:], label="train loss")
plt.plot(loss_test[1:], label="test loss")
plt.legend()
plt.title("Batch Loss")
plt.savefig(f"{PATH_experiment}/fig/INFERENCE_loss_train_and_test.png")


observed_map_compressed, _ = compressor.apply(
    parameters_compressor,
    opt_state_compressor,
    None,
    m_data.reshape([1, field_npix, field_npix, nbins]),
)

posterior_ff = nf_sample_ff.apply(
    params_ff,
    rng=jax.random.PRNGKey(70),
    y=observed_map_compressed * jnp.ones([100_000, summary_stat_dim]),
)


c = ChainConsumer()

c.add_chain(
    posterior_ff, shade_alpha=0.5, name="Implicit full-field", parameters=params_name
)

fig = c.plotter.plot(figsize=1.0, truth=truth)
plt.savefig(f"{PATH_experiment}/fig/INFERENCE_contour_plots.png")

print("######## DONE ########")


print("######## SAVING SBI RESULTS ########")

jnp.save(f"{PATH_experiment}/FINAL_RESULTS_implicit_inference_lpt.npy", posterior_ff)

print("######## DONE ########")
