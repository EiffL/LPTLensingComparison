import dataclasses

import jax_cosmo as jc
import numpy as np
from numpyro import distributions as dist
from sbi_lens.simulator.redshift import subdivide
from scipy.stats import norm


@dataclasses.dataclass
class config:
    """Configuration for the LPT simulation."""

    # LPT simulation parameters
    box_shape: list[int]  # Number of pixels in the box
    box_size: list[float]  # Physical size of the box, in Mpc/h
    # Lensing simulation parameters
    field_size: float  # Size of the lensing field, in degrees
    field_npix: int  # Number of pixels in the lensing field
    nz_shear: list[float]  # Redshift bins for the lensing source galaxies
    sigma_e: float  # Intrinsic ellipticity dispersion
    # Cosmological parameters
    fiducial_cosmology: jc.Cosmology  # Fiducial cosmology
    priors: dict[
        str, dist.Distribution
    ]  # Priors for the cosmological parameters to be sampled
    ell_max: int  # Maximum ell for the 2pt correlation function


# Porqueres 2023 configuration
# from https://arxiv.org/abs/2304.04785
Porqueres_2023 = config(
    box_shape=[64, 64, 128],
    box_size=[1000.0, 1000.0, 4500.0],
    field_size=16.0,
    field_npix=64,
    fiducial_cosmology=jc.Cosmology(
        Omega_c=0.315 - 0.049,
        Omega_b=0.049,
        h=0.677,
        n_s=0.9624,
        sigma8=0.8,
        w0=-1.0,
        Omega_k=0.0,
        wa=0.0,
    ),
    priors={
        "Omega_c": dist.Uniform(0.2 - 0.049, 0.7 - 0.049),
        "sigma8": dist.Uniform(0.5, 1.6),
    },
    nz_shear=[
        jc.redshift.kde_nz(
            np.linspace(0, 2.5, 1000),
            norm.pdf(np.linspace(0, 2.5, 1000), loc=z_center, scale=0.12),
            bw=0.01,
            zmax=2.5,
            gals_per_arcmin2=g,
        )
        for z_center, g in zip([0.5, 1.0, 1.5, 2.0], [7, 8.5, 7.5, 7])
    ],
    sigma_e=0.3,
    ell_max=1000,
)


# LSST Y10 SRD configuration
LSST_Y10 = config(
    box_shape=[300, 300, 128],
    box_size=[400.0, 400.0, 4000.0],
    field_size=5.0,
    field_npix=60,
    fiducial_cosmology=jc.Planck15(
        Omega_c=0.2664, Omega_b=0.0492, sigma8=0.831, h=0.6727, n_s=0.9645
    ),
    priors={
        "Omega_c": dist.Normal(0.2664, 0.2),
        "Omega_b": dist.Normal(0.0492, 0.006),
        "sigma8": dist.Normal(0.831, 0.14),
        "h": dist.Normal(0.6727, 0.063),
        "n_s": dist.Normal(0.9645, 0.08),
        "w0": dist.Normal(-1.0, 0.8),
    },
    nz_shear=subdivide(
        jc.redshift.smail_nz(2, 0.68, 0.11, gals_per_arcmin2=27, zmax=2.6),
        nbins=5,
        zmax=2.5,
    ),
    sigma_e=0.26,
    ell_max=3000,
)

# DES Y3 configuration
# n(z) extracted from 2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits
# Number density extracted from Table I of 2105.13541
# Assuming constant sig_e taking straight-up average 0.267=(0.243+0.262+0.259+0.301)/4
nzs = np.load("nz/nz_desy3_metacal.npy")
DES_Y3 = config(
    box_shape=[300, 300, 128],
    box_size=[400.0, 400.0, 4000.0],
    field_size=5.0,
    field_npix=60,
    fiducial_cosmology=jc.Planck15(
        Omega_c=0.2664, Omega_b=0.0492, sigma8=0.831, h=0.6727, n_s=0.9645
    ),
    priors={
        "Omega_c": dist.Normal(0.2664, 0.2),
        "Omega_b": dist.Normal(0.0492, 0.006),
        "sigma8": dist.Normal(0.831, 0.14),
        "h": dist.Normal(0.6727, 0.063),
        "n_s": dist.Normal(0.9645, 0.08),
        "w0": dist.Normal(-1.0, 0.8),
    },
    nz_shear=[
        jc.redshift.kde_nz(
            nzs[:, 0], nzs[:, zi + 1], bw=0.01, zmax=3.0, gals_per_arcmin2=g
        )
        for zi, g in enumerate([1.476, 1.479, 1.484, 1.461])
    ],
    sigma_e=0.267,
    ell_max=3000,
)
