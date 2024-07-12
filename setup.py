from setuptools import setup

setup(
    name='jax_lensing',
    version='1.0',
    description='A small package for comparing full-field lensing constraints using JAX',
    author='Eiffl, Justinezgh, yomori, chihway',
    packages=['jax_lensing'],
    package_dir={"jax_lensing": "jax_lensing"},
    package_data={
        "jax_lensing": ["nz/*.npy"],
    },
    install_requires=[
        'jax-cosmo',
        'numpyro',
        'lenstools',
        'chainconsumer',
        'cmasher',
        'JaxPM @ git+https://github.com/DifferentiableUniverseInitiative/JaxPM.git',
        'sbi_lens @ git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git'
    ]
)


