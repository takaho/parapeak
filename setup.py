from setuptools import setup, find_packages

setup(
    name='parapeak',
    version='0.1.0',
    description='ChIP-seq peak caller with parallel computation and GC correction',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'pysam>=0.21',
        'numpy>=1.24',
        'scipy>=1.11',
        'numba>=0.58',
    ],
    entry_points={
        'console_scripts': [
            'parapeak=parapeak.__main__:main',
        ],
    },
)
