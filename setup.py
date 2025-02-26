from setuptools import setup, find_packages

setup(
    name='IBALearn',
    version='0.1.0',
    author='IBA',
    package_dir={'':'IBALearn'},
    install_requires=[
        'pandas>=2.2.3',
        'statsmodels>=0.14.4',
        'numpy>=2.2.3',
        'scipy>=1.15.2'
    ]
)