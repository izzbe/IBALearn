from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'IBALearn.neural_network.ibatensor',
        ['IBATensor/IBATensor.cpp', 'IBATensor/IBATensor_binding.cpp'],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
        language='c++',
        extra_compile_args=['-std=c++17'], 
    ),
]

setup(
    name='IBALearn',
    version='0.1.0',
    author='IBA',
    ext_modules=ext_modules,
    install_requires=[
        'pandas>=2.2.3',
        'statsmodels>=0.14.4',
        'numpy>=2.2.3',
        'scipy>=1.15.2',
        'pybind11>=2.6.0'
    ],
    zip_safe=False,
)
