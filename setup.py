from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os
import subprocess
import sys
from distutils.unixccompiler import UnixCCompiler

# Add support for CUDA source files
class CUDA_build_ext(build_ext):
    def build_extensions(self):
        cuda_root = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        
        # Process each extension
        for ext in self.extensions:
            cuda_sources = []
            cpp_sources = []
            
            # Separate CUDA sources from C++ sources
            for source in ext.sources:
                if source.endswith('.cu'):
                    cuda_sources.append(source)
                else:
                    cpp_sources.append(source)
            
            # Update extension with only C++ sources for now
            ext.sources = cpp_sources
            
            # Compile CUDA files to object files
            if cuda_sources:
                # Create obj directory if needed
                obj_dir = os.path.join(os.path.dirname(self.get_ext_fullpath(ext.name)), 'obj')
                os.makedirs(obj_dir, exist_ok=True)
                
                # Compile each CUDA file
                objects = []
                for cuda_src in cuda_sources:
                    # Determine object file path
                    obj_base = os.path.splitext(os.path.basename(cuda_src))[0] + '.o'
                    obj_file = os.path.join(obj_dir, obj_base)
                    
                    # Define nvcc command
                    nvcc_cmd = [
                        f'{cuda_root}/bin/nvcc',
                        '-std=c++17',
                        '-O3',
                        '-Xcompiler',
                        '-fPIC',
                        f'-I{cuda_root}/include',
                        f'-I{pybind11.get_include()}',
                        f'-I{pybind11.get_include(user=True)}',
                        '-I./IBATensor',
                        '-c',
                        cuda_src,
                        '-o',
                        obj_file
                    ]
                    
                    # Execute nvcc command
                    self.announce(f'Compiling CUDA file: {cuda_src}', level=2)
                    subprocess.check_call(nvcc_cmd)
                    objects.append(obj_file)
                
                # Add object files to extra objects for linking
                ext.extra_objects = objects + (ext.extra_objects or [])
                
                # Add CUDA libraries
                ext.libraries.append('cudart')
                ext.library_dirs.append(f'{cuda_root}/lib64')
                ext.runtime_library_dirs.append(f'{cuda_root}/lib64')
        
        # Now build the extensions with C++ compiler
        super().build_extensions()


ext_modules = [
    Extension(
        'IBALearn.neural_network.ibatensor',
        [
            'IBATensor/IBATensor.cpp', 
            'IBATensor/iba_tensor_binding.cpp',
            'IBATensor/IBADeviceData/CPUData.cpp',
            'IBATensor/IBADeviceData/CudaData.cu'  
        ],
        include_dirs=[
            pybind11.get_include(), 
            pybind11.get_include(user=True),
            'IBATensor'
        ],
        libraries=[],
        library_dirs=[],
        extra_compile_args=['-std=c++17'],
        language='c++',
    ),
]

setup(
    name='IBALearn',
    version='0.1.0',
    author='IBA',
    ext_modules=ext_modules,
    cmdclass={'build_ext': CUDA_build_ext},
    install_requires=[
        'pandas>=2.2.3',
        'statsmodels>=0.14.4',
        'numpy>=2.2.3',
        'scipy>=1.15.2',
        'pybind11>=2.6.0'
    ],
    zip_safe=False,
)