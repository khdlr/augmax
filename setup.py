# Copyright 2021 Konrad Heidler
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup, find_packages

_version_dict = dict()
with open('augmax/version.py') as f:
    exec(f.read(), _version_dict)
__version__ = _version_dict['__version__']

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='augmax',
    version=__version__,
    description='Efficiently Composable Data Augmentation on the GPU with Jax',
    author='Konrad Heidler',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires = [
        'jax>=0.1',
        'einops>=0.3'
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/khdlr/augmax',
    license='',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License'
    ]
)

