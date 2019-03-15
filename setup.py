# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tideturb',
    version='0.1.0',
    description='A model for simulating a tide-influenced turibidity current in a submarine canyon',
    long_description=readme,
    author='Hajime Naruse',
    author_email='naruse@kueps.kyoto-u.ac.jp',
    url='https://github.com/narusehajime/tideturb',
    license=license,
    install_requires=['numpy', 'matplotlib'],
    packages=find_packages(exclude=('tests', 'docs'))
)

