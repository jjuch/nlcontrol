"""
Nonlinear control toolbox, to work with symbolic nonlinear systems in a control context.

See:
https://github.ugent.be/jjuchem/nlcontrol
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get version
exec(open('nlcontrol/version.py').read())

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

DOCSTRING = long_description.split('\n')

setup(
    name='nlcontrol',
    version=__version__,
    description='A toolbox to simulate and analyse nonlinear systems in a control context.',
    long_description_content_type='text/markdown',
    long_description=DOCSTRING[3],
    keywords='control systems nonlinear simulation',
    url='https://github.com/jjuch/nlcontrol',
    author='Jasper Juchem',
    author_email='Jasper.Juchem@UGent.be',
    classifiers=[
        'Development States :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience ::  Education',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.17.0',
        'scipy>=1.3.0',
        'sympy==1.4.0',
        'control>=0.8.1',
        'simupy>=1.0.0',
        'matplotlib>=3.1.1'
    ]
)