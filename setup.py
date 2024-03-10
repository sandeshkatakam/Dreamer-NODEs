import setuptools
import pathlib


setuptools.setup(
    name='dreamer-nodes',
    version='0.0.1',
    description='Implementing Neural ODE based World Models using Dreamer Agents',
    url='http://github.com/sandeshkatakam/Dreamer-NODEs',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['dreamer-nodes', 'dreamer-nodes.dreamer', 'dreamer-nodes.NODE', 'dreamer-nodes.utils', 'dreamer-nodes.dataset'],
    package_data={'dreamer-nodes': ['configs.json']},
    entry_points={'console_scripts': ['dreamer-nodes=dreamer-nodes.train:main']},
    install_requires=[
        'gym[atari]', 'atari_py', 'crafter', 'dm_control', 'ruamel.yaml',
        'jax', 'tensorflow_probability'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
