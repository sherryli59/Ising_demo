import os
import io
import glob
import sys
from setuptools import setup, find_packages, Command


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

setup(
    name="ising",
    version='1.0',
    packages=find_packages("."),
    package_dir={"": "."},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "celluloid",
        "pytest",
    ],
    cmdclass={
        'clean': CleanCommand,
    },
    include_package_data=True,
    license="MIT",
)
