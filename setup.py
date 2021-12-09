from setuptools import find_packages, setup
import free

setup(
    name="free_energy",
    packages=find_packages(
        include=["free"],
    ),
    include_package_data=True,
    version=free.__version__,
    description="EBM models",
    author="Nikita Balagansky",
)