from setuptools import setup, find_packages
from typing import List

def get_requirements(file:str)->list[str]:
    requirements=[]
    with open(file) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements





setup(
    name='ML-Project-Template',
    version='0.0.1',
    author='Abdul',
    author_email="",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
    )