from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:  
    requirements = []
    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name="Lungs X-rays Classification",
    version="0.0.1",
    author="Akshay Redekar",
    author_email="akshayredekar4441@gmail.com",
    install_requires=get_requirements("requirements.txt"),  # Pass file path
    packages=find_packages()
)
