from setuptools import setup, find_packages
from typing import List
import os

# Constant for the local editable install trigger
HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements from the requirements.txt file.
    It also removes '-e .' if present to prevent circular installation loops.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Filter out comments, empty lines, and strip newline characters
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="demand-prediction-system",
    version="0.1.0",
    author="Monower Hossen",
    author_email="monower.cse@gmail.com",
    description="An industry-level end-to-end ML pipeline for railway ticket demand prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Monower-Hossen/Bangladesh-Railway-Ticket-Demand-Prediction-System",
    
    # Automatically finds all packages (folders with __init__.py)
    packages=find_packages(), 
    
    # Dynamically loads requirements from your requirements.txt file
    install_requires=get_requirements('requirements.txt'),
    
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)