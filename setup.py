from setuptools import setup, find_packages

# Read the requirements from `requirements.txt`
def load_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()

setup(
    name="Electrogenomics",
    version="0.1.0",
    description="A package for ecological deep learning models",
    author="Dieter Henrik Heiland",
    python_requires='==3.9',
    packages=find_packages(),
    install_requires=load_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
)
