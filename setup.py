from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements=f.read().splitlines()

setup(
    name='llm_lsp',
    version='1.0.0',
    author='Niklas Loeser',
    description='Description of my package',
    packages=["llm_lsp"],    
    install_requires=requirements,
)