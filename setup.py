import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="black_box_optimizer", 
    version="0.1.0",
    author="Katherine P. Bronstein",
    author_email="bronstek@oregonstate.edu",
    description="A modular black box optimizer package for experimental optimization, including methods such as GA, CMA-ES, CMA-ES-GI, Bayesian Optimization, and Actor-Critic RL.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kbronstein56/Black-Box-Optimizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",  
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.16.2",
        "scipy>=1.2.1",
        "matplotlib>=3.0.3",
        "pandas>=1.0.0",
        "bayes_opt>=1.2.0",
        "pycma>=0.9.0",
    ],
)

