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
    python_requires='>=3.8, <3.12',
    install_requires=[
        "numpy>=2.0.1",
        "scipy>=1.2.1",
        "matplotlib>=3.10.0",
        "pandas>=2.2.3",
        "bayes_opt>=2.0.3",
        "pycma>=4.0.0",
    ],
)

