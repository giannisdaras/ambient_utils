import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]

setuptools.setup(
    name="ambient_utils",
    version="0.0.1",
    author="giannisdaras",
    author_email="daras.giannhs@gmail.com",
    description="Utility functions for learning generative models from corrupted data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giannisdaras/ambient_utils",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)", 
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.6',
)
