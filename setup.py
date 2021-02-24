import setuptools
import gazeclassify

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="gazeclassify",
    version=gazeclassify.__version__,
    author="Daniel Müller",
    author_email="daniel@science.football",
    description="Algorithmic eye-tracking analysis",
    install_requires=[],
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/footballdaniel/gazeclassify",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
)
