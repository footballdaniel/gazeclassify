import setuptools  # type: ignore

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="gazeclassify",
    version="0.9.2",
    author="Daniel Müller",
    author_email="daniel@science.football",
    description="Algorithmic eye-tracking analysis",
    install_requires=[
        'tensorflow>=2.5.0',
        'ffmpeg-python>=0.2.0',
        'opencv-python>=4.1.2',
        'pixellib>=0.6.6',
        'tqdm>=4.60.0',
        'moviepy>=1.0.3',
        'tabulate>=0.8.9',
        'pandas>=1.2.5'
    ],
    tests_require=['pytest'],
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/footballdaniel/gazeclassify",
    packages=setuptools.find_packages(exclude=['*.tutorials']),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
    package_data={'gazeclassify': ['example_data/trial/*']},
)
