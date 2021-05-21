import setuptools   # type: ignore

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="gazeclassify",
    version="0.2",
    author="Daniel Müller",
    author_email="daniel@science.football",
    description="Algorithmic eye-tracking analysis",
    install_requires=[
        'tensorflow>=2.5.0',
        'pixellib>=0.6.1',
        'ffmpeg-python>=0.2.0',
        'opencv-python>=4.5.1.48',
        'Pillow>=8.1.0',
        'tqdm>=4.59.0'
    ],
    tests_require=['pytest'],
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/footballdaniel/gazeclassify",
    packages=setuptools.find_packages(exclude="*tutorials"),
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
    package_data={'gazeclassify': ['example_data/*']},
)
