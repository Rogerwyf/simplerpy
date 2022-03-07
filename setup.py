import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simplerpy",
    version="1.0.0",
    author=["Roger Wang", "Michelle Hsieh", "Regina-Mae Dominguez"],
    author_email=["rogerwyf@uw.edu", "mh808@uw.edu", "rmvd@uw.edu"],
    description="rpy2 made easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rogerwyf/simplerpy",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
