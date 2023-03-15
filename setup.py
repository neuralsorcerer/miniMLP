import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="miniMLP",
    version="0.0.1",
    author="Soumyadip Sarkar",
    author_email="soumyadipsarkar@outlook.com",
    description="Implementation of very small scale Neural Network from scratch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/papaaannn/miniMLP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
