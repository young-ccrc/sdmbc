from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("requirements_doc.txt") as f:
    docs_requires = f.read().splitlines()
    
setup(
    name="sdmbc",
    version="0.1",
    description="Sub-Daily Multivariate Bias Correction (SDMBC)",
    author="Youngil Kim",
    author_email="youngil.kim@student.unsw.edu.au",
    packages=find_packages(),
    license='LICENSE',
    long_description=open('README.md').read(),
    install_requires=install_requires,
    extras_require={
      'documentation': docs_requires
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    package_data={
        "sdmbc": ["main_biascorrection.exe"],
    },
)