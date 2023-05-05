from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("requirements_doc.txt") as f:
    docs_requires = f.read().splitlines()
    
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name="sdmbc",
    version="0.1.1",
    description="Sub-Daily Multivariate Bias Correction (SDMBC)",
    author="Youngil Kim",
    author_email="youngil.kim@student.unsw.edu.au",
    packages=find_packages(),
    license='LICENSE',
    long_description=long_description,
    long_description_content_type='text/markdown',
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