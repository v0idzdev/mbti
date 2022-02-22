from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="matthewflegg-mbti",
    version="0.0.1",
    description="Predicting MBTI types based on Twitter posts",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/matthewflegg/mbti",
    author='Matthew Flegg',
    author_email="matthewflegg@outlook.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="MBTI, LSTM, machine learning, neural network, BERT, keras",
    package_dir={"": "src"},
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "keras",
        "numpy",
        "pandas",
        "transformers"
    ]
)