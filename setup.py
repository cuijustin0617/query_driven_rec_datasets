from setuptools import setup, find_packages

setup(
    name="per_query_labeling",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai",
        "google-generativeai",
        "tqdm",
        "python-dotenv",
        "pathlib",
    ],
) 