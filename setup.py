import setuptools
from pathlib import Path


# Reading the long description from README.md
def read_long_description():
    try:
        return Path("README.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "A description of AcademicRAG is currently unavailable."


# Retrieving metadata from __init__.py
def retrieve_metadata():
    vars2find = ["__author__", "__version__", "__url__"]
    vars2readme = {}
    try:
        with open("./academicrag/__init__.py") as f:
            for line in f.readlines():
                for v in vars2find:
                    if line.startswith(v):
                        line = (
                            line.replace(" ", "")
                            .replace('"', "")
                            .replace("'", "")
                            .strip()
                        )
                        vars2readme[v] = line.split("=")[1]
    except FileNotFoundError:
        raise FileNotFoundError("Metadata file './academicrag/__init__.py' not found.")

    # Checking if all required variables are found
    missing_vars = [v for v in vars2find if v not in vars2readme]
    if missing_vars:
        raise ValueError(
            f"Missing required metadata variables in __init__.py: {missing_vars}"
        )

    return vars2readme


# Reading dependencies from requirements.txt
def read_requirements():
    deps = []
    try:
        with open("./requirements.txt") as f:
            deps = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(
            "Warning: 'requirements.txt' not found. No dependencies will be installed."
        )
    return deps


def read_api_requirements():
    api_deps = []
    try:
        with open("./academicrag/api/requirements.txt") as f:
            api_deps = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: API requirements.txt not found.")
    return api_deps


def read_extra_requirements():
    api_deps = []
    try:
        with open("./academicrag/tools/academicrag_visualizer/requirements.txt") as f:
            api_deps = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: API requirements.txt not found.")
    return api_deps


metadata = retrieve_metadata()
long_description = read_long_description()
requirements = read_requirements()

setuptools.setup(
    name="academicrag",
    url=metadata["__url__"],
    version=metadata["__version__"],
    author=metadata["__author__"],
    description="AcademicRAG: a versatile GraphRAG-based framework for semantic search and discovery in academic contexts.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        exclude=("tests*", "docs*")
    ),  # Automatically find packages
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,  # Includes non-code files from MANIFEST.in
    project_urls={  # Additional project metadata
        "Documentation": metadata.get("__url__", ""),
        "Source": metadata.get("__url__", ""),
        "Tracker": f"{metadata.get('__url__', '')}/issues"
        if metadata.get("__url__")
        else "",
    },
    extras_require={
        "api": read_api_requirements(),  # API requirements as optional
        "tools": read_extra_requirements(),  # API requirements as optional
    },
    entry_points={
        "console_scripts": [
            "academicrag-server=academicrag.api.academicrag_server:main [api]",
            "academicrag-gunicorn=academicrag.api.run_with_gunicorn:main [api]",
            "academicrag-viewer=academicrag.tools.academicrag_visualizer.graph_visualizer:main [tools]",
        ],
    },
)
