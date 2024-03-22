import os
import re
import subprocess
from setuptools import setup, find_packages


_mydir = os.path.dirname(__file__)

# parse version number from splatsetup/__init__.py
version_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
with open(os.path.join(_mydir, "splatsetup", "__init__.py")) as f:
    content = f.read()
re_search = re.search(version_re, content, re.M)
if re_search:
    version_str = re_search.group(1)
else:
    raise RuntimeError("Could not parse version string from __init__.py")

# add the git hash to version_str
if os.path.exists(os.path.join(_mydir, ".git")):
    git_hash = subprocess.check_output(
        "git rev-parse --verify --short HEAD", cwd=_mydir, text=True, shell=True
    ).strip()
    git_version_str = f"v{version_str}"
    tags = subprocess.check_output("git tag", cwd=_mydir, text=True, shell=True)
    if git_version_str not in tags:
        subprocess.check_output(
            f"git tag -a {git_version_str} {git_hash} -m 'tagged by setup.py to {version_str}'",
            cwd=_mydir,
            text=True,
            shell=True,
        )
    version_str = f"{version_str}+git.{git_hash}"

setup(
    name="splatsetup",
    version=version_str,
    description="Utilities to write SPLAT .control files from a .toml template. Bokeh application for creating .control files",
    url="https://github.com/rocheseb/splat_setup",
    author="Sebastien Roche",
    author_email="sroche@g.harvard.edu",
    license="MIT",
    packages=["splatsetup"],
    package_dir={"splatsetup": "splatsetup"},
    package_data={"splatsetup": ["inputs/*.toml", "inputs/*.control"]},
    entry_points={
        "console_scripts": [
            "ctrlsetup=splatsetup.conttrol_setup:main",
            "toml2ctrl=splatsetup.toml_to_control:main",
            "updtoml=splatsetup.update_toml:main",
            "splatsetup=splatsetup.main:main",
        ],
    },
    zip_safe=False,
    include_package_data=True,
    install_requires=["bokeh>=3.10.0", "toml"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
