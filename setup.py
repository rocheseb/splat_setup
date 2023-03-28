from setuptools import setup

setup(
    name="splatsetup",
    version="1.0.0",
    description="Utilities to write SPLAT .control files from a .toml template. Bokeh application for creating .control files",
    url="https://github.com/rocheseb/splat_setup",
    author="Sebastien Roche",
    author_email="sroche@g.harvard.edu",
    license="MIT",
    packages=["splatsetup"],
    package_dir={"splatsetup": "splatsetup"},
    entry_points={
        "console_scripts": [
            "ctrlsetup=splatsetup.conttrol_setup:main",
            "toml2ctrl=splatsetup.toml_to_control:main",
            "splatsetup=splatsetup.main:main",
        ],
    },
    zip_safe=False,
    install_requires=["bokeh==3.1.0", "toml"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.10.9",
)
