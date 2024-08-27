from setuptools import setup

with open("requirements.txt") as f:
    REQUIREMENTS = [x.strip() for x in f.readlines()]

setup(
    name="relic",
    packages=["relic"],
    install_requires=REQUIREMENTS,
    extras_require={"dev": ["pre-commit"]},
    version="0.1",
)
