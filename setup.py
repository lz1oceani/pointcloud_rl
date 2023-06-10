from setuptools import setup, find_packages


def get_version():
    version_file = "pyrl/version.py"
    with open(version_file, "r", encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


setup(
    name="pyrl",
    version=get_version(),
    packages=find_packages(),
    author="",
    zip_safe=False,
)
