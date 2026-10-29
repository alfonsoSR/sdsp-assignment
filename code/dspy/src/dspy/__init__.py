from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dspy")
except PackageNotFoundError:
    pass
