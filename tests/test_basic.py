"""Basic tests to ensure the package is properly installed.
TODO: develop more tests.
"""

import importlib


def test_package_import() -> None:
    """Test that the package can be imported."""
    # This will be replaced by cookiecutter with the actual package name
    package_name = "valuation_curator"
    module = importlib.import_module(package_name)
    assert module is not None


def test_version_exists() -> None:
    """Test that the package has a version attribute."""
    package_name = "valuation_curator"
    module = importlib.import_module(package_name)
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)
