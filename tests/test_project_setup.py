"""Smoke tests for the ShopSense project foundation."""

from pathlib import Path

import src
from src import preprocessing


def test_project_package_imports_from_repository_root() -> None:
    """The project package and existing preprocessing module are importable."""
    assert Path(src.__file__).is_file()
    assert Path(preprocessing.__file__).is_file()


def test_project_configuration_and_test_directory_exist() -> None:
    """The repository has the minimal configuration expected by contributors."""
    project_root = Path(__file__).resolve().parents[1]

    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "requirements.txt").is_file()
    assert (project_root / "tests").is_dir()
