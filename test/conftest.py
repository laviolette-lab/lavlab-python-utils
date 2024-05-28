# pylint: skip-file
# type: ignore
import importlib.resources
from importlib.metadata import EntryPoint, EntryPoints

import pytest
import toml

import lavlab


def load_entry_points_from_toml(toml_path, group):
    """
    Load entry points from a pyproject.toml file for a specified group using importlib.metadata classes.

    Args:
    toml_path (str): Path to the pyproject.toml file.
    group (str): The entry point group to load.

    Returns:
    EntryPoints: An EntryPoints object containing all entry points for the specified group.
    """
    with open(toml_path, "r", encoding="utf-8") as file:
        data = toml.load(file)

    entry_points_data = data.get("project", {}).get("entry-points", {})
    group_data = entry_points_data.get(group, {})
    entry_points = []
    for name, value in group_data.items():
        module = value.split(":")[0]
        entry_point = EntryPoint(name, value, module)
        entry_points.append(entry_point)

    return EntryPoints(entry_points)


@pytest.fixture(scope="session", autouse=True)
def mock_entrypoints():

    lavlab.ctx = lavlab.UtilContext(
        default_system_file=importlib.resources.files("test").joinpath("test.yaml")
    )
    lavlab.ServiceProviderFactory.entries = load_entry_points_from_toml(
        "pyproject.toml", "lavlab_python_utils.service_providers"
    )


# Update the method to use the EntryPoints object correctly
def get_service_provider(name):
    """
    Retrieves the service provider based on its name.

    Args:
    name (str): The name of the service provider to retrieve.

    Returns:
    A service provider instance if available.
    """
    # Use .select to filter and find the correct entry point
    filtered_entry_points = lavlab.ServiceProviderFactory.entries.select(name=name)
    if filtered_entry_points:
        entry_point = next(
            iter(filtered_entry_points)
        )  # Get the first (should be only) entry
        return (
            entry_point.load()()
        )  # Assuming the entry point is callable and returns an instance
    return None
