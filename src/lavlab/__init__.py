"""Handles the configuration of our python modules
"""

import importlib.resources
import logging
import multiprocessing
import os
import platform
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from importlib.metadata import entry_points
from typing import Optional, Union

import psutil  # type: ignore
import yaml  # type: ignore


def assure_multiplication_string_is_int(string: str) -> int:
    """Converts a string with a multiplication operator into an integer."""
    if isinstance(string, str) and "*" in string:
        mem = 1
        int_strings = string.split("*")
        for integer in int_strings:
            mem *= int(integer.strip())
    else:
        mem = int(string)
    return mem


class ServiceProviderFactory:
    """Factory to dynamically load and create service providers based on registered entry points."""

    entries = entry_points(group="lavlab-python-utils.service_providers")

    @staticmethod
    def create_service_provider(service_name: str, **kwargs):
        """
        Creates service provider based off registered entry points.

        Parameters
        ----------
        service_name : str
            service name to create provider for.

        Returns
        -------
        ServiceProvider
            Implementation of the service provider.

        Raises
        ------
        ValueError
            Cannot find service provider for the given service name.
        """
        service_name = service_name.upper()
        entrypoints = tuple(ServiceProviderFactory.entries.select(name=service_name))
        if entrypoints:
            entry_point = entrypoints[0]
            provider_class = entry_point.load()
            return provider_class(**kwargs)
        raise ValueError(f"No service provider available for {service_name}")


FILETYPE_MAP = {
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ndpi": "image/vnd.hamamatsu.ndpi",
    ".svs": "image/vnd.aperio.svs",
    ".bif": "image/vnd.leica.bif",
    ".dicom": "application/dicom",
    ".dcm": "application/dicom",
    ".nii": "application/nifti",
    ".gz": "application/x-gzip",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".xml": "application/xml",
    ".json": "application/json",
}


def get_mimetype(file_extension: str) -> str:
    """
    Retrieve the MIME type based on the file extension.

    Parameters
    ----------
    file_extension : str
        The extension of the file (including the period).

    Returns
    -------
    str
        The MIME type of the file. Defaults to 'application/octet-stream' if unknown.
    """
    return FILETYPE_MAP.get(file_extension.lower(), "application/octet-stream")


def get_mimetype_from_path(file_path: str) -> str:
    """Wrapper for get_mimetype, just splits the file extension off a path.

    Parameters
    ----------
    file_path : str
        Posix path.

    Returns
    -------
    str
        Mimetype string from enum.
    """
    return get_mimetype(f'.{file_path.split(".")[-1]}')


class DependencyThreadConfiguration(Enum):
    """Converts a module name into the environemental variable for controlling max cpu threads.
    Used for passing the module thread configurations into dependencies.
    """

    PYVIPS = "VIPS_CONCURRENCY"

    @staticmethod
    def is_module_imported(module_name: str) -> bool:
        """Checks if a given module is imported by the module's name.

        Parameters
        ----------
        module_name : str
            Name of the given module, ex. "pyvips".

        Returns
        -------
        bool
            Bool representing whether the module has already been imported.
        """
        return module_name in sys.modules


class ResourceContext:
    """Controls resource utilization of child processes.
    Allows the configuration of CPU and memory limits.
    Ideally permeates to all dependencies for centralized configuration.
    """

    DEP_THREAD_ENUM = DependencyThreadConfiguration

    def __init__(self, config: dict, logger=None) -> None:
        if logger is None:
            self.logger = logging.getLogger(f"{__name__}.resources")
        self._max_cores = config.get("max_cores", 1)
        self._memory_usage = config.get("memory_usage", 1)
        self._max_memory = assure_multiplication_string_is_int(
            config.get(
                "max_memory", int(psutil.virtual_memory().total * self._memory_usage)
            )
        )
        self._max_temp_storage = config.get(
            "max_temp_storage", None
        )  # None indicates no limit

        self._io_max_threads = config.get("io_max_threads", 1)
        self._io_pool: Optional[ThreadPoolExecutor] = None

    def context_summary(self) -> list[str]:
        """
        Summary of configurations used for logging.

        Returns
        -------
        list of str
            List of lines to log
        """
        free_memory = psutil.virtual_memory().available // (1024**3)  # GB
        return [
            f"=== Resource Context Summary ===\n"
            f"Max CPU Cores: {self.max_cores}\n"
            f"Max IO Threads: {self.io_max_threads}\n"
            f"Memory Usage Limit: {int(self.max_memory) // (1024 ** 3)} GB\n"
            f"Free System Memory: {free_memory} GB\n"
        ]

    @property
    def max_cores(self) -> int:
        """Controls maximum cores to be used.
        Must not be greater than total cores or less than 1!
        """
        return self._max_cores

    @max_cores.setter
    def max_cores(self, value: int) -> None:
        value = int(value)
        if value < 1 or value > psutil.cpu_count():
            self.logger.error(
                "max_cores must be between 1 and your total cores! Changing max_cores to 1..."
            )
            value = 1
        self._max_cores = value
        for envvar in self.DEP_THREAD_ENUM:
            if self.DEP_THREAD_ENUM.is_module_imported(envvar.name.lower()):
                self.logger.warning(
                    "Cannot set %s's concurrency after module has been imported.",
                    envvar.name.lower(),
                )
            else:
                os.environ[envvar.value] = str(value)

    @property
    def memory_usage(self) -> float:
        """Controls ratio of max memory to use.
        Useful for using most of a systems memory.
        floating point 0-1
        """
        return self._memory_usage

    @memory_usage.setter
    def memory_usage(self, value: float) -> None:
        value = float(value)
        if value > 1 or value < 0:
            self.logger.error(
                "memory_usage must be between 0 and 1! Changing memory_usage to 1..."
            )
            value = 1
        self._memory_usage = value

    @property
    def max_memory(self) -> int:
        """Controls max memory to be considered in bytes.
        Useful for exact limits on memory consumption when paired with memory_usage=1"""
        return self._max_memory

    @max_memory.setter
    def max_memory(self, value: int) -> None:
        value = int(value)
        if value < 1 or value > psutil.virtual_memory().total:
            self.logger.error(
                "max_memory must be between 0 and your max! Changing max_memory to system max..."
            )
            value = psutil.virtual_memory().total
        self._max_memory = value

    @property
    def io_max_threads(self) -> int:
        """Amount of threads to spawn for IO requests.
        Useful for speeding up OMERO and XNAT requests."""
        return self._io_max_threads

    @io_max_threads.setter
    def io_max_threads(self, value: int) -> None:
        value = int(value)
        if value < 1:
            self.logger.error(
                "io_max_threads must be 1 or more! Changing io_max_threads to 1..."
            )
            value = 1
        if value > 8:
            self.logger.warning(
                "i sincerely doubt you need all these io_max_threads, but go off queen"
            )
        self._io_max_threads = value

    @property
    def io_pool(self) -> ThreadPoolExecutor:
        """Pool used for (mostly network) IO
        Mostly for internal stuff.

        Returns
        -------
        ThreadPoolExecutor
            Creates new threadpool if necessary
        """
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(
                self._io_max_threads, thread_name_prefix="lavlab-python-utils.io_pool"
            )
        elif self._io_pool._max_workers != self.io_max_threads:  # pylint: disable=W0212
            self._io_pool.shutdown()
            self._io_pool = ThreadPoolExecutor(
                self._io_max_threads, thread_name_prefix="lavlab-python-utils.io_pool"
            )
        elif self._io_pool._shutdown:  # pylint: disable=W0212
            self._io_pool = ThreadPoolExecutor(
                self._io_max_threads, thread_name_prefix="lavlab-python-utils.io_pool"
            )
        return self._io_pool

    @io_pool.setter
    def io_pool(self, value: ThreadPoolExecutor) -> None:
        self.logger.warning(
            "This isn't intented to be set like this, hope you know what you're doing!"
        )
        self._io_pool = value

    @property
    def max_temp_storage(self) -> int:
        """Amount of temporary storage to be used in bytes"""
        return self._max_temp_storage

    @max_temp_storage.setter
    def max_temp_storage(self, value: int) -> None:
        value = int(value)
        self._max_temp_storage = value


class HistologyContext:  # TODO proper setters and getters
    """Provides configuration options for histology tools and services
    Expected to be held in some kind of utilcontext
    """

    def __init__(self, config: dict, logger=None) -> None:
        if logger is None:
            self.logger = logging.getLogger(f"{__name__}.histology")
        self.size_threshold = assure_multiplication_string_is_int(
            config.get("size_threshold", -1)
        )
        self.service = config.get("service", {})
        self.use_fast_compression = config.get("use_fast_compression", True)
        self.fast_compression_options = config.get("fast_compression_options", {})
        self.slow_compression_options = config.get("slow_compression_options", {})
        self.tiling_options = config.get("tiling_options", {})
        self.service_provider = None

    def get_service_provider(self):
        """
        Uses the ServiceProviderFactory to create a service provider based on the configuration.

        Returns
        -------
        ServiceProvider
            Implementation of the service provider.
        """
        if self.service_provider is None:
            kwargs = self.service
            self.service_provider = ServiceProviderFactory.create_service_provider(
                kwargs.pop("name"), **kwargs
            )
        return self.service_provider


class ConfigCompiler:
    """Handles reading configs in the proper priorities."""

    @staticmethod
    def load_default_config() -> dict:
        """Reads default.yaml from the package internals

        Returns
        -------
        dict
            default.yaml
        """
        return ConfigCompiler._read_yaml_config(
            str(importlib.resources.files("lavlab").joinpath("default.yaml"))
        )

    @staticmethod
    def compile(**kwargs) -> dict:
        """
        Compiles the configuration from the default, system, and user configs.

        Returns
        -------
        dict
            dictionary containing the compiled configuration.
        """
        default_config = ConfigCompiler.load_default_config()
        # override defaults with kwargs, primarily for tests
        default_config = ConfigCompiler._merge_configs(default_config, kwargs)
        system_config_file = os.path.expanduser(default_config["default_system_file"])
        system_config = ConfigCompiler._read_yaml_config(system_config_file)

        # Read user config
        user_config_file = os.path.expanduser(default_config["default_user_file"])
        user_config = ConfigCompiler._read_yaml_config(user_config_file)

        # Merge configs: default < system < user
        merged_config = ConfigCompiler._merge_configs(default_config, system_config)
        merged_config = ConfigCompiler._merge_configs(merged_config, user_config)

        # Override with environment variables if available
        merged_config = ConfigCompiler._override_with_env_vars(merged_config)

        # Set dynamic values
        final_config = ConfigCompiler._set_dynamic_values(merged_config)

        return final_config

    @staticmethod
    def _read_yaml_config(file_path: Union[str, bytes, os.PathLike]) -> dict:
        """Reads config file from the filesystem.

        Parameters
        ----------
        file_path : str
            Path to the YAML file.

        Returns
        -------
        dict
            YAML configuration as a dictionary.
        """
        if os.path.exists(file_path):
            with open(file_path, encoding="utf-8") as file:
                return yaml.safe_load(file)
        return {}

    @staticmethod
    def _merge_configs(base_config: dict, override_config: dict) -> dict:
        """Merge dictionaries, overwriting parameters and recursing as necessary.

        Parameters
        ----------
        base_config : dict
            Configuration to be overridden.
        override_config : dict
            Configuration to override the base.

        Returns
        -------
        dict
            Merged configuration.
        """
        for key, value in override_config.items():
            if isinstance(value, dict):
                base_config[key] = ConfigCompiler._merge_configs(
                    base_config.get(key, {}), value
                )
            else:
                base_config[key] = value
        return base_config

    @staticmethod
    def _override_with_env_vars(config):
        # TODO environment configuration overrides, add docstrings and typing
        # You can add logic here to override specific config values with environment variables
        # For example:
        # if "MY_APP_LOG_LEVEL" in os.environ:
        #     config["log_level"] = os.environ["MY_APP_LOG_LEVEL"]
        return config

    @staticmethod
    def _set_dynamic_values(config):
        # Set max_cores if not specified
        if config["resources"]["max_cores"] is None:
            config["resources"]["max_cores"] = multiprocessing.cpu_count()

        # Set max_memory if not specified
        if config["resources"]["max_memory"] is None:
            avail = psutil.virtual_memory().available
            config["resources"]["max_memory"] = (
                avail * config["resources"]["memory_usage"]
            )

        # Set temp_dir if not specified
        if config["temp_dir"] is None:
            config["temp_dir"] = tempfile.gettempdir()

        return config


class UtilContext:
    """Provides the base for configuring the LavLab Python Utilities
    Expected to be extended through a child class, see LavLabContext
    """

    HISTOLOGY_CLASS = HistologyContext
    RESOURCE_CLASS = ResourceContext
    # FILETYPE_CLASS = FiletypeEnum

    def __init__(self, logger=None, **kwargs):
        if logger is None:
            self.logger = logging.getLogger(f"{__name__}.histology")

        config = ConfigCompiler.compile(**kwargs)

        self.temp_dir = config.get("temp_dir")
        self.log_level = config.get("log_level")
        self.noninteractive = config.get("noninteractive")
        self.histology = self.HISTOLOGY_CLASS(config.get("histology"))
        self.resources = self.RESOURCE_CLASS(config.get("resources"))
        self.log_context_summary()

    def log_context_summary(self):
        """Log summary information about the hardware configurations in use."""
        os_info = platform.uname()
        summary = [
            f"=== Utility Context Summary ===\n"
            f"Operating System: {os_info.system} {os_info.release} ({os_info.version})\n"
            f"Temp Directory: {self.temp_dir}\n"
        ]
        summary.append(self.resources.context_summary())

        self.logger.info(summary)

    @property
    def log_level(self):
        """Logging level for the application."""
        return self._log_level

    @log_level.setter
    def log_level(self, value):
        self._log_level = value
        self.logger.setLevel(value)

    @property
    def temp_dir(self):
        """Directory for temporary files."""
        return self._temp_dir

    @temp_dir.setter
    def temp_dir(self, value):
        self._temp_dir = value

    @property
    def noninteractive(self):
        """Flag to indicate if the application should run in non-interactive mode."""
        return self._noninteractive

    @noninteractive.setter
    def noninteractive(self, value):
        self._noninteractive = bool(value)


# class UtilContext:
#     """Provides the base for configuring the LavLab Python Utilities
#     Expected to be extended through a child class, see LavLabContext
#     """
#     # Static class variables
#     HISTOLOGY_CLASS = HistologyContext
#     RESOURCE_CLASS = ResourceContext
#     FILETYPE_CLASS = FiletypeEnum
#     logger = logging.getLogger(f"{__name__}.UtilContext")

#     _temp_dir = "/default/temp/dir"
#     _log_level = logging.INFO
#     _noninteractive = False
#     _histology = HISTOLOGY_CLASS()
#     _resources = RESOURCE_CLASS()

#     @classmethod
#     def initialize(cls, **kwargs):
#         cls._temp_dir = kwargs.get("temp_dir", cls._temp_dir)
#         cls._log_level = kwargs.get("log_level", cls._log_level)
#         cls._noninteractive = kwargs.get("noninteractive", cls._noninteractive)
#         cls._histology = cls.HISTOLOGY_CLASS(kwargs.get("histology", {}))
#         cls._resources = cls.RESOURCE_CLASS(kwargs.get("resources", {}))
#         cls.log_context_summary()

#     @classmethod
#     def log_context_summary(cls):
#         """Log summary information about the hardware configurations in use."""
#         os_info = platform.uname()
#         summary = [
#             f"=== Utility Context Summary ===\n"
#             f"Operating System: {os_info.system} {os_info.release} ({os_info.version})\n"
#             f"Temp Directory: {cls._temp_dir}\n"
#         ]
#         summary.append(cls._resources.context_summary())
#         cls.logger.info("\n".join(summary))

#     @classmethod
#     def set_log_level(cls, value):
#         cls._log_level = value
#         cls.logger.setLevel(value)

#     @classmethod
#     def set_temp_dir(cls, value):
#         cls._temp_dir = value

#     @classmethod
#     def set_noninteractive(cls, value):
#         cls._noninteractive = bool(value)


# UtilContext.initialize(temp_dir="/path/to/temp", log_level=logging.DEBUG, noninteractive=True)


# class LavLabContext(UtilContext):  # TODO finish lavlab context
# def __init__(self, logger=None, **kwargs):
#     super().__init__(logger, **kwargs)

ctx = UtilContext()
