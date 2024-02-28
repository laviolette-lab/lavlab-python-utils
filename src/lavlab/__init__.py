import os
import sys
import yaml
import shutil
import psutil
import getpass
import logging
import platform
import tempfile
import threading
import multiprocessing

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from authlib.integrations.requests_client import OAuth2Session


import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

def is_module_imported(module_name):
    return module_name in sys.modules

class BaseContext:
    LOGGER = logging.getLogger(__name__)
    DEFAULT_USER_FILE = "~/.lavlab_user.yml"
    FAST_COMPRESSION_OPTIONS = {"compression": "lz4"}
    SLOW_COMPRESSION_OPTIONS = {"compression": "webp", "lossless": True}
    TILING_OPTIONS = {"tile":True, "tile_width": 1024, "tile_height": 1024}
    C_THREADED = {"pyvips": "VIPS_CONCURRENCY"}
    def __init__(self):
        self._log_level = self.get_log_level_from_env()
        self.LOGGER.setLevel(self._log_level)
        self._temp_dir = self.get_temp_dir_from_env()
        self._noninteractive = self.get_from_env("NONINTERACTIVE", False)

        self._omero_url = self.get_from_env("OMERO_URL")

        self._xnat_url = self.get_from_env("XNAT_URL")

        self._max_cores = self.get_max_cores_from_env()
        self._memory_usage = self.get_memory_usage_from_env()
        self._max_memory = self.get_max_memory_from_env()

        self._io_max_threads = self.get_io_max_threads_from_env()
        self._io_thread_pool = None
        self._io_thread_prefix = ""
        self._io_thread_initializer = None
        self._io_thread_init_args = ()
        self.thread_local = None

        self._credentials = {}
        self._user_file = None
        self._load_user_file()

        self._os_info = platform.uname()  # Saving the OS info
        self.log_context_summary()

    def get_from_env(self, key, default=None):
        return os.getenv(key, default)

    def get_log_level_from_env(self):
        return os.getenv("LOG_LEVEL", logging.WARNING)

    def get_max_cores_from_env(self):
        return os.getenv("MAX_CORES", multiprocessing.cpu_count())

    def get_io_max_threads_from_env(self):
        return int(os.getenv("IO_MAX_THREADS", 4))

    def get_memory_usage_from_env(self):
        return float(os.getenv("MEMORY_USAGE", 0.9))

    def get_max_memory_from_env(self):
        return int(os.getenv("MAX_MEMORY", psutil.virtual_memory().available * self._memory_usage))

    def get_temp_dir_from_env(self):
        return os.getenv("TEMP_DIR", tempfile.gettempdir())

    def get_io_max_threads_from_env(self):
        return int(os.getenv("IO_MAX_THREADS", 4))

    def log_context_summary(self):
        """Log summary information about the hardware configurations in use."""

        # Get available free space in the temp directory
        temp_dir_stat = shutil.disk_usage(self._temp_dir)
        free_space_temp = temp_dir_stat.free // (1024 ** 3)  # GB

        # Get free memory
        free_memory = psutil.virtual_memory().available // (1024 ** 3)  # GB

        summary = (
            f"=== Context Summary ===\n"
            f"Operating System: {self._os_info.system} {self._os_info.release} ({self._os_info.version})\n"
            f"OMERO URL: {self._omero_url}\n"
            f"XNAT URL: {self._xnat_url}\n"
            f"Max CPU Cores: {self._max_cores}\n"
            f"Max IO Threads: {self._io_max_threads}\n"
            f"Memory Usage Limit: {self._max_memory // (1024 ** 3)} GB\n"
            f"Free System Memory: {free_memory} GB\n"
            f"Temp Directory: {self._temp_dir}\n"
            f"Available Temp Storage: {free_space_temp} GB\n"
        )

        self.LOGGER.info(summary)

    def getLogger(cls, name):
        logger = logging.getLogger(name)
        logger.setLevel(cls.LOGGER.level)
        return logger

    @property
    def log_level(self):
        """Logging level for the application. Set using LOG_LEVEL environment variable or through the property."""
        return self._log_level

    @log_level.setter
    def log_level(self, value):
        self._log_level = value
        self.LOGGER.setLevel(value)

    @property
    def omero_url(self):
        """URL for the OMERO service, configured through environment variables or explicitly set."""
        return self._omero_url

    @omero_url.setter
    def omero_url(self, value):
        self._omero_url = value

    @property
    def xnat_url(self):
        """URL for the XNAT service, configured through environment variables or explicitly set."""
        return self._xnat_url

    @xnat_url.setter
    def xnat_url(self, value):
        self._xnat_url = value

    @property
    def max_cores(self):
        """Maximum number of CPU cores to be used, defined either through MAX_CORES environment variable or explicitly set."""
        return self._max_cores

    @max_cores.setter
    def max_cores(self, value):
        self._max_cores = value
        for lib, envvar in self.C_THREADED:
            if is_module_imported(lib):
                self.LOGGER.error(f"Cannot set {lib} concurrency after module has been imported.")
            else:
                os.environ[envvar] = str(value)

    @property
    def io_max_threads(self):
        """Maximum number of IO threads for outgoing requests. Can be adjusted dynamically."""
        return self._io_max_threads

    @io_max_threads.setter
    def io_max_threads(self, value):
        self._io_max_threads = value
        if self._io_thread_pool:
            self._io_thread_pool = self._io_thread_pool.shutdown()

    @property
    def io_thread_pool(self):
        if self._io_thread_pool is not None:
            self._io_thread_pool.shutdown()

        self.thread_local = threading.local()
        self._io_thread_pool = ThreadPoolExecutor(
            max_workers=self._io_max_threads,
            initializer=self._io_thread_initializer,
            initargs=self._io_thread_init_args,
            thread_name_prefix=self._io_thread_prefix
        )
        return self._io_thread_pool

    @property
    def io_thread_prefix(self):
        return self._io_thread_prefix

    @io_thread_prefix.setter
    def io_thread_prefix(self, value):
        self._io_thread_prefix = value
        if self._io_thread_pool:
            self._io_thread_pool = self._io_thread_pool.shutdown()

    @property
    def io_thread_initializer(self):
        return self._io_thread_initializer

    @io_thread_initializer.setter
    def io_thread_initializer(self, initializer_function):
        self._io_thread_initializer = initializer_function
        if self._io_thread_pool:
            self._io_thread_pool = self._io_thread_pool.shutdown()

    @property
    def io_thread_init_args(self):
        return self._io_thread_init_args

    @io_thread_init_args.setter
    def io_thread_init_args(self, args):
        self._io_thread_init_args = args
        if self._io_thread_pool:
            self._io_thread_pool.shutdown()

    @property
    def memory_usage(self):
        """Fraction of total memory to be used by the application. Can be configured through environment variables or explicitly set."""
        return self._memory_usage

    @memory_usage.setter
    def memory_usage(self, value):
        self._memory_usage = value

    @property
    def max_memory(self):
        """Maximum amount of memory to be used, either defined through MAX_MEMORY environment variable or explicitly set."""
        return self._max_memory

    @max_memory.setter
    def max_memory(self, value):
        self._max_memory = value

    @property
    def temp_dir(self):
        """Directory for temporary files, defined either through TEMP_DIR environment variable or explicitly set."""
        return self._temp_dir

    @temp_dir.setter
    def temp_dir(self, value):
        self._temp_dir = value

    @property
    def noninteractive(self):
        """Flag to indicate if the application should run in non-interactive mode, i.e., not prompt for input."""
        if self._noninteractive is None:
            self._noninteractive = bool(os.getenv("NONINTERACTIVE", False))
        return self._noninteractive

    @noninteractive.setter
    def noninteractive(self, value):
        self._noninteractive = bool(value)

    @property
    def user_file(self):
        """Path to the YAML file containing usernames and passwords for various services. Defaults to ~/.lavlab_user.yml"""
        if self._user_file is None:
            self._user_file = os.getenv("USER_FILE", self.DEFAULT_USER_FILE)
        return self._user_file

    @user_file.setter
    def user_file(self, value):
        self._user_file = value
        self._load_user_file()

    def _load_user_file(self):
        user_file_path = Path(self.user_file).expanduser()
        if user_file_path.exists():
            try:
                with open(user_file_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    self._credentials = loaded_config.get('credentials', {})
                    self._oauth2_config = loaded_config.get('oauth2', {})
            except Exception as e:
                self.LOGGER.warning(f"Failed to load credentials from file: {e}")
        else:
            self.LOGGER.warning(f"No user file found at {user_file_path}")

    def _prompt_for_credentials(self, service):
        """Prompt for credentials and optionally save them to the default user file."""
        if not self.noninteractive:
            username = input(f"Enter {service} username: ")
            password = getpass.getpass(f"Enter {service} password: ")

            save_option = input("Would you like to save these credentials? [y/N]: ").strip().lower()
            if save_option == 'y':
                if not self._credentials:
                    self._credentials = {}
                self._credentials[service] = {"username": username, "password": password}
                self._save_to_user_file()

            return username, password
        return None, None

    def _save_to_user_file(self):
        """Save credentials to the user file."""
        user_file_path = Path(self.user_file).expanduser()
        with open(user_file_path, 'w') as f:
            yaml.safe_dump(self._credentials, f)

        self.LOGGER.info(f"Saved credentials to {user_file_path}")


    def get_credentials(self, service):
        username = os.getenv(f"{service.upper()}_USERNAME") or \
                   self._credentials.get(service, {}).get("username")
        password = os.getenv(f"{service.upper()}_PASSWORD") or \
                   self._credentials.get(service, {}).get("password")

        if username is None or password is None:
            username, password = self._prompt_for_credentials(service)

        return username, password


class LavLabContext(BaseContext):
   def __init__(self):
        super().__init__()
        self._omero_url = "wsi.lavlab.mcw.edu"
        self._xnat_url = "mri.lavlab.mcw.edu"


ctx = LavLabContext()
