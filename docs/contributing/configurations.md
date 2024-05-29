# Expanding Configurations

## Overview

This guide covers how to expand configurations in LavLab Python Utils.

## Steps

1. **Identify the configuration**: Determine the configuration settings that need to be added or modified.
2. **Update configuration files**: Modify the default.yaml (and optionally test.yaml)
3. **Implement configuration handling**: Update the code to handle and use the new or modified configuration settings.

## Example

Here is an example of how we implemented memory safety:
1.  **Identify the configuration**: For memory safety I envision two potential scenarios
* Standard workflow: In the case of a normal user developing a script using our toolkit it is simply nice not to worry about OOM issues, in this case allowing some percentage of system of system memory to be allocated makes the most sense.
* Automated workflow: In the case of using the toolkit in automated tasks like some cronjob, it makes sense to have a hard memory cap. Especially in the case of parallelizing it is helpful to know that a given workload will not use more than a certain threshold of memory, also mapping helpfully to a kubernetes pod's resource limits configuration.

2. **Update configuration files***: With these usecases in mind we can add configurations like so:
```yaml
# src/lavlab/default.yaml
resources:
  memory_usage: 0.9
  max_memory: null 
```
We define some default values, but since we want a dynamic value (max_memory=system_memory), we will set that as null for now. Also, since max_mmemory is in bytes, it would be nice to be able to define as a multiplication string for example: `4*1024*1024*1024` for 4GiB. 

3. **Implement configuration handling**: Now for the real work
We'll start by pulling our newly defined configurations. Since this is a resouce configuration we'll put it in the ResourceContext. We can put helpers for config parsing directly in \_\_init\_\_.py
```python
# lavlab/__init__.py

# our helper to convert "1024 *1024" -> 1048576
def assure_multiplication_string_is_int(string: str) -> int:
    """Converts a string with a multiplication operator into an integer."""
    # make sure it's a string configuration
    if isinstance(string, str) and "*" in string:
        mem = 1
        # "1024 * 1024" -> ["1024 ", " 1024"]
        int_strings = string.split("*")
        for integer in int_strings:
            # "1024 " -> 1024   then get new product
            mem *= int(integer.strip())
    else: 
        # otherwise we will assure it's an integer, which should be a direct byte configuration
        mem = int(string)
    return mem

...

class ConfigCompiler:
    @staticmethod
    def _set_dynamic_values(config):
        # we get our dynamic system memory configuration here.
        if config["resources"]["max_memory"] is None:
            config["resources"]["max_memory"] = psutil.virtual_memory().total

...

class ResourceContext:
    def __init__(self, config: dict) -> None:
        # use our helper function to get max_memory in bytes, avoid using .get() as this pushes the problem down the line, default values should be set in the default.yml or at least the config compiler. That way if something is wrong in config compilation we can catch it earlier.
        # NOTICE we set _max_memory here, not max_memory. We will use getters and setters to make sure that if this is modified at runtime, everything is up to par.
        self._memory_usage = float(config["memory_usage"]) # if you do not have a helper to parse value, be sure to type it, just in case
        self._max_memory = assure_multiplication_string_is_int(config["max_memory"])

    # memory_usage is simple, but still make sure to document properly
    @property
    def memory_usage(self) -> float:
        """Controls ratio of max memory to use.
        Useful for using most of a systems memory.
        floating point 0-1
        """
        return self._memory_usage
    # we don't want users to be able to break this by setting anything crazy, assure the value is 0-1
    @memory_usage.setter
    def memory_usage(self, value: float) -> None:
        value = float(value)
        if value > 1 or value < 0:
            # Modules should have a LOGGER var for printing text. See the section on creating a new module for info on instantiating one of your own.
            LOGGER.error(
                "memory_usage must be between 0 and 1! Changing memory_usage to 1..."
            )
            value = 1
        self._memory_usage = value
    # max_memory is where getters and setters really shine, this way we can store the base value then use the memory_usage config to get the amount of memory we are actually allowed to use
    @property
    def max_memory(self) -> int:
        """Controls max memory to be considered in bytes.
        Useful for exact limits on memory consumption when paired with memory_usage=1"""
        # We can't use half a byte, so be sure the value is an integer.
        return int(self._max_memory * self.memory_usage) # don't use _vars outside of their getters and setters
    # we again don't want users to go crazy, make sure it is between 1 byte and the amount of memory in the system.
    @max_memory.setter
    def max_memory(self, value: int) -> None:
        value = int(value)
        if value < 1 or value > psutil.virtual_memory().total:
            LOGGER.error(
                "max_memory must be between 0 and your max! Changing max_memory to system max..."
            )
            value = psutil.virtual_memory().total
        self._max_memory = value
```
Great! Now we have added memory configurations to our context. All that's left is to use them. The primary application for this is numpy arrays. Remember helpers are our friends, we will make a new set of functions to use this contextual info when creating numpy arrays.
```python
# lavlab/python_util.py
import lavlab
# helper function to calculate the size of a proposed array in memory
def is_memsafe_array(shape: tuple[int, ...], dtype=np.float64) -> bool:
    """
    Checks if a desired array of given shape and datatype is too large for the memory constraints

    Parameters
    ----------
    shape : tuple[int,...]
        shape of the array
    dtype : np.dtype, optional
        datatype of given array, by default np.float64 for max safety

    Returns
    -------
    bool
        True if the array is safe to create in memory, False if it will blow your pc up.
    """
    size = np.prod(shape) * np.dtype(dtype).itemsize
    return size < lavlab.ctx.resources.max_memory

# use our helper in... another helper
def create_array(shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
    """
    Creates an in-memory array or a disk-based memmap array based on the available system memory.

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    dtype : np.dtype, Default: np.float64
        Data-type of the array's elements.

    Returns
    -------
    array
        Numpy in-memory array or memmap array based on the available system memory.
    """
    # if it's safe, create it
    if is_memsafe_array(shape, dtype):
        return np.zeros(shape, dtype)
    # storage safe arrays are defined similaryly with another configuration (lavlab.ctx.resources.max_temp_storage), it's all the same process though. 
    if not is_storage_safe_img(shape, dtype):
        raise MemoryError(
            f"Array of shape {shape} with dtype {dtype} is too large for storage."
        )
    path = tempfile.mkstemp(dir=lavlab.ctx.temp_dir)[1]
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)
```
Awesome! This is already a feature implemented, but with something as common as creating an array, chances are it's already happening somewhere in the code. Ideally you would take a peak around the package to make sure that wherever you create an array, you're using this method to ensure that it's type safe. For example:
```python
# lavlab/omero/images.py
def get_image_at_resolution_level(
    img: ImageWrapper, res_lvl: int, conn: BlitzGateway = None
) -> np.ndarray[np.uint8]:
    """Gets a full OMERO image from a given pyramid level.

    Parameters
    ----------
    img : ImageWrapper
        OMERO Image Object
    res_lvl : int
        Resolution level to pull from
    conn : BlitzGateway, optional
        OMERO Blitz Gateway, defaults to None

    Returns
    -------
    np.ndarray[np.uint8]
        OMERO Image as numpy array
    """
    if conn is None:
        conn = img._conn  # pylint: disable=W0212
    rps, close_rps = force_rps(img)
    img = force_image_wrapper(conn, img)
    rps.setResolutionLevel(res_lvl)

    size_x, size_y = get_rps_xy(rps)
    # use our new context-aware helper!
    arr = create_array((size_y, size_x, img.getSizeC()), np.uint8)
    for c in range(img.getSizeC()):
        arr[:, :, c] = get_plane_at_resolution_level(rps, res_lvl, 0, c, 0, conn=conn)

    if close_rps is True:
        rps.close()
    return arr
```

## Testing

Depending on your configuration, you may not need to add tests, for example create_array is tested through the other functions using it. Although it may be apt to create tests setting your new settings a few times in a few ways just to assure everything works as expected. Like in this case you might want to set max_memory a few times to make sure it doesn't slowly get smaller due to it getting modified by memory_usage. ( But I didn't :p )