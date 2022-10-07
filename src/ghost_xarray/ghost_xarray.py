from __future__ import annotations

from collections import defaultdict
from typing import Iterable

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import xarray
import xarray.backends


class BinaryBackendArray(xarray.backends.BackendArray):
    def __init__(self, file, shape, dtype):
        self.file = file
        self.shape = shape
        self.dtype = dtype

    @cached_property
    def array(self):
        return np.memmap(
            self.file,
            mode="c",  # copy-on-write
            dtype=self.dtype,
            shape=self.shape,
            order="F",
        )

    def __getitem__(self, key: tuple):
        return xarray.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xarray.core.indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        return self.array[key]


class BinaryBackend(xarray.backends.BackendEntrypoint):
    def open_dataset(
        self,
        file: Path,
        *,
        drop_variables=None,
        name: str = None,
        shape: tuple[int],
        dims: tuple[str],
        dtype: np.dtype,
    ):
        if name is None:
            name = file.stem.split(".", maxsplit=1)[0]

        backend_array = BinaryBackendArray(
            file=file,
            shape=shape,
            dtype=dtype,
        )
        data = xarray.core.indexing.LazilyIndexedArray(backend_array)
        return xarray.Dataset({name: (dims, data)})


def _removeprefix(line: str, prefix: str):
    # On Python >= 3.9, we could use str.removeprefix
    if line.startswith(prefix):
        line = line[len(prefix) :]
    return line


def _build_filelist(directory: str | Path, variable_name: str) -> pd.DataFrame:
    """Builds a DataFrame of the files found in the given directory
    for the specified variable_name, with a row for each timepoint.

    >>> _build_filelist("/my_dir", "vx")
        1   /my_dir/vx.0001.out
        2   /my_dir/vx.0002.out
        ...

    >>> _build_filelist("/my_dir", "v")
            x                    y
        1   /my_dir/vx.0001.out  /my_dir/vx.0001.out
        2   /my_dir/vx.0002.out  /my_dir/vy.0002.out
        ...
    """
    files = defaultdict(dict)
    for file in Path(directory).glob(f"{variable_name}*"):
        name, _, time = file.stem.partition(".")
        name = _removeprefix(name, variable_name)
        files[name][int(time)] = file
    return pd.DataFrame(files).sort_index()


def _normalize_coords(
    shape: tuple[int] | dict[str, np.ndarray]
) -> tuple[tuple[int], dict[str, np.ndarray]]:
    """Returns a shape tuple for a given dict of coordinates,
    or generates a dict of coordinates from a shape tuple.

    The generated coordinates are equispaced between [0, 2π).
    """
    if isinstance(shape, dict):
        coords = shape.copy()  # Copy, as it will be modified in-place later.
        shape = []
        for name in ("x", "y", "z"):
            try:
                shape.append(coords[name].size)
            except KeyError:
                pass
        shape = tuple(shape)
    else:
        coords = {
            name: np.linspace(0, 2 * np.pi, size, endpoint=False)
            for name, size in zip(("x", "y", "z"), shape)
        }

    return shape, coords


def open_dataarray(
    directory: str | Path,
    name: str,
    *,
    dt: float,
    shape: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
) -> xarray.DataArray:
    """Opens a variable as an xarray.DataArray."""
    shape, coords = _normalize_coords(shape)

    if len(shape) == 2:
        dims = ("x", "y")
    elif len(shape) == 3:
        dims = ("x", "y", "z")
    else:
        raise ValueError

    filelist = _build_filelist(directory, name)

    # GHOST numbers the timepoints from 1.
    coords["t"] = dt * (filelist.index - 1)

    # If the filelist contains more then one column,
    # they are assumed to be the vector components,
    # and are concatenated along the "i" dimension.
    if len(filelist.columns) == 1:
        filelist = filelist.squeeze()
        concat_dim = ["t"]
    else:
        concat_dim = ["t", "i"]
        coords["i"] = filelist.columns

    dataset = xarray.open_mfdataset(
        filelist.values.tolist(),
        engine=BinaryBackend,
        name=name,
        shape=shape,
        dims=dims,
        dtype=dtype,
        concat_dim=concat_dim,
        combine="nested",
    )
    dataarray = dataset.get(name).assign_coords(coords)
    return dataarray


def open_dataset(
    directory: str | Path,
    names: Iterable[str],
    *,
    dt: float,
    shape: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
) -> xarray.Dataset:
    """Opens multiple variables as an xarray.Dataset."""
    return xarray.merge(
        open_dataarray(
            directory,
            name,
            dt=dt,
            shape=shape,
            dtype=dtype,
        )
        for name in names
    )
