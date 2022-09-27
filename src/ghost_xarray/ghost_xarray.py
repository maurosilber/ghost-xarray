from __future__ import annotations

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property
from pathlib import Path

import numpy as np
import xarray
import xarray.backends


class BinaryBackendArray(xarray.backends.BackendArray):
    def __init__(self, file, shape, dtype):
        self.file = file
        self.shape = shape
        self.dtype = dtype
        self.array = np.memmap(
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
        file,
        *,
        drop_variables=None,
        name: str = None,
        coords: Coords,
        dtype: np.dtype,
    ):
        if name is None:
            name = str(file)

        backend_array = BinaryBackendArray(
            file=file,
            shape=coords.as_tuple,
            dtype=dtype,
        )
        data = xarray.core.indexing.LazilyIndexedArray(backend_array)
        var = xarray.Variable(dims=coords.coord_names, data=data)
        return xarray.Dataset({name: var})


class Coords:
    coords_names: list[str]

    def __new__(cls, coords: Coords):
        if isinstance(coords, Coords):
            return coords
        return super().__new__(cls)

    def __init__(self, coords: tuple[int] | dict[str, np.ndarray]):
        if isinstance(coords, Coords):
            return

        if len(coords) == 2:
            self.coord_names = ["x", "y"]
        elif len(coords) == 3:
            self.coord_names = ["x", "y", "z"]

        if isinstance(coords, tuple):
            self.as_tuple = coords
        elif isinstance(coords, dict):
            self.as_dict = coords

    @cached_property
    def as_tuple(self) -> tuple[int]:
        return tuple(self.as_dict[i].size for i in self.coord_names)

    @cached_property
    def as_dict(self):
        return {
            k: np.linspace(0, 2 * np.pi, v, endpoint=False)
            for k, v in zip("xyz", self.as_tuple)
        }


def load_scalar(
    file: str | Path,
    *,
    coords: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
    chunks: int | dict[str, int] = None,
):
    coords: Coords = Coords(coords)
    return xarray.open_dataarray(
        file,
        engine=BinaryBackend,
        chunks=chunks,
        coords=coords,
        dtype=dtype,
    ).assign_coords(coords.as_dict)


def load_scalar_timeseries(
    directory: str | Path,
    name: str,
    *,
    dt: float,
    coords: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
    chunks: int | dict[str, int] = None,
):
    coords = Coords(coords)
    files = Path(directory).glob(f"{name}.*.out")
    t, arrays = [], []
    for file in sorted(files):
        _, ti = file.stem.split(".")
        t.append(ti)
        arrays.append(
            load_scalar(
                file,
                coords=coords,
                chunks=chunks,
                dtype=dtype,
            )
        )
    t = dt * (np.array(t, dtype=float) - 1)
    return xarray.concat(
        arrays,
        dim=xarray.IndexVariable("t", t),
        coords="all",
        compat="override",
        join="override",
    )


def load_vector_timeseries(
    directory: str | Path,
    name: str,
    *,
    dt: float,
    coords: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
    chunks: int | dict[str, int] = None,
):
    coords: Coords = Coords(coords)
    components = [
        load_scalar_timeseries(
            directory,
            name=f"{name}{i}",
            dt=dt,
            coords=coords,
            chunks=chunks,
            dtype=dtype,
        )
        for i in coords.coord_names
    ]
    return xarray.concat(
        components,
        dim=xarray.IndexVariable("i", coords.coord_names),
        coords="all",
        compat="override",
        join="override",
    )


def load_dataset(
    directory: str | Path,
    names: list[str],
    *,
    dt: float,
    coords: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
    chunks: int | dict[str, int] = None,
):
    coords = Coords(coords)
    return xarray.Dataset(
        {
            name: load_vector_timeseries(
                directory,
                name,
                dt=dt,
                coords=coords,
                chunks=chunks,
                dtype=dtype,
            )
            for name in names
        }
    )
