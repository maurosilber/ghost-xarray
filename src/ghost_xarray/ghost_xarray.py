from __future__ import annotations

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
        coords: dict[str, np.ndarray],
        dtype: np.dtype,
    ):
        if len(coords) == 2:
            coord_order = ["x", "y"]
        elif len(coords) == 3:
            coord_order = ["x", "y", "z"]
        else:
            raise ValueError("len(coords) must be 2 or 3.")

        if name is None:
            name = str(file)
        backend_array = BinaryBackendArray(
            file=file,
            shape=tuple(coords[i].size for i in coord_order),
            dtype=dtype,
        )
        data = xarray.core.indexing.LazilyIndexedArray(backend_array)
        var = xarray.Variable(dims=coord_order, data=data)
        return xarray.Dataset({name: var}, coords=coords)


def _normalize_coords(
    coords: tuple[int] | dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if isinstance(coords, dict):
        return coords
    elif isinstance(coords, tuple):
        return {
            name: np.linspace(0, 2 * np.pi, size, endpoint=False)
            for name, size in zip("xyz", coords)
        }


def load_scalar(
    file: str | Path,
    *,
    coords: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
    chunks: int | dict[str, int] = None,
):
    return xarray.open_dataarray(
        file,
        engine=BinaryBackend,
        chunks=chunks,
        coords=_normalize_coords(coords),
        dtype=dtype,
    )


def load_scalar_timeseries(
    directory: str | Path,
    name: str,
    *,
    dt: float,
    coords: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
    chunks: int | dict[str, int] = None,
):
    coords = _normalize_coords(coords)
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
    t = dt * np.array(t, dtype=float)
    return xarray.concat(arrays, dim=xarray.IndexVariable("t", t))


def load_vector_timeseries(
    directory: str | Path,
    name: str,
    *,
    dt: float,
    coords: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
    chunks: int | dict[str, int] = None,
):
    coords = _normalize_coords(coords)
    component_names = list(coords.keys())
    components = [
        load_scalar_timeseries(
            directory,
            name=f"{name}{i}",
            dt=dt,
            coords=coords,
            chunks=chunks,
            dtype=dtype,
        )
        for i in component_names
    ]
    return xarray.concat(components, dim=xarray.IndexVariable("i", component_names))


def load_dataset(
    directory: str | Path,
    names: list[str],
    *,
    dt: float,
    coords: tuple[int] | dict[str, np.ndarray],
    dtype: np.dtype,
    chunks: int | dict[str, int] = None,
):
    coords = _normalize_coords(coords)
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
