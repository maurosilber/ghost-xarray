from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

import ghost_xarray

DTYPE = np.float64
SIZE = 8
N_FILES = 5


def generate_random(
    dirpath: Path,
    *,
    name: str,
    dims: Iterable[str] = None,
    n_times: int,
    shape: tuple[int],
    dtype: np.dtype,
):
    if dims is None:
        dims = ("",)

    files = []
    for i in range(1, n_times + 1):
        for dim in dims:
            file = dirpath / f"{name}{dim}.{i}.out"
            x = np.random.random(shape).astype(dtype)
            x.tofile(file)
            files.append(file)
    return files


def load_file(filename: str | Path, *, dtype: np.dtype, shape: tuple[int] = None):
    data = np.fromfile(filename, dtype=dtype)
    if shape is not None:
        data = data.reshape(shape)
    return data


def generate_coords(size, dims) -> tuple[tuple, dict]:
    shape = (size,) * len(dims)
    coords = {dim: np.linspace(0, 1, size) for dim in dims}
    return shape, coords


@pytest.mark.parametrize("name", ("scalar", "vector"))
@pytest.mark.parametrize("dims", (["x", "y"], ["x", "y", "z"]))
def test_dataarray(tmp_path, name, dims):
    shape, coords = generate_coords(SIZE, dims)

    files = generate_random(
        tmp_path,
        name=name,
        dims=dims if name == "vector" else None,
        n_times=N_FILES,
        shape=shape,
        dtype=DTYPE,
    )

    data_from_shape = ghost_xarray.open_dataarray(
        tmp_path, name, dt=1, shape=shape, dtype=DTYPE
    )
    data_from_coords = ghost_xarray.open_dataarray(
        tmp_path, name, dt=1, shape=coords, dtype=DTYPE
    )

    x_sum = 0.0
    for file in files:
        x_sum += np.sum(load_file(file, dtype=DTYPE, shape=shape))

    np.testing.assert_almost_equal(x_sum, data_from_shape.sum().compute())
    np.testing.assert_almost_equal(x_sum, data_from_coords.sum().compute())


@pytest.mark.parametrize(
    "names",
    [
        ("scalar", "scalar"),
        ("vector", "vector"),
        ("scalar", "vector"),
        ("vector", "scalar"),
    ],
)
@pytest.mark.parametrize("dims", (["x", "y"], ["x", "y", "z"]))
def test_dataset(tmp_path, names, dims):
    shape, coords = generate_coords(SIZE, dims)

    files = {
        name: generate_random(
            tmp_path,
            name=name,
            dims=dims if name == "vector" else None,
            n_times=N_FILES,
            shape=shape,
            dtype=DTYPE,
        )
        for name in names
    }

    data_from_shape = ghost_xarray.open_dataset(
        tmp_path, names, dt=1, shape=shape, dtype=DTYPE
    )
    data_from_coords = ghost_xarray.open_dataset(
        tmp_path, names, dt=1, shape=coords, dtype=DTYPE
    )

    for name, filelist in files.items():
        x_sum = 0.0
        for file in filelist:
            x_sum += np.sum(load_file(file, dtype=DTYPE, shape=shape))

        np.testing.assert_almost_equal(x_sum, data_from_shape.get(name).sum().compute())
        np.testing.assert_almost_equal(
            x_sum, data_from_coords.get(name).sum().compute()
        )
