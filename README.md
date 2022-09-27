# GHOST-xarray

GHOST-xarray provides helper functions
to load output files from
[GHOST](https://github.com/pmininni/GHOST)
(the Geophysical High-Order Suite for Turbulence)
into [xarray](https://github.com/pydata/xarray)
labelled multidimensional arrays.

## Installation

```
pip install ghost-xarray
```

## Usage

GHOST-xarray provides 4 functions:

- load_scalar
- load_scalar_timeseries
- load_vector_timeseries
- load_dataset

To load a scalar at a particular timepoint,
`load_scalar` needs
its filename,
the coordinates
and the datatype:

```python
import ghost_xarray
import numpy as np

vx = ghost_xarray.load_scalar(
    "path/to/file/vx.0001.out",
    coords=dict(
        x=np.linspace(0, 2 * np.pi, 128, endpoint=False),
        y=np.linspace(0, 2 * np.pi, 128, endpoint=False),
    ),
    dtype=np.float32,
)
vx.plot()  # makes a 2D imshow
```

Coordinates can also be a tuple of `int`s,
in which case it generates the corresponding coordinates
as `np.linspace(0, 2 * np.pi, N, endpoint=False)`:

```python
vx = ghost_xarray.load_scalar(
    "path/to/file/vx.0001.out",
    coords=(128, 128, 128),  # 3D dataset
    dtype=np.float32,
)
vx.isel(y=64).plot()  # select a slice at the index 64 of y and plot (a 2D imshow).
```

![A plot of the x-component of the velocity.](figures/vx.png)

To load a scalar timeseries,
`load_scalar_timeseries` needs
a directory and the variable name,
the timestep `dt`,
and the coordinate and datatype as before:

```python
vx = ghost_xarray.load_scalar_timeseries(
    "path/to/directory",
    name="vx",
    dt=0.5,
    coords=(128, 128),
    dtype=np.float32,
)
vx.sel(t=1.5).plot()  # makes a 2D imshow of the x-z plane at t=1.5.
```

where time dimension is named `"t"`.

For a vector timeseries,
it is analogous:

```python
v = ghost_xarray.load_vector_timeseries(
    "path/to/directory",
    name="v",
    dt=0.5,
    coords=(128, 128, 128),
    dtype=np.float32,
)
v.isel(y=64).sel(t=1.5).plot(col="i")  # makes three 2D imshows, one for each component.
```

![A plot for each component of the velocity.](figures/v.png)

adding an additional dimension named `"i"` for each vector component.

Finally,
`load_dataset` loads several variables
into a `xarray.Dataset`,
which provides a dict-like interface.

```python
data = ghost_xarray.load_dataset(
    "path/to/directory/",
    names=["v", "w"],
    dt=0.5,
    coords=(128, 128),
    dtype=np.float32,
)

h = (data.v * data.w).sum(dim="i")  # computes the sum along components (dimension "i").
h.isel(t=slice(0, 4)).plot(col="t")  # plots the first 4 timepoints
```

## Delayed loading with Dask

By adding the `chunks` parameter,
delays loading the dataset,
and builds a dask graph instead.
Operations are performed lazily:

```python
data = ghost_xarray.load_dataset(
    "path/to/directory/",
    names=["v", "w"],
    dt=0.5,
    coords=(128, 128),
    dtype=np.float32,
    chunks={},
)

v_sum = data.v.isel(i="x", t=0).mean(dim=["x", "y"])  # lazy
v_sum = v_sum.compute()  # performs the actual calculation
```

In the above example,
only the `vx` component at `t=0` will be loaded,
and averaged in the `x-y` coordinates.
