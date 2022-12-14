# GHOST-xarray

GHOST-xarray provides helper functions
to load output files from
[GHOST](https://github.com/pmininni/GHOST)
(the Geophysical High-Order Suite for Turbulence)
into [xarray](https://xarray.dev/)
labelled multidimensional arrays.

## Why should I use xarray?

`xarray` provides two abstractions on top of NumPy arrays:
labeled dimensions
and coordinates.

With labeled dimensions,
this operation:

```python
y = x[:, 3]  # select y=3
y.sum(1)  # sum over z, which was axis 2 before
```

becomes this:

```python
y = x.isel(y=3)  # isel means index selection
y.sum("z")
```

We do not have to track which axis belong to which physical dimension.

Coordinates are an extension of
[pandas](https://pandas.pydata.org/)
indexes,
which allow to index by "physical" location:

```python
t = np.array([0, 10, 20, 30])
x = np.array([0, 1, 2, 3])

x_with_coords = xarray.DataArray(x, coords={"t": t})

x_with_coords.isel(t=1)  # Select by index position
x_with_coords.sel(t=10)  # Select by coordinate
```

Among many niceties,
which can be read in its [documentation](https://xarray.dev/),
`xarray` provides several conveniences for plotting,
some of which are shown below.

## Installation

```
pip install ghost-xarray
```

## Usage

GHOST-xarray provides 2 functions:

- `open_dataarray`, which opens a single variable,
- `open_dataset`, which opens multiple variables.

`open_dataarray` needs:
a directory, a variable name,
and its timestep `dt`, shape and datatype:

```python
import ghost_xarray

vx = ghost_xarray.open_dataarray(
    "path/to/directory",
    "vx",
    dt=0.5,
    shape=(128, 128, 128),
    dtype=np.float32,
)
vx.isel(t=0, y=64).plot()  # isel selects by index position
```

![A plot of the x-component of the velocity at t=0.](figures/vx.png)

By default,
it generates the corresponding coordinates
from the `shape` parameter
as `np.linspace(0, 2 * np.pi, shape[i], endpoint=False)`.
But `shape` can also be a dictionary of `np.ndarray` with explicit coordinates:

```python
vx = ghost_xarray.open_dataarray(
    "path/to/directory",
    "vx",
    dt=0.5,
    shape=dict(x=np.linspace(0, 10, 64), y=np.linspace(0, 100, 11)),
    dtype=np.float32,
)
v_average = vx.isel(t=0).sel(y=50).mean(dim="x")  # sel selects by coordinate
v_average.compute()
```

All operations are lazy,
until `.compute()` or `.plot()` are called.
Hence,
the above example only loads a single file to memory,
instead of actually loading data for all times.

For a vector-valued variable,
it adds an additional `i` dimension,
corresponding to `x, y[, z]`:

```python
v = ghost_xarray.open_dataarray(
    "path/to/directory",
    name="v",
    dt=0.5,
    coords=(128, 128, 128),
    dtype=np.float32,
)
v.isel(y=64).sel(t=1.5).plot(col="i")  # makes three 2D imshows, one for each component.
```

![A plot for each component of the velocity.](figures/v.png)

Finally,
`open_dataset` opens several variables
into a `xarray.Dataset`,
which provides a dict-like interface.

```python
data = ghost_xarray.open_dataset(
    "path/to/directory/",
    names=["v", "w"],
    dt=0.5,
    coords=(128, 128, 128),
    dtype=np.float32,
)

h = (data.v * data.w).sum(dim="i")  # computes the sum along components (dimension "i").
h.isel(t=slice(0, 4), x=64).plot(col="t")  # plots the first 4 timepoints
```

![Plot of the helicity for multiple timepoints.](figures/h.png)
