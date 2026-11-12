import os
import re
from datetime import datetime

import numpy as np
import rasterio
import rasterio.warp
import xarray as xr
import pandas as pd
from scipy.stats import norm, lognorm

def read_and_reproject_tif(file_path, dst_crs, dst_transform, dst_width, dst_height):
    with rasterio.open(file_path) as src:
        data = src.read(
            out_shape=(src.count, dst_height, dst_width),
            resampling=rasterio.enums.Resampling.bilinear
        )
        data = data.astype(src.profile['dtype'])

        # Reproject the data
        for i in range(1, src.count + 1):
            rasterio.warp.reproject(
                source=rasterio.band(src, i),
                destination=data[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=rasterio.enums.Resampling.bilinear
            )

    return data

def create_xarray_gee_dataset_from_folder(folder_path):


    # List to store data arrays and timestamps
    data_arrays = []
    times = []

    # Regex supports both:
    #   S1_Field5_20251004.tif
    #   S1_Field5_20251004T153022.tif
    pattern = re.compile(r'S1_[A-Za-z0-9]+_(\d{8}(T\d{6})?)')

    # Determine common shape and transformation from the first .tif file
    initialized = False

    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            match = pattern.search(filename)
            if match:
                file_path = os.path.join(folder_path, filename)
                with rasterio.open(file_path) as src:
                    if not initialized:
                        dst_crs = src.crs
                        dst_transform = src.transform
                        dst_width, dst_height = src.width, src.height
                        initialized = True
                        break

    # Generate x and y coordinates
    x_coords = np.arange(dst_width) * dst_transform[0] + dst_transform[2]
    y_coords = np.arange(dst_height) * dst_transform[4] + dst_transform[5]

    # Process files
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            match = pattern.search(filename)
            if match:
                timestamp_str = match.group(1)

                # Detect timestamp format
                if 'T' in timestamp_str:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
                else:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d')

                # Read and reproject
                file_path = os.path.join(folder_path, filename)
                data_array = read_and_reproject_tif(
                    file_path, dst_crs, dst_transform, dst_width, dst_height
                )

                data_arrays.append(data_array)
                times.append(timestamp)

    # Stack into xarray
    data_array = np.stack(data_arrays, axis=0)
    data_xr = xr.DataArray(
        data_array,
        dims=['time', 'band', 'y', 'x'],
        coords={'time': times, 'x': x_coords, 'y': y_coords}
    )

    dataset = xr.Dataset({'data': data_xr})
    return dataset


def compute_percentile_dataset(field_ds, pol='VV', min_month=1, max_month=12):
    """
    Compute per-pixel percentile (0–1) over time
    for VV or VH values in a GEE xarray dataset.
    """
    # Extract VV (assuming VV is band index 0)
    if pol=='VV':
        data = field_ds['data'][:, 0, :, :].drop_duplicates(dim='time')# dims: time, y, x
    elif pol=='VH':
        data = field_ds['data'][:, 1, :, :].drop_duplicates(dim='time')   # dims: time, y, x
    elif pol=='VV-VH':
        data = 10**(field_ds['data'][:, 0 , :, :].drop_duplicates(dim='time')/10) / 10**(field_ds['data'][:, 1, :, :].drop_duplicates(dim='time')/10)
        
    month_mask = (data['time.month'] >= min_month) & (data['time.month'] <= max_month)
    data = data.sel(time=month_mask)    
    
    # Percentile ranks over time per pixel
    ranks = data.rank(dim='time', pct=True)
    
    # Assemble dataset with original coords
    if pol=='VV':

        return xr.Dataset(
            {"vv_percentile": ranks},
            coords=field_ds.coords
        )
    if pol=='VH':
        return xr.Dataset(
            {"vh_percentile": ranks},
            coords=field_ds.coords
        )    
    if pol=='VV-VH':
        return xr.Dataset(
            {"vv_vh_percentile": ranks},
            coords=field_ds.coords
        )   

def plot_pdf(ax, values):

    values = values[~np.isnan(values)]

    if len(values) == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "No data",
                ha="center", va="center",
                fontsize=9)
        return

    mu, sigma = np.mean(values), np.std(values)
    if sigma == 0:
        sigma = 1e-6

    x = np.linspace(values.min(), values.max(), 300)

    ax.hist(values, bins=30, density=True, alpha=0.6)
    ax.plot(x, norm.pdf(x, mu, sigma), linewidth=1.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4)