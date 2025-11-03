# swanmod/prep_boundary.py
import os
import glob
import numpy as np
import xarray as xr
from scipy.spatial.distance import euclidean
from pathlib import Path
from datetime import datetime, timedelta
from pyproj import Transformer
import cftime
# from swanmod.utilities import setup_logger

# log = setup_logger(__name__)

def extract_boundary_data(
    config, log, n_nodes=50, buffer_days=1,
    lon_var="lon", lat_var="lat",
    hs_var="HS", tpp_var="TPP", pkd_var="PEAKD", pkdsp_var="PEAKDSPR"
):
    """
    Extract open boundary data from NetCDF WWM output and fort.14 node IDs,
    interpolate or sample boundary nodes, and write TPAR + boundary list files
    for SWAN open boundary forcing.
    """

    # ------------------------------
    # 1. Setup paths and time window
    # ------------------------------
    start = datetime.fromisoformat(config["run_period"]["start"]) - timedelta(days=buffer_days)
    end = datetime.fromisoformat(config["run_period"]["end"]) + timedelta(days=buffer_days)

    fort14_file = Path(config["paths"]["oldfort_dir"]) / "oldfort.14"
    data_dir = Path(config["paths"]["bounddata_dir"])
    case_dir = Path(config["paths"]["case_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # 2. Read open boundary node IDs from fort.14
    # ------------------------------
    def read_open_boundary_nodes(fort14_path: Path) -> list[int]:
        with open(fort14_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "Number of open boundaries" in line:
                node_count = int(lines[i + 2].split()[0])
                start_idx = i + 3
                return [int(lines[start_idx + j].strip()) for j in range(node_count)]
        raise ValueError("Open boundary section not found in fort.14")

    open_nodes = read_open_boundary_nodes(fort14_file)
    log.info(f"✅ Read {len(open_nodes)} open boundary nodes from fort.14")

    # ------------------------------
    # 3. Load coordinates from any available NetCDF file
    # ------------------------------
    nc_candidates = list(data_dir.glob("*/*.nc")) or list(data_dir.glob("*.nc"))
    if not nc_candidates:
        raise FileNotFoundError(f"No NetCDF files found in {data_dir}")

    ds_sample = xr.open_dataset(nc_candidates[0])
    lon, lat = ds_sample[lon_var].values, ds_sample[lat_var].values
    ds_sample.close()

    open_indices = np.array(open_nodes) - 1  # convert 1-based ADCIRC → 0-based Python index

    # ------------------------------
    # 4. Project coordinates (for distance spacing)
    # ------------------------------
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28350", always_xy=True)
    x_open, y_open = transformer.transform(lon[open_indices], lat[open_indices])
    coords_m = np.column_stack((x_open, y_open))

    # cumulative distance along boundary
    distances = np.insert(
        np.cumsum([euclidean(coords_m[i], coords_m[i - 1]) for i in range(1, len(coords_m))]),
        0, 0.0
    )

    even_idx = np.linspace(5, len(open_indices) - 5, n_nodes, dtype=int)
    selected_indices = open_indices[even_idx]
    selected_distances = distances[even_idx]
    log.info(f"✅ Selected {len(selected_indices)} evenly spaced boundary nodes")

    # ------------------------------
    # 5. Select relevant NetCDF files by date (YYYYMMDD in filename)
    # ------------------------------
    years = range(start.year, end.year + 1)
    nc_files = []
    for yr in years:
        for f in sorted((data_dir / str(yr)).glob("*.nc")):
            digits = "".join(c for c in f.name if c.isdigit())
            if len(digits) >= 8:
                file_date = datetime.strptime(digits[:8], "%Y%m%d")
                if start.date() <= file_date.date() <= end.date():
                    nc_files.append(f)

    if not nc_files:
        log.warning(f"⚠️ No NetCDF files found for {start:%Y-%m-%d} → {end:%Y-%m-%d}")
        return []

    log.info(f"✅ Found {len(nc_files)} NetCDF files covering requested period")

    # ------------------------------
    # 6. Extract time series for each node
    # ------------------------------
    times = []
    node_data = {idx: [] for idx in selected_indices}

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)

        # identify time variable
        for tvar in ["time", "ocean_time"]:
            if tvar in ds.variables:
                t_all = ds[tvar].values
                break
        else:
            t_all = list(ds.coords.values())[0].values

        valid_idx = [
            i for i, t in enumerate(t_all)
            if start <= t.astype("datetime64[s]").astype(datetime) <= end
        ]
        if not valid_idx:
            ds.close()
            continue

        times.extend([t_all[i].astype("datetime64[s]").astype(datetime) for i in valid_idx])
        hs_all = ds[hs_var][valid_idx, :].values
        tpp_all = ds[tpp_var][valid_idx, :].values
        pkd_all = ds[pkd_var][valid_idx, :].values
        pkdspr_all = ds[pkdsp_var][valid_idx, :].values
        ds.close()

        for idx in selected_indices:
            i = int(idx)
            hs, tpp, pkd, pkdspr = hs_all[:, i], tpp_all[:, i], pkd_all[:, i], pkdspr_all[:, i]
            node_data[idx].append(np.stack([hs, tpp, pkd, pkdspr], axis=1))

    # ------------------------------
    # 7. Write TPAR files and boundary list
    # ------------------------------
    boundary_list = []
    for i, idx in enumerate(selected_indices):
        data = np.vstack(node_data[idx])
        fname = f"node_{idx+1}_dist_{selected_distances[i]:.2f}m.txt"
        out_path = case_dir / fname

        with open(out_path, "w") as f:
            f.write("TPAR\n")
            for t, row in zip(times, data):
                if isinstance(t, (np.datetime64, datetime, cftime.datetime)):
                    dt = t.astype("datetime64[s]").astype(datetime) if isinstance(t, np.datetime64) else t
                    f.write(f"{dt:%Y%m%d.%H%M%S} {row[0]:.3f} {row[1]:.3f} {row[2]:.1f} {row[3]:.1f}\n")

        log.info(f"🟢 Saved {fname}")
        boundary_list.append((selected_distances[i], fname))

    # SWAN input list
    list_path = case_dir / "boundary_list.txt"
    with open(list_path, "w") as f:
        for j, (dist, fname) in enumerate(boundary_list):
            end_char = " &\n" if j < len(boundary_list) - 1 else "\n"
            f.write(f" {dist:.2f} '{fname}' 1{end_char}")
    log.info(f"✅ Boundary list saved: {list_path}")

    # ------------------------------
    # 8. Save selected node coordinates
    # ------------------------------
    coord_path = case_dir / "selected_node_coords.txt"
    with open(coord_path, "w") as f:
        for idx in selected_indices:
            x_m, y_m = transformer.transform(lon[idx], lat[idx])
            f.write(f"{x_m:.3f} {y_m:.3f}\n")
    log.info(f"✅ Saved coordinates: {coord_path}")

    return boundary_list
