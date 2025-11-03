# swanmod/prepwind.py (EPSG:7850 regular grid interpolation)
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from pyproj import Transformer
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt

def Barrac2_slice(config, log, buffer_days=2):
    """
    Process BARRA-C2 wind data to SWAN .dat:
    - Slice original lon/lat grid
    - Generate regular EPSG:7850 grid with similar extent/resolution
    - Interpolate sliced wind to new EPSG:7850 grid
    """

    # --- Time setup
    start = datetime.fromisoformat(config["run_period"]["start"]) - timedelta(days=buffer_days)
    end   = datetime.fromisoformat(config["run_period"]["end"])   + timedelta(days=buffer_days)

    case_dir = Path(config["paths"]["case_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)
    out_file = case_dir / f"Barrac2_{start:%Y_%m_%d_%H%M}_to_{end:%Y_%m_%d_%H%M}.dat"

    log.info(f"🌀 Preparing wind data between {start} and {end}")
    log.info(f"💾 Output file: {out_file}")

    # --- Wind data folders
    root_dir = Path(config["paths"]["winddata_dir"])
    u_dir = root_dir / "uas" / "v20250528"
    v_dir = root_dir / "vas" / "v20250528"
    u_files = sorted(u_dir.glob("uas_*.nc"))
    v_files = sorted(v_dir.glob("vas_*.nc"))
    assert len(u_files) == len(v_files), "U/V file count mismatch"

    # --- Geographic domain bounds (lon/lat)
    lon_min, lon_max = 115.30, 116.14
    lat_min, lat_max = -32.49, -31.49

    lon_min_bg, lon_max_bg = 115.18, 116.26
    lat_min_bg, lat_max_bg = -32.65, -31.33

    # --- Preselect relevant monthly files
    files_to_open = []
    for uf, vf in zip(u_files, v_files):
        yyyymm = uf.name.split("_")[-1].split("-")[0]
        yr, mo = int(yyyymm[:4]), int(yyyymm[4:6])
        file_start = datetime(yr, mo, 1)
        file_end = datetime(yr + (mo == 12), (mo % 12) + 1, 1) - timedelta(seconds=1)
        if not (file_end < start or file_start > end):
            files_to_open.append((uf, vf))

    if not files_to_open:
        log.warning("⚠️ No wind files found for selected period.")
        return None

    # --- Load and slice first file to determine grid
    ds_u = xr.open_dataset(files_to_open[0][0])
    ds_v = xr.open_dataset(files_to_open[0][1])

    lon_sel = ds_u.lon.where((ds_u.lon >= lon_min) & (ds_u.lon <= lon_max), drop=True).values
    lat_sel = ds_u.lat.where((ds_u.lat >= lat_min) & (ds_u.lat <= lat_max), drop=True).values
    lon2d, lat2d = np.meshgrid(lon_sel, lat_sel)

    dx_deg = np.mean(np.diff(lon_sel))
    dy_deg = np.mean(np.diff(lat_sel))
    log.info(f"Lon/lat slice resolution: dx={dx_deg:.5f}°, dy={dy_deg:.5f}°")

    ds_u.close()
    ds_v.close()

    # --- Generate regular EPSG:7850 grid based on projected extent
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:7850", always_xy=True)
    x_proj, y_proj = transformer.transform(lon2d, lat2d)

    # Approximate spacing in meters
    dx_m = np.mean(np.diff(x_proj[0, :]))
    dy_m = np.mean(np.diff(y_proj[:, 0]))
    log.info(f"Target EPSG:7850 resolution: dx={dx_m:.1f} m, dy={dy_m:.1f} m")

    # Create new regular projected grid
    x_min_m, x_max_m = x_proj.min(), x_proj.max()
    y_min_m, y_max_m = y_proj.min(), y_proj.max()
    x_new = np.arange(x_min_m, x_max_m + dx_m, dx_m)
    y_new = np.arange(y_min_m, y_max_m + dy_m, dy_m)
    x_new2d, y_new2d = np.meshgrid(x_new, y_new)
    log.info(f"New projected grid: nx={len(x_new)}, ny={len(y_new)}")

    # --- Load all data and interpolate to new EPSG:7850 grid
    u_list, v_list = [], []
    for uf, vf in files_to_open:
        ds_u = xr.open_dataset(uf)
        ds_v = xr.open_dataset(vf)

        mask = (ds_u["time"].values >= np.datetime64(start)) & (ds_u["time"].values <= np.datetime64(end))
        if not np.any(mask):
            ds_u.close(); ds_v.close()
            continue

        u_sel = ds_u["uas"].isel(time=mask).sel(lat=slice(lat_min_bg, lat_max_bg), lon=slice(lon_min_bg, lon_max_bg))
        v_sel = ds_v["vas"].isel(time=mask).sel(lat=slice(lat_min_bg, lat_max_bg), lon=slice(lon_min_bg, lon_max_bg))

        # Project original lon/lat of sliced data to EPSG:7850
        lon2d_s, lat2d_s = np.meshgrid(u_sel.lon.values, u_sel.lat.values)
        x_s, y_s = transformer.transform(lon2d_s, lat2d_s)

        # Flatten for interpolation
        points = np.column_stack((x_s.ravel(), y_s.ravel()))
        n_time = u_sel.shape[0]

        for t in range(n_time):
            u_t = u_sel.isel(time=t).values.ravel()
            v_t = v_sel.isel(time=t).values.ravel()
            u_interp = griddata(points, u_t, (x_new2d, y_new2d), method="linear")
            v_interp = griddata(points, v_t, (x_new2d, y_new2d), method="linear")
            u_list.append(u_interp)
            v_list.append(v_interp)

        ds_u.close()
        ds_v.close()

    u_arr = np.array(u_list)  # shape: [time, ny, nx]
    v_arr = np.array(v_list)
    n_time = u_arr.shape[0]
    n_y, n_x = u_arr.shape[1], u_arr.shape[2]

    log.info(f"✅ Interpolated wind data: {n_time} timesteps, {n_x} x {n_y} grid starting/reso as {x_new.min()}, {y_new.min()}, {dx_m}, {dy_m}, {len(x_new)}, {len(y_new)}")

    # --- Write SWAN .dat file (regular projected grid)
    with open(out_file, "w") as f:
        for t in range(n_time):
            u_t = u_arr[t, ::-1, :]  # u/v are towards east/north
            v_t = v_arr[t, ::-1, :]
            for j in range(n_y):
                f.write(" ".join(f"{val:.2f}" for val in u_t[j, :]) + "\n")
            for j in range(n_y):
                f.write(" ".join(f"{val:.2f}" for val in v_t[j, :]) + "\n")

    log.info(f"💨 SWAN wind file written: {out_file.name}")
    return out_file, f"{start:%Y%m%d.%H%M%S}", f"{end:%Y%m%d.%H%M%S}"



def correct_barrac2_with_bom_single(config, log, buffer_days=2):
    """
    Correct gridded BARRA-C2 wind using BoM station observations and plot comparisons.

    Parameters
    ----------
    config : dict
        Contains run_period, paths, grid info, etc.
    log : logging.Logger
    buffer_days : int
        Extra days around run period to include.
    """
    output_dir = Path(config["paths"]["case_post_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- Time setup
    start = datetime.fromisoformat(config["run_period"]["start"]) - timedelta(days=buffer_days)
    end   = datetime.fromisoformat(config["run_period"]["end"]) + timedelta(days=buffer_days)

    case_dir = Path(config["paths"]["case_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)
    out_file = case_dir / f"Barrac2_corrected_{start:%Y_%m_%d_%H%M}_to_{end:%Y_%m_%d_%H%M}.dat"

    log.info(f"🌀 Preparing wind data between {start} and {end}")
    log.info(f"💾 Output file: {out_file}")

    # --- BOM stations info
    stations = [
        {"name": "ROTTNEST", "lon": 115.5022, "lat": -32.0069,
         "bom_file": Path(f"{config['paths']['obsdata_dir']}/BOM/ROTTNEST.nc"), "height": 43.1},
        {"name": "OCEANREEF", "lon": 115.7278, "lat": -31.7594,
         "bom_file": Path(f"{config['paths']['obsdata_dir']}/BOM/OCEANREEF.nc"), "height": 10.0},
        {"name": "SWANBOURNE", "lon": 115.7619, "lat": -31.9560,
         "bom_file": Path(f"{config['paths']['obsdata_dir']}/BOM/SWANBOURNE.nc"), "height": 41.0},
        {"name": "GARDENISLAND", "lon": 115.6839, "lat": -32.2433,
         "bom_file": Path(f"{config['paths']['obsdata_dir']}/BOM/GARDENISLAND.nc"), "height": 6.0},
    ]

    # --- Wind data folders
    root_dir = Path(config["paths"]["winddata_dir"])
    u_dir = root_dir / "uas" / "v20250528"
    v_dir = root_dir / "vas" / "v20250528"
    u_files = sorted(u_dir.glob("uas_*.nc"))
    v_files = sorted(v_dir.glob("vas_*.nc"))
    assert len(u_files) == len(v_files), "U/V file count mismatch"

    # --- Geographic domain bounds
    lon_min, lon_max = 115.30, 116.14
    lat_min, lat_max = -32.49, -31.49

    lon_min_bg, lon_max_bg = 115.18, 116.26
    lat_min_bg, lat_max_bg = -32.65, -31.33

    # --- Preselect relevant monthly files
    files_to_open = []
    for uf, vf in zip(u_files, v_files):
        yyyymm = uf.name.split("_")[-1].split("-")[0]
        yr, mo = int(yyyymm[:4]), int(yyyymm[4:6])
        file_start = datetime(yr, mo, 1)
        file_end = datetime(yr + (mo == 12), (mo % 12) + 1, 1) - timedelta(seconds=1)
        if not (file_end < start or file_start > end):
            files_to_open.append((uf, vf))

    if not files_to_open:
        log.warning("⚠️ No BARRA wind files found for selected period.")
        return None

    # --- Load first file to define grid
    ds_u = xr.open_dataset(files_to_open[0][0])
    ds_v = xr.open_dataset(files_to_open[0][1])
    lon_sel = ds_u.lon.where((ds_u.lon >= lon_min) & (ds_u.lon <= lon_max), drop=True).values
    lat_sel = ds_u.lat.where((ds_u.lat >= lat_min) & (ds_u.lat <= lat_max), drop=True).values
    lon2d, lat2d = np.meshgrid(lon_sel, lat_sel)
    dx_deg = np.mean(np.diff(lon_sel))
    dy_deg = np.mean(np.diff(lat_sel))
    log.info(f"Lon/lat slice resolution: dx={dx_deg:.5f}°, dy={dy_deg:.5f}°")
    ds_u.close(); ds_v.close()

    # --- Project to EPSG:7850
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:7850", always_xy=True)
    x_proj, y_proj = transformer.transform(lon2d, lat2d)
    dx_m = np.mean(np.diff(x_proj[0, :]))
    dy_m = np.mean(np.diff(y_proj[:, 0]))
    log.info(f"Target EPSG:7850 resolution: dx={dx_m:.1f} m, dy={dy_m:.1f} m")

    # --- Create regular projected grid
    x_min_m, x_max_m = x_proj.min(), x_proj.max()
    y_min_m, y_max_m = y_proj.min(), y_proj.max()
    x_new = np.arange(x_min_m, x_max_m + dx_m, dx_m)
    y_new = np.arange(y_min_m, y_max_m + dy_m, dy_m)
    x_new2d, y_new2d = np.meshgrid(x_new, y_new)
    nx, ny = len(x_new), len(y_new)
    log.info(f"New projected grid: nx={nx}, ny={ny}")

    # --- Load all files and interpolate to regular grid
    u_list, v_list = [], []
    for uf, vf in files_to_open:
        ds_u = xr.open_dataset(uf)
        ds_v = xr.open_dataset(vf)
        mask = (ds_u["time"].values >= np.datetime64(start)) & (ds_u["time"].values <= np.datetime64(end))
        if not np.any(mask):
            ds_u.close(); ds_v.close()
            continue
        u_sel = ds_u["uas"].isel(time=mask).sel(lat=slice(lat_min_bg, lat_max_bg), lon=slice(lon_min_bg, lon_max_bg))
        v_sel = ds_v["vas"].isel(time=mask).sel(lat=slice(lat_min_bg, lat_max_bg), lon=slice(lon_min_bg, lon_max_bg))
        lon2d_s, lat2d_s = np.meshgrid(u_sel.lon.values, u_sel.lat.values)
        x_s, y_s = transformer.transform(lon2d_s, lat2d_s)
        points = np.column_stack((x_s.ravel(), y_s.ravel()))
        n_time_file = u_sel.shape[0]
        for t in range(n_time_file):
            u_t = u_sel.isel(time=t).values.ravel()
            v_t = v_sel.isel(time=t).values.ravel()
            u_interp = griddata(points, u_t, (x_new2d, y_new2d), method="linear")
            v_interp = griddata(points, v_t, (x_new2d, y_new2d), method="linear")
            u_list.append(u_interp)
            v_list.append(v_interp)
        ds_u.close(); ds_v.close()

    u_arr = np.array(u_list)
    v_arr = np.array(v_list)
    n_time = u_arr.shape[0]
    u_arr_orig = u_arr.copy()
    v_arr_orig = v_arr.copy()
    log.info(f"✅ Interpolated BARRA-C2: {n_time} timesteps, {nx} x {ny} grid")

    # --- Time vector
    time_mod = pd.date_range(start=start, end=end, periods=n_time)

    # --- Correct using BOM stations
    fric_coe = 0.03
    # Store interpolated station data for plotting
    station_data = []

    for st in stations:
        log.info(f"🔹 Processing station: {st['name']}")
        ds = xr.open_dataset(st["bom_file"])
        ds = ds.sortby("time")
        if st["name"] == "GARDENISLAND":
            ds["wind-speed_raw"].values = ds["wind-speed_raw"].values * 3.6
            bom = pd.DataFrame({
                "time": pd.to_datetime(ds["time"].values),
                "spd_obs": ds["wind-speed_raw"].values / 3.6,  # km/h -> m/s
                "dir_obs": ds["wind-direction_raw"].values
            })
        elif st["name"] == "ROTTNEST":
            bom = pd.DataFrame({
                "time": pd.to_datetime(ds["time"].values),
                "spd_obs": ds["speed"].values / 3.6,  # km/h -> m/s
                "dir_obs": ds["dir"].values
            })
        elif st["name"] == "SWANBOURNE":
            bom = pd.DataFrame({
                "time": pd.to_datetime(ds["time"].values),
                "spd_obs": ds["speed"].values / 3.6,  # km/h -> m/s
                "dir_obs": ds["dir"].values
            })
        elif st["name"] == "OCEANREEF":
            bom = pd.DataFrame({
                "time": pd.to_datetime(ds["time"].values),
                "spd_obs": ds["speed"].values / 3.6,  # km/h -> m/s
                "dir_obs": ds["dir"].values
            })
        else:
            log.info(f"❌ Failed to read BoM data, stations not found")
        
        ds.close()
        bom["time"] = (pd.to_datetime(bom["time"]).dt.tz_localize("Etc/GMT-8")
                       .dt.tz_convert("UTC").dt.tz_localize(None))
        # Height correction to 10 m
        z_obs = st["height"]
        z_ref = 10
        bom["spd_10m"] = bom["spd_obs"] * np.log(z_ref/fric_coe) / np.log(z_obs/fric_coe)
        bom["u_obs"] = -bom["spd_10m"] * np.sin(np.radians(bom["dir_obs"]))
        bom["v_obs"] = -bom["spd_10m"] * np.cos(np.radians(bom["dir_obs"]))

        # Project station coordinates to grid
        x_st, y_st = transformer.transform(st["lon"], st["lat"])
        ix = np.argmin(np.abs(x_new - x_st))
        iy = np.argmin(np.abs(y_new - y_st))

        u_mod = u_arr_orig[:, iy, ix]
        v_mod = v_arr_orig[:, iy, ix]
        bom_interp_u = np.interp(pd.to_numeric(time_mod), pd.to_numeric(bom["time"]), bom["u_obs"])
        bom_interp_v = np.interp(pd.to_numeric(time_mod), pd.to_numeric(bom["time"]), bom["v_obs"])
        # Compute correction ratios
        ratio_u = bom_interp_u / u_mod
        ratio_v = bom_interp_v / v_mod
        ratio_u[np.isnan(ratio_u)] = 1.0
        ratio_v[np.isnan(ratio_v)] = 1.0

        # Apply correction only locally (not overwriting global array)
        u_corr_local = u_mod * ratio_u
        v_corr_local = v_mod * ratio_v

        # Store data for plotting
        station_data.append({
            "name": st["name"],
            "u_orig": u_mod,
            "v_orig": v_mod,
            "u_corr": u_corr_local,
            "v_corr": v_corr_local,
            "u_obs": bom_interp_u,
            "v_obs": bom_interp_v
        })

    # --- Apply correction to the whole field using 4-station ratios
    log.info("⚙️ Applying multi-station correction to full BARRA-C2 field")

    station_points = []
    ratio_u_all = []
    ratio_v_all = []

    for st_data, st in zip(station_data, stations):
        # Station projected coordinates
        x_st, y_st = transformer.transform(st["lon"], st["lat"])
        station_points.append([x_st, y_st])
        # Each station ratio as function of time
        ratio_u_all.append(st_data["u_corr"] / st_data["u_orig"])
        ratio_v_all.append(st_data["v_corr"] / st_data["v_orig"])

    station_points = np.array(station_points)
    ratio_u_all = np.array(ratio_u_all)  # shape: (n_stations, n_time)
    ratio_v_all = np.array(ratio_v_all)

    for t in range(n_time):
        ratios_u_t = ratio_u_all[:, t]
        ratios_v_t = ratio_v_all[:, t]
        # Interpolate to full grid
        ratio_u_field = griddata(station_points, ratios_u_t, (x_new2d, y_new2d), method="nearest")
        ratio_v_field = griddata(station_points, ratios_v_t, (x_new2d, y_new2d), method="nearest")
        # #optional gaussian filter ratio
        # from scipy.ndimage import gaussian_filter
        # ratio_u_field = gaussian_filter(ratio_u_field, sigma=3)
        # ratio_v_field = gaussian_filter(ratio_v_field, sigma=3)
        # #optioanl
        # ratio_u_field = np.clip(ratio_u_field, 0.5, 2.0)
        # ratio_v_field = np.clip(ratio_v_field, 0.5, 2.0)
        
        # Apply correction
        u_arr[t, :, :] = u_arr_orig[t, :, :] * ratio_u_field
        v_arr[t, :, :] = v_arr_orig[t, :, :] * ratio_v_field

    log.info("✅ Multi-station correction applied to wind field")

    # --- Recalculate final corrected values at stations for comparison (all time steps)
    for st_data, st in zip(station_data, stations):
        x_st, y_st = transformer.transform(st["lon"], st["lat"])
        u_corr_final = np.zeros(n_time)
        v_corr_final = np.zeros(n_time)

        for t in range(n_time):
            u_corr_final[t] = griddata(
                (x_new2d.flatten(), y_new2d.flatten()),
                u_arr[t, :, :].flatten(),
                (x_st, y_st),
                method="nearest"
            )
            v_corr_final[t] = griddata(
                (x_new2d.flatten(), y_new2d.flatten()),
                v_arr[t, :, :].flatten(),
                (x_st, y_st),
                method="nearest"
            )

        st_data["u_corr_final"] = u_corr_final
        st_data["v_corr_final"] = v_corr_final


    # --- Multi-station comparison plot (8 rows × 1 column)
    n_stations = len(station_data)
    fig, axes = plt.subplots(8, 1, figsize=(15, 20), sharex=True)

    for i, st_data in enumerate(station_data):
        name = st_data["name"]

        # --- Wind speed (rows 0–3)
        ax_spd = axes[i]
        spd_orig = np.sqrt(st_data["u_orig"]**2 + st_data["v_orig"]**2)
        spd_corr = np.sqrt(st_data["u_corr_final"]**2 + st_data["v_corr_final"]**2)
        spd_obs  = np.sqrt(st_data["u_obs"]**2 + st_data["v_obs"]**2)

        ax_spd.plot(time_mod, spd_orig, label="BARRA-C2 original", color="C0")
        ax_spd.plot(time_mod, spd_corr, label="BARRA-C2 corrected", color="C1")
        ax_spd.plot(time_mod, spd_obs,  label="BoM observed", color="C2", linestyle="--")
        ax_spd.set_ylabel(f"{name}\nSpeed [m/s]")
        ax_spd.grid(True)
        if i == 0:
            ax_spd.legend(fontsize=8)

        # --- Wind direction (rows 4–7, nautical convention: 0° from north)
        ax_dir = axes[i + 4]
        dir_orig = (np.degrees(np.arctan2(st_data["u_orig"], st_data["v_orig"])) + 360) % 360
        dir_corr = (np.degrees(np.arctan2(st_data["u_corr_final"], st_data["v_corr_final"])) + 360) % 360
        dir_obs  = (np.degrees(np.arctan2(st_data["u_obs"], st_data["v_obs"])) + 360) % 360

        ax_dir.plot(time_mod, dir_orig, label="BARRA-C2 original", color="C0")
        ax_dir.plot(time_mod, dir_corr, label="BARRA-C2 corrected", color="C1")
        ax_dir.plot(time_mod, dir_obs,  label="BoM observed", color="C2", linestyle="--")
        ax_dir.set_ylabel(f"{name}\nDir [°]")
        ax_dir.set_ylim(0, 360)
        ax_dir.grid(True)
        if i == 0:
            ax_dir.legend(fontsize=8)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    out_path = output_dir / "BARRAC2_vs_BoM_Wind_single.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ 8-row comparison plot saved: {out_path}")


    # --- Write corrected SWAN .dat
    with open(out_file, "w") as f:
        for t in range(n_time):
            u_t = u_arr[t, ::-1, :]
            v_t = v_arr[t, ::-1, :]
            for j in range(ny):
                f.write(" ".join(f"{val:.2f}" for val in u_t[j, :]) + "\n")
            for j in range(ny):
                f.write(" ".join(f"{val:.2f}" for val in v_t[j, :]) + "\n")

    log.info(f"✅ Corrected SWAN wind file written: {out_file.name}")
    return out_file, f"{start:%Y%m%d.%H%M%S}", f"{end:%Y%m%d.%H%M%S}"




def correct_barrac2_with_bom_multi(config, log, buffer_days=2):
    """
    Correct BARRA-C2 wind using multiple BoM stations simultaneously (spatially interpolated),
    and plot 8 subplots: original vs corrected BARRA-C2 and BoM obs for speed & direction at 4 stations.
    """
    output_dir = Path(config["paths"]["case_post_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- Time setup
    start = datetime.fromisoformat(config["run_period"]["start"]) - timedelta(days=buffer_days)
    end   = datetime.fromisoformat(config["run_period"]["end"]) + timedelta(days=buffer_days)
    case_dir = Path(config["paths"]["case_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)
    out_file = case_dir / f"Barrac2_corrected_{start:%Y_%m_%d_%H%M}_to_{end:%Y_%m_%d_%H%M}.dat"
    log.info(f"🌀 Preparing wind data between {start} and {end}")
    log.info(f"💾 Output file: {out_file}")

    # --- BOM stations info
    stations = [
        {"name": "ROTTNEST", "lon": 115.5022, "lat": -32.0069,
         "bom_file": Path(f"{config['paths']['obsdata_dir']}/BOM/ROTTNEST.nc"), "height": 43.1},
        {"name": "OCEANREEF", "lon": 115.7278, "lat": -31.7594,
         "bom_file": Path(f"{config['paths']['obsdata_dir']}/BOM/OCEANREEF.nc"), "height": 10.0},
        {"name": "SWANBOURNE", "lon": 115.7619, "lat": -31.9560,
         "bom_file": Path(f"{config['paths']['obsdata_dir']}/BOM/SWANBOURNE.nc"), "height": 41.0},
        {"name": "GARDENISLAND", "lon": 115.6839, "lat": -32.2433,
         "bom_file": Path(f"{config['paths']['obsdata_dir']}/BOM/GARDENISLAND.nc"), "height": 6.0},
    ]

    # --- Wind data folders
    root_dir = Path(config["paths"]["winddata_dir"])
    u_dir = root_dir / "uas" / "v20250528"
    v_dir = root_dir / "vas" / "v20250528"
    u_files = sorted(u_dir.glob("uas_*.nc"))
    v_files = sorted(v_dir.glob("vas_*.nc"))
    assert len(u_files) == len(v_files), "U/V file count mismatch"

    # --- Geographic domain bounds
    lon_min, lon_max = 115.30, 116.14
    lat_min, lat_max = -32.49, -31.49

    lon_min_bg, lon_max_bg = 115.18, 116.26
    lat_min_bg, lat_max_bg = -32.65, -31.33

    # --- Preselect relevant files
    files_to_open = []
    for uf, vf in zip(u_files, v_files):
        yyyymm = uf.name.split("_")[-1].split("-")[0]
        yr, mo = int(yyyymm[:4]), int(yyyymm[4:6])
        file_start = datetime(yr, mo, 1)
        file_end = datetime(yr + (mo == 12), (mo % 12) + 1, 1) - timedelta(seconds=1)
        if not (file_end < start or file_start > end):
            files_to_open.append((uf, vf))
    if not files_to_open:
        log.warning("⚠️ No BARRA wind files found for selected period.")
        return None

    # --- Load first file to define grid
    ds_u = xr.open_dataset(files_to_open[0][0])
    ds_v = xr.open_dataset(files_to_open[0][1])
    lon_sel = ds_u.lon.where((ds_u.lon >= lon_min) & (ds_u.lon <= lon_max), drop=True).values
    lat_sel = ds_u.lat.where((ds_u.lat >= lat_min) & (ds_u.lat <= lat_max), drop=True).values
    lon2d, lat2d = np.meshgrid(lon_sel, lat_sel)
    ds_u.close(); ds_v.close()

    # --- Project to EPSG:7850
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:7850", always_xy=True)
    x_proj, y_proj = transformer.transform(lon2d, lat2d)
    dx_m = np.mean(np.diff(x_proj[0, :]))
    dy_m = np.mean(np.diff(y_proj[:, 0]))
    x_min_m, x_max_m = x_proj.min(), x_proj.max()
    y_min_m, y_max_m = y_proj.min(), y_proj.max()
    x_new = np.arange(x_min_m, x_max_m + dx_m, dx_m)
    y_new = np.arange(y_min_m, y_max_m + dy_m, dy_m)
    x_new2d, y_new2d = np.meshgrid(x_new, y_new)
    nx, ny = len(x_new), len(y_new)

    # --- Load all files and interpolate
    u_list, v_list = [], []
    for uf, vf in files_to_open:
        ds_u = xr.open_dataset(uf); ds_v = xr.open_dataset(vf)
        mask = (ds_u["time"].values >= np.datetime64(start)) & (ds_u["time"].values <= np.datetime64(end))
        if not np.any(mask):
            ds_u.close(); ds_v.close(); continue
        u_sel = ds_u["uas"].isel(time=mask).sel(lat=slice(lat_min_bg, lat_max_bg), lon=slice(lon_min_bg, lon_max_bg))
        v_sel = ds_v["vas"].isel(time=mask).sel(lat=slice(lat_min_bg, lat_max_bg), lon=slice(lon_min_bg, lon_max_bg))
        lon2d_s, lat2d_s = np.meshgrid(u_sel.lon.values, u_sel.lat.values)
        x_s, y_s = transformer.transform(lon2d_s, lat2d_s)
        points = np.column_stack((x_s.ravel(), y_s.ravel()))
        n_time_file = u_sel.shape[0]
        for t in range(n_time_file):
            u_interp = griddata(points, u_sel.isel(time=t).values.ravel(), (x_new2d, y_new2d), method="linear")
            v_interp = griddata(points, v_sel.isel(time=t).values.ravel(), (x_new2d, y_new2d), method="linear")
            u_list.append(u_interp); v_list.append(v_interp)
        ds_u.close(); ds_v.close()
    u_arr = np.array(u_list); v_arr = np.array(v_list)
    n_time = u_arr.shape[0]; u_arr_orig = u_arr.copy(); v_arr_orig = v_arr.copy()

    # --- Time vector
    time_mod = pd.date_range(start=start, end=end, periods=n_time)

    # --- Prepare station observations and ratios
    fric_coe = 0.03
    station_points = []
    ratio_u_station = []
    ratio_v_station = []
    bom_list = []  # Store BoM interpolated data for plotting

    for st in stations:
        ds = xr.open_dataset(st["bom_file"])
        ds = ds.sortby("time")
        if st["name"] == "GARDENISLAND":
            ds["wind-speed_raw"].values = ds["wind-speed_raw"].values * 3.6
            bom = pd.DataFrame({
                "time": pd.to_datetime(ds["time"].values),
                "spd_obs": ds["wind-speed_raw"].values / 3.6,  # km/h -> m/s
                "dir_obs": ds["wind-direction_raw"].values
            })
        else:
            bom = pd.DataFrame({
                "time": pd.to_datetime(ds["time"].values),
                "spd_obs": ds["speed"].values / 3.6,  # km/h -> m/s
                "dir_obs": ds["dir"].values
            })
        ds.close()
        bom["time"] = (pd.to_datetime(bom["time"]).dt.tz_localize("Etc/GMT-8")
                       .dt.tz_convert("UTC").dt.tz_localize(None))
        z_obs = st["height"]; z_ref = 10
        bom["spd_10m"] = bom["spd_obs"] * np.log(z_ref/fric_coe) / np.log(z_obs/fric_coe)
        bom["u_obs"] = -bom["spd_10m"] * np.sin(np.radians(bom["dir_obs"]))
        bom["v_obs"] = -bom["spd_10m"] * np.cos(np.radians(bom["dir_obs"]))

        # Interpolate to model time
        u_bom = np.interp(pd.to_numeric(time_mod), pd.to_numeric(bom["time"]), bom["u_obs"])
        v_bom = np.interp(pd.to_numeric(time_mod), pd.to_numeric(bom["time"]), bom["v_obs"])
        bom_list.append(pd.DataFrame({"time": time_mod, "u_obs": u_bom, "v_obs": v_bom}))

        # Nearest grid point
        x_st, y_st = transformer.transform(st["lon"], st["lat"])
        station_points.append((x_st, y_st))
        ix = np.argmin(np.abs(x_new - x_st))
        iy = np.argmin(np.abs(y_new - y_st))
        ratio_u_station.append(u_bom / u_arr[:, iy, ix])
        ratio_v_station.append(v_bom / v_arr[:, iy, ix])

    station_points = np.array(station_points)

    # --- Apply interpolated ratio
    for t in range(n_time):
        ratios_u = np.array([ratio_u_station[i][t] for i in range(len(stations))])
        ratios_v = np.array([ratio_v_station[i][t] for i in range(len(stations))])
        u_corr_field = griddata(station_points, ratios_u, (x_new2d, y_new2d), method="nearest")
        v_corr_field = griddata(station_points, ratios_v, (x_new2d, y_new2d), method="nearest")
        # # Apply Gaussian filter (sigma controls the smoothing)
        # # sigma = number of grid points for smoothing, try small values first, e.g., 1~3
        # sigma = 1
        # from scipy.ndimage import gaussian_filter
        # u_corr_field = gaussian_filter(u_corr_field, sigma=sigma)
        # v_corr_field = gaussian_filter(v_corr_field, sigma=sigma)
        # # optional
        # u_corr_field = np.clip(u_corr_field, 0.5, 2.0)
        # v_corr_field = np.clip(v_corr_field, 0.5, 2.0)
        # if np.nanmax(u_corr_field)>20.0 or np.nanmax(v_corr_field)>20.0:
        #     log.warning(f"⚠️ wind correction extreme ratio appeared")
        u_arr[t, :, :] *= u_corr_field
        v_arr[t, :, :] *= v_corr_field

        # --- Plot 8 rows x 1 column (speed & direction for 4 stations)
    fig, axes = plt.subplots(8, 1, figsize=(15, 20), sharex=True)
    for i, st in enumerate(stations):
        # Nearest grid point
        x_st, y_st = transformer.transform(st["lon"], st["lat"])
        ix = np.argmin(np.abs(x_new - x_st))
        iy = np.argmin(np.abs(y_new - y_st))

        # BARRA-C2
        u_orig = u_arr_orig[:, iy, ix]
        v_orig = v_arr_orig[:, iy, ix]
        u_corr = u_arr[:, iy, ix]
        v_corr = v_arr[:, iy, ix]

        # BoM
        u_bom = bom_list[i]["u_obs"].values
        v_bom = bom_list[i]["v_obs"].values

        # --- Speed subplot (top 4)
        ax_spd = axes[i]
        ax_spd.plot(time_mod, np.sqrt(u_orig**2 + v_orig**2), label="BARRA-C2 original", color="C0")
        ax_spd.plot(time_mod, np.sqrt(u_corr**2 + v_corr**2), label="BARRA-C2 corrected", color="C1")
        ax_spd.plot(time_mod, np.sqrt(u_bom**2 + v_bom**2), label="BoM obs", color="C2", linestyle="--")
        ax_spd.set_ylabel(f"{st['name']} Speed [m/s]")
        ax_spd.grid(True)
        if i == 0: ax_spd.legend(fontsize=8)

        # --- Direction subplot (bottom 4)
        ax_dir = axes[i+4]
        dir_orig = (np.degrees(np.arctan2(u_orig, v_orig)) + 360) % 360
        dir_corr = (np.degrees(np.arctan2(u_corr, v_corr)) + 360) % 360
        dir_bom = (np.degrees(np.arctan2(u_bom, v_bom)) + 360) % 360
        ax_dir.plot(time_mod, dir_orig, label="BARRA-C2 original", color="C0")
        ax_dir.plot(time_mod, dir_corr, label="BARRA-C2 corrected", color="C1")
        ax_dir.plot(time_mod, dir_bom, label="BoM obs", color="C2", linestyle="--")
        ax_dir.set_ylabel(f"{st['name']} Dir [°]")
        ax_dir.set_ylim(0, 360)
        ax_dir.grid(True)
        if i == 0: ax_dir.legend(fontsize=8)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    out_path = output_dir / "BARRAC2_vs_BoM_Wind_multi.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Comparison BARRAC2 vs BoM wind saved: {out_path}")


    # --- plot fields of Last time step
    t_last = -1

    # Compute wind speed and direction
    speed_orig = np.sqrt(u_arr_orig[t_last, :, :]**2 + v_arr_orig[t_last, :, :]**2)
    dir_orig   = (np.degrees(np.arctan2(u_arr_orig[t_last, :, :], v_arr_orig[t_last, :, :])) + 360) % 360

    speed_corr = np.sqrt(u_arr[t_last, :, :]**2 + v_arr[t_last, :, :]**2)
    dir_corr   = (np.degrees(np.arctan2(u_arr[t_last, :, :], v_arr[t_last, :, :])) + 360) % 360

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    # --- Original wind
    ax = axes[0]
    pcm = ax.pcolormesh(x_new2d, y_new2d, speed_orig, cmap="viridis", shading="auto", vmin=0, vmax=15)
    q = ax.quiver(
        x_new2d, y_new2d,
        u_arr_orig[t_last, :, :], v_arr_orig[t_last, :, :],
        scale=200, color="white", pivot="middle"
    )
    ax.set_title("BARRA-C2 Original Wind")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    fig.colorbar(pcm, ax=ax, label="Speed [m/s]")

    # --- Corrected wind
    ax = axes[1]
    pcm = ax.pcolormesh(x_new2d, y_new2d, speed_corr, cmap="viridis", shading="auto", vmin=0, vmax=15)
    q = ax.quiver(
        x_new2d, y_new2d,
        u_arr[t_last, :, :], v_arr[t_last, :, :],
        scale=200, color="white", pivot="middle"
    )
    ax.set_title("BARRA-C2 Corrected Wind")
    ax.set_xlabel("X [m]")
    fig.colorbar(pcm, ax=ax, label="Speed [m/s]")

    plt.tight_layout()
    out_path = output_dir / f"WindField_correction_{time_mod[t_last]}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Wind field speed & direction plot saved: {out_path}")


    # --- Write corrected SWAN .dat
    with open(out_file, "w") as f:
        for t in range(n_time):
            u_t = u_arr[t, ::-1, :]
            v_t = v_arr[t, ::-1, :]
            for j in range(ny):
                f.write(" ".join(f"{val:.2f}" for val in u_t[j, :]) + "\n")
            for j in range(ny):
                f.write(" ".join(f"{val:.2f}" for val in v_t[j, :]) + "\n")

    log.info(f"✅ Corrected SWAN wind file written: {out_file.name}")
    return out_file, f"{start:%Y%m%d.%H%M%S}", f"{end:%Y%m%d.%H%M%S}"

