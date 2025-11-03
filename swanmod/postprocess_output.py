from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import geopandas as gpd
import meshio
import re
import pyvista as pv
from rasterio.features import rasterize
from rasterio.transform import from_origin
from skimage.measure import block_reduce
from scipy.interpolate import griddata
from shapely.geometry import box
from datetime import datetime, timedelta
import xarray as xr

def postprocess_field_plot(config, log, target_time):
    """
    Post-process SWAN .vtu output and plot wave and wind fields at the latest timestep.
    """

    case_name = config["project_name"]
    log.info(f"▶ Starting post-processing for {case_name}")

    case_dir = Path(config["paths"]["case_dir"])
    input_dir = Path(config["paths"]["oldfort_dir"])
    output_dir = Path(config["paths"]["case_out_dir"])
    post_dir = Path(config["paths"]["case_post_dir"])
    results_dir = case_dir / f"{config["project_name"]}_output"

    output_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    # # ─── Locate latest VTU file numerically ───
    # def extract_number(fpath):
    #     match = re.search(r"_(\d+)\.vtu$", fpath.name)
    #     return int(match.group(1)) if match else -1

    # vtu_files = sorted(results_dir.glob("*.vtu"), key=extract_number)
    # if not vtu_files:
    #     log.error("❌ No .vtu files found in results directory.")
    #     return

    # vtu_file = vtu_files[-1]
    # log.info(f"Using VTU file: {vtu_file.name}")

    # ─── Locate VTU file by target time ───
    def extract_number(fpath):
        match = re.search(r"_(\d+)\.vtu$", fpath.name, re.IGNORECASE)
        return int(match.group(1)) if match else -1

    vtu_files = sorted(results_dir.glob("*.vtu"), key=extract_number)
    if not vtu_files:
        log.error("❌ No .vtu files found in results directory.")
        return

    # --- Load config info ---
    target_time = datetime.fromisoformat(target_time)
    start_time = datetime.fromisoformat(config["run_period"]["start"])
    dt_hrs = int(config["run_period"]["output_dt_hours"])

    # --- Compute file index for target time ---
    elapsed_seconds = (target_time - start_time).total_seconds() / 3600
    if elapsed_seconds < 0:
        log.error(f"❌ Target time {target_time} is before model start time {start_time}")
        return

    index = int(round(elapsed_seconds / dt_hrs)) + 1  # SWAN files start at 1

    # --- Find closest available VTU file ---
    available_indices = [extract_number(f) for f in vtu_files]
    closest_idx = min(available_indices, key=lambda x: abs(x - index))
    vtu_file = next(f for f in vtu_files if extract_number(f) == closest_idx)

    # --- Logging ---
    if closest_idx != index:
        log.warning(f"⚠️ Exact match not found for {target_time}, using closest index {closest_idx}")

    log.info(f"✅ Using VTU file: {vtu_file.name} for time {target_time}")

    # ─── Clean VTU file using PyVista ───
    mesh_pv = pv.read(vtu_file)
    for comp in ["Windv", "Windu"]:
        if comp in mesh_pv.point_data:
            arr = mesh_pv.point_data[comp]
            if arr.ndim == 1 and arr.size == mesh_pv.n_points * 3:
                mesh_pv.point_data[comp] = arr.reshape((mesh_pv.n_points, 3))
                log.info(f"Reshaped {comp} to {mesh_pv.point_data[comp].shape}")

    # Remove invalid point arrays
    bad_keys = [k for k, v in mesh_pv.point_data.items() if v.shape[0] != mesh_pv.n_points]
    for k in bad_keys:
        del mesh_pv.point_data[k]
        log.warning(f"Removed invalid point_data: {k}")

    clean_vtu_file = output_dir / f"{vtu_file.stem}_clean.vtu"
    mesh_pv.save(clean_vtu_file)

    # ─── Load mesh for plotting ───
    mesh = meshio.read(clean_vtu_file, file_format="vtu")
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    cells = mesh.cells_dict.get("triangle", None)
    if cells is None:
        raise ValueError("No triangle cells found in VTU mesh.")
    triang = tri.Triangulation(x, y, cells)

    Hsig = mesh.point_data.get("Hsig")
    Tm01 = mesh.point_data.get("TPsmoo")
    Dir = mesh.point_data.get("PkDir")

    # ─── Domain bounds ───
    # xmin, xmax = 362500, 386600
    # ymin, ymax = 6427400, 6464600
    xmin, xmax = 348285,390000
    ymin, ymax = 6409355, 6499491
    velocity_res = 200
    mask_res = 50

    # ─── Load and clip land shapefile ───
    shapefile = input_dir / "Coastline_LGATE_070.shp"
    land = gpd.read_file(shapefile).to_crs("EPSG:7850")

    buffer = 100  # add small buffer to ensure full coastline captured
    bbox = box(xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)
    land_clipped = gpd.clip(land, bbox)

    # ─── Make land mask ───
    x_mask = np.arange(xmin, xmax, mask_res)
    y_mask = np.arange(ymax, ymin, -mask_res)
    highres_shape = (len(y_mask), len(x_mask))
    highres_transform = from_origin(xmin, ymax, mask_res, mask_res)
    highres_mask = rasterize(
        [(geom, 1) for geom in land_clipped.geometry],
        out_shape=highres_shape,
        transform=highres_transform,
        fill=0,
        dtype='uint8'
    )
    factor = velocity_res // mask_res

    # ─── Grid for interpolation ───
    x_grid = np.arange(xmin, xmax, velocity_res)
    y_grid = np.arange(ymax, ymin, -velocity_res)
    xg, yg = np.meshgrid(x_grid, y_grid)
    land_mask = block_reduce(highres_mask, block_size=(factor, factor), func=np.max)
    land_mask = land_mask[:len(y_grid), :len(x_grid)]

    # ─── Wave direction vectors ───
    dir_interp = griddata((x, y), Dir, (xg, yg), method="linear", fill_value=np.nan)
    theta_to = (270 - dir_interp) % 360
    rad = np.radians(theta_to)
    udir, vdir = np.cos(rad), np.sin(rad)
    udir[land_mask == 1] = np.nan
    vdir[land_mask == 1] = np.nan

    # ─── Plot setup ───
    shp_path = input_dir / "18k_channel_bounds.shp"
    gdf = gpd.read_file(shp_path)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    titles = [
        "Significant Wave Height (m)",
        "Peak Wave Period (s)",
        "Peak Wave Direction (°)",
        "Wind Speed & Vectors"
    ]
    cmaps = ["Blues", "Reds", "jet", "cool"]

    # Hsig
    levels = np.linspace(0, np.nanmax(Hsig), 10)
    cf = axs[0, 0].tricontourf(triang, Hsig, levels=levels, cmap=cmaps[0])
    plt.colorbar(cf, ax=axs[0, 0])
    axs[0, 0].set(title=titles[0], aspect="equal", xlim=(xmin, xmax), ylim=(ymin, ymax))
    land_clipped.plot(ax=axs[0, 0], color="lightgray", edgecolor="black", zorder=2)
    gdf.plot(ax=axs[0, 0],color='lightgray', linewidth=0.5, linestyle='--')

    # Tm01
    levels = np.linspace(0, np.nanmax(Tm01), 10)
    cf = axs[0, 1].tricontourf(triang, Tm01, levels=levels, cmap=cmaps[1])
    plt.colorbar(cf, ax=axs[0, 1])
    axs[0, 1].set(title=titles[1], aspect="equal", xlim=(xmin, xmax), ylim=(ymin, ymax))
    land_clipped.plot(ax=axs[0, 1], color="lightgray", edgecolor="black", zorder=2)
    gdf.plot(ax=axs[0, 1],color='lightgray', linewidth=0.5, linestyle='--')

    # Wave direction
    cf = axs[1, 0].pcolormesh(xg, yg, dir_interp, cmap=cmaps[2], shading="auto", vmin=0, vmax=360)
    axs[1, 0].quiver(xg[::10, ::10], yg[::10, ::10], udir[::10, ::10], vdir[::10, ::10], scale=40, color="black")
    plt.colorbar(cf, ax=axs[1, 0])
    axs[1, 0].set(title=titles[2], aspect="equal", xlim=(xmin, xmax), ylim=(ymin, ymax))
    land_clipped.plot(ax=axs[1, 0], color="lightgray", edgecolor="black", zorder=2)
    gdf.plot(ax=axs[1, 0],color='lightgray', linewidth=0.5, linestyle='--')

    # ─── Wind field ───
    if "Windv" in mesh.point_data:
        windv = mesh.point_data["Windv"]
        u_wind = windv[:, 0]
        v_wind = windv[:, 1]
        # print(windv[:, 0], windv[:, 1], windv[:, 2])
        # Interpolate
        u_interp = griddata((x, y), u_wind, (xg, yg), method="linear", fill_value=np.nan)
        v_interp = griddata((x, y), v_wind, (xg, yg), method="linear", fill_value=np.nan)

        # 
        u_interp_to, v_interp_to = u_interp, v_interp
        wind_speed = np.hypot(u_interp_to, v_interp_to)

        u_interp_to[land_mask == 1] = np.nan
        v_interp_to[land_mask == 1] = np.nan
        wind_speed[land_mask == 1] = np.nan

        cf = axs[1, 1].pcolormesh(xg, yg, wind_speed, shading="auto",
                                  cmap=cmaps[3], vmin=0, vmax=np.nanmax(wind_speed))
        axs[1, 1].quiver(xg[::10, ::10], yg[::10, ::10],
                         u_interp_to[::10, ::10], v_interp_to[::10, ::10],
                         scale=80, color="black")
        axs[1, 1].set(title=titles[3], aspect="equal", xlim=(xmin, xmax), ylim=(ymin, ymax))
        land_clipped.plot(ax=axs[1, 1], color="lightgray", edgecolor="black", zorder=2)
        gdf.plot(ax=axs[1, 1],color='lightgray', linewidth=0.5, linestyle='--')
        plt.colorbar(cf, ax=axs[1, 1])
    else:
        log.warning("⚠️ 'Windv' not found in VTU file, skipping wind plot.")

    fig.suptitle(f"SWAN output at {target_time} UTC", fontsize=14)
    # ─── Save figure ───
    timestep_match = re.search(r"_(\d+)\.vtu$", vtu_file.name)
    timestep_str = timestep_match.group(1) if timestep_match else "latest"
    fig_name = f"{case_name}_wave_fields_{timestep_str}_{target_time.strftime("%Y%m%d%H%M%S")}_UTC.png"
    plt.savefig(post_dir / fig_name, dpi=300)
    plt.close(fig)

    log.info(f"✅ Post-processing completed successfully. Figure saved as {fig_name}")


def validate_swan_vs_obs(config, log, stationnm):
    """
    Validate SWAN output against observations for Hs, Tp, and Dir.
    Produces one figure with 3 subplots and displays skill metrics.

    Parameters
    ----------
    swan_file : str or Path
        Path to SWAN .tab output file.
    obs_file : str or Path
        Path to observed Excel file.
    log : logging.Logger
        Logger instance.
    output_dir : str or Path, optional
        Directory to save output figure.
    # Bias: Average difference between model (m) and observation (o).
    # RMSE
    # MAE: The mean of absolute differences. 
    # SCATTER INDEX: RMSE normalized by the mean observed value.
    # Correlation Coefficient (Corr)
    # Willmott’s Index of Agreement (d): Developed by Willmott (1981) to assess predictive skill.Ideal value: 1 (perfect agreement).
    # Relative Bias (%): Definition: Mean bias expressed as a percentage of mean observation.
    """

    swan_file = Path(config["paths"]["case_dir"], f"{config["project_name"]}.tab")
    obs_file = Path(config["paths"]["obsdata_dir"],stationnm)
    output_dir = Path(config["paths"]["case_post_dir"])

    # ─── Skill Metrics ───

    def bias(m, o):  return np.mean(m - o)
    def rmse(m, o):  return np.sqrt(np.mean((m - o)**2))
    def mae(m, o):   return np.mean(np.abs(m - o))
    def si(m, o):    return rmse(m, o) / np.mean(o)
    def corr(m, o):  return np.corrcoef(m, o)[0,1]
    def willmott(m, o):
        return 1 - np.sum((m - o)**2) / np.sum((np.abs(m - np.mean(o)) + np.abs(o - np.mean(o)))**2)
    def rel_bias(m, o): 
        return 100 * (np.mean(m) - np.mean(o)) / np.mean(o)

    # ─── Read SWAN Output ───
    log.info(f"Reading SWAN output from {swan_file}")
    try:

        # Read variable names from line 5
        with open(swan_file, 'r') as f:
            for _ in range(4):
                f.readline()  # skip first 4 header lines
            var_line = f.readline().strip()
        swan_vars = [v.strip('%').strip() for v in var_line.split() if v.strip('%').strip()]

        # Read numeric data (skip headers + units)
        swan = pd.read_csv(
            swan_file,
            delim_whitespace=True,
            skiprows=6,
            names=swan_vars,
            comment='%',
            engine='python'
        )

        # Extract the first location (every 12th line starting from 0)
        nloc = 12
        swan = swan.iloc[::nloc, :].reset_index(drop=True)

        # Create time column based on config info
        start = datetime.fromisoformat(config["run_period"]["start"])
        end = datetime.fromisoformat(config["run_period"]["end"])
        outdt_hr = config["run_period"]["output_dt_hours"]

        # Construct time sequence
        time_index = pd.date_range(start=start, end=end, freq=f"{outdt_hr}H")
        if len(time_index) != len(swan):
            log.warning(f"Time length mismatch: expected {len(time_index)} steps, found {len(swan)} lines")
            time_index = time_index[:len(swan)]

        swan["time"] = time_index

        # Keep only relevant columns
        expected_cols = ['Hsig', 'TPsmoo', 'Dir']
        available = [c for c in expected_cols if c in swan.columns]
        if len(available) < 3:
            raise ValueError(f"Missing expected SWAN variables, found only: {swan.columns.tolist()}")

        swan = swan[['time'] + available]

        swan.columns = ["time", "Hs_model", "Tp_model", "Dir_model"]

        log.info(f"SWAN output read successfully for {len(swan)} timesteps, using location 1 of {nloc}")

    except Exception as e:
        log.error(f"Failed to read SWAN output: {e}")
        return



    # ─── Read Observations ───
    log.info(f"Reading observation files")
    try:
        obs_dir = Path(obs_file)  # folder containing RDWYYYY_Z.xlsx
        start = datetime.fromisoformat(config["run_period"]["start"])
        end = datetime.fromisoformat(config["run_period"]["end"])
        model_years = list(range(start.year, end.year + 1))

        # Select files matching model years
        obs_files = []
        for year in model_years:
            matches = list(obs_dir.glob(f"*{year}_*.xlsx"))
            obs_files.extend(matches)

        if not obs_files:
            log.warning(f"No observation files found for years {model_years}")
            return

        all_obs = []
        for f in obs_files:
            try:
                tmp = pd.read_excel(f, skiprows=4)
                tmp = tmp.rename(columns={tmp.columns[0]: "time"})
                tmp["time"] = pd.to_datetime(tmp["time"], errors="coerce")
                tmp = tmp.dropna(subset=["time"])
                all_obs.append(tmp)
                log.debug(f"Loaded {len(tmp)} rows from {f.name}")
            except Exception as e:
                log.warning(f"Skipping {f.name}: {e}")

        if not all_obs:
            raise FileNotFoundError("No valid observation data for selected years.")

        obs = pd.concat(all_obs, ignore_index=True)
        obs = obs.dropna(subset=["time"])

        # Detect wave variable columns
        hs_cols = [c for c in obs.columns if "hs" in c.lower()]
        tp_cols = [c for c in obs.columns if "tp" in c.lower()]
        dir_cols = [c for c in obs.columns if "dir" in c.lower()]
        if not all([hs_cols[0], tp_cols[0], dir_cols[0]]):
            raise ValueError(f"Missing expected wave columns. Found: {obs.columns.tolist()}")

        obs = obs[["time", hs_cols[0], tp_cols[0], dir_cols[0]]]
        obs.columns = ["time", "Hs_obs", "Tp_obs", "Dir_obs"]
        obs["time"] = (pd.to_datetime(obs["time"]).dt.tz_localize("Australia/Perth").dt.tz_convert("UTC").dt.tz_localize(None))

        # Filter to model run period
        obs = obs[(obs["time"] >= start) & (obs["time"] <= end)]
        if obs.empty:
            log.warning("No observation data found within model period.")
        else:
            log.info(f"Filtered observation data to {len(obs)} points between {start} and {end}: note that hs/tp is for total, dir is for swell")

    except Exception as e:
        log.error(f"Failed to read observation data: {e}")
        return


    # ─── Merge by Time ───
    log.info("Merging model and observation datasets by time")
    df = pd.merge_asof(
        swan.sort_values('time'),
        obs.sort_values('time'),
        on='time',
        tolerance=pd.Timedelta('30min')
    ).dropna()

    if df.empty:
        log.warning("No overlapping time found between SWAN and observation data.")
        return

    # ─── Plot 3 Subplots ───
    variables = [
        ('Hs', 'Significant Wave Height (m)'),
        ('Tp', 'Peak Period (s)'),
        ('Dir', 'Mean Wave Direction (°)')
    ]

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    for ax, (var, label) in zip(axs, variables):
        m = df[f'{var}_model'].values
        o = df[f'{var}_obs'].values

        ax.plot(df['time'], o, 'k-', label='Observed', linewidth=1.2)
        ax.plot(df['time'], m, 'r--', label='Model', linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

        # ─── Compute Metrics ───
        metrics = {
            'Bias': bias(m, o),
            'RMSE': rmse(m, o),
            'MAE': mae(m, o),
            'SI': si(m, o),
            'Corr': corr(m, o),
            'Willmott': willmott(m, o),
            'RelBias(%)': rel_bias(m, o)
        }

        text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        ax.text(
            0.01, 0.98, text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey')
        )

        log.info(f"Skill metrics for {var}: " + ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()]))

    axs[0].legend(loc='upper right')
    axs[-1].set_xlabel("Time")
    fig.suptitle(f"SWAN Validation vs Observations ({stationnm})", fontsize=14, y=1.02)

    # ─── Save or Show ───
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_name = Path(output_dir) / f"SWAN_Validation_{stationnm}.png"
        plt.savefig(out_name, dpi=300, bbox_inches='tight')
        log.info(f"✅ Validation figure saved: {out_name}")
    else:
        log.info("Displaying figure without saving")

    log.info("✅ Validation process completed successfully")

    return df



def compare_swan_bom_wind(config, log, bomstation):
    """
    Compare SWAN model wind output (X-Windv, Y-Windv) with BoM wind observations (NetCDF).
    Subsets BoM to SWAN time range and plots comparison.
    """

    # ─── File Paths ───
    swan_file = Path(config["paths"]["case_dir"]) / f"{config['project_name']}.tab"
    bom_file  = Path(config["paths"]["obsdata_dir"]) / "BOM" / f"{bomstation}.nc"
    output_dir = Path(config["paths"]["case_post_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Read SWAN Wind Output ───
    log.info("🔹 Reading SWAN wind output")
    try:
        with open(swan_file, 'r') as f:
            for _ in range(4):
                f.readline()
            var_line = f.readline().strip()
        swan_vars = [v.strip('%').strip() for v in var_line.split() if v.strip('%').strip()]

        swan = pd.read_csv(
            swan_file,
            delim_whitespace=True,
            skiprows=6,
            names=swan_vars,
            comment='%',
            engine='python'
        )

        nloc = 12
        swan = swan.iloc[::nloc, :].reset_index(drop=True)

        start = datetime.fromisoformat(config["run_period"]["start"])
        end   = datetime.fromisoformat(config["run_period"]["end"])
        outdt_hr = config["run_period"]["output_dt_hours"]
        time_index = pd.date_range(start=start, end=end, freq=f"{outdt_hr}H")
        swan["time"] = time_index[:len(swan)]

        # ─── SWAN U/V → speed & "from north" direction ───
        if 'X-Windv' in swan.columns and 'Y-Windv' in swan.columns:
            U = swan['X-Windv'].to_numpy()  # from east
            V = swan['Y-Windv'].to_numpy()  # from north
            spd = np.sqrt(U**2 + V**2)
            dir_from = (np.degrees(np.arctan2(-U, -V))) % 360  # 0° = from North, clockwise
        else:
            raise ValueError("SWAN output missing X-Windv and Y-Windv columns")

        swan = pd.DataFrame({
            "time": swan["time"],
            "spd_model": spd,
            "dir_model": dir_from
        })
        log.info(f"SWAN wind data read successfully for {len(swan)} timesteps")

    except Exception as e:
        log.error(f"❌ Failed to read SWAN wind output: {e}")
        return

    # ─── Read BoM Wind Data ───
    log.info(f"🔹 Reading BoM wind file: {bom_file}")
    try:
        ds = xr.open_dataset(bom_file)
        ds = ds.sortby("time")
        bom = pd.DataFrame({
            "time": pd.to_datetime(ds["time"].values),
            "spd_obs": ds["speed"].values/3.6,
            "dir_obs": ds["dir"].values
        })
        ds.close()
        log.info(f"BoM wind data loaded with {len(bom)} records")

        # Subset BoM to SWAN time range
        bom["time"] = (pd.to_datetime(bom["time"]).dt.tz_localize("Etc/GMT-8").dt.tz_convert("UTC").dt.tz_localize(None))
        
        bom = bom[(bom["time"] >= swan["time"].min()) & (bom["time"] <= swan["time"].max())]
        log.info(f"BoM data subsetted to SWAN time range: {len(bom)} records")

    except Exception as e:
        log.error(f"❌ Failed to read BoM NetCDF: {e}")
        return

    # ─── Merge by Time (closest match within tolerance) ───
    df = pd.merge_asof(
        swan.sort_values("time"),
        bom.sort_values("time"),
        on="time",
        tolerance=pd.Timedelta("30min")
    ).dropna()

    if df.empty:
        log.warning("⚠️ No overlapping time found between SWAN and BoM data.")
        return

    # ─── Compute Wind Speed Metrics ───
    def bias(m, o):  return np.mean(m - o)
    def rmse(m, o):  return np.sqrt(np.mean((m - o)**2))
    def corr(m, o):  return np.corrcoef(m, o)[0,1]

    ws_metrics = {
        "Bias": bias(df["spd_model"], df["spd_obs"]),
        "RMSE": rmse(df["spd_model"], df["spd_obs"]),
        "Corr": corr(df["spd_model"], df["spd_obs"])
    }

    # ─── Plot Comparison ───
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    # Wind speed
    axs[0].plot(df["time"], df["spd_obs"], "k-", label="BoM", linewidth=1.2)
    axs[0].plot(df["time"], df["spd_model"], "r--", label="SWAN", linewidth=1.2)
    axs[0].set_ylabel("Wind Speed (m/s)")
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    axs[0].text(
        0.01, 0.95,
        "\n".join([f"{k}: {v:.2f}" for k,v in ws_metrics.items()]),
        transform=axs[0].transAxes,
        fontsize=9,
        va="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey")
    )

    # Wind direction
    axs[1].plot(df["time"], df["dir_obs"], "k-", label="BoM", linewidth=1.2)
    axs[1].plot(df["time"], df["dir_model"], "r--", label="SWAN", linewidth=1.2)
    axs[1].set_ylabel("Wind Direction (° from North)")
    axs[1].set_xlabel("Time")
    axs[1].grid(alpha=0.3)

    fig.suptitle(f"SWAN vs BoM Wind Comparison ({bomstation})", fontsize=14, y=1.02)

    # ─── Save Plot ───
    out_path = output_dir / f"SWAN_vs_BoM_Wind_{bomstation}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    log.info(f"✅ Comparison plot saved: {out_path}")

    return df

def validate_swan_spectrum_1d(config, log, target_time, stationnm):
    """
    Plot and save 1D SWAN spectrum for the first location at a given time step.

    Parameters
    ----------
    config : dict
        Contains:
          paths.case_dir       : directory with SWAN outputs
          paths.case_post_dir  : output directory for figures
          project_name         : SWAN project name (used in filenames)
    log : logging.Logger
        Logger instance.
    target_time : datetime
        Target timestamp to extract (UTC or model time).
    stationnm : str
        Station name (used in output filename).
    """

    swan_spc_file = Path(config["paths"]["case_dir"], f"{config['project_name']}.spec")
    output_dir = Path(config["paths"]["case_post_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading SWAN spectrum file: {swan_spc_file}")

    if stationnm.upper() == "ROTTNEST":
        locnum = 0

    try:
        with open(swan_spc_file, "r") as f:
            # ---- Skip general header ----
            for _ in range(20):
                f.readline()

            # ---- Frequencies ----
            nfreq = int(f.readline().strip().split(' ')[0])
            freqs = np.array([float(f.readline().strip()) for _ in range(nfreq)])

            # ---- Directions ----
            f.readline()
            ndir = int(f.readline().strip().split(' ')[0])
            dirs = np.array([float(f.readline().strip()) for _ in range(ndir)])

            # ---- Skip QUANT, VaDens, unit, exception value ----
            for _ in range(5):
                f.readline()

            found = False
            spectrum_1d = None

            # ----------------------------------------------------------------
            # Loop through each time step block
            # ----------------------------------------------------------------
            while True:
                line = f.readline()
                if not line:
                    break  # EOF

                line = line.strip().split(' ')[0]
                if len(line) >= 15 and line[8] == ".":  # time line
                    try:
                        step_time = datetime.strptime(line, "%Y%m%d.%H%M%S")
                    except Exception:
                        continue

                    # If this time matches the target_time, process
                    if step_time == datetime.strptime(target_time, "%Y%m%d.%H%M%S"):
                        log.info(f"✅ Found matching time step: {step_time}")

                        # Loop through 12 locations for this timestamp
                        for loc in range(12):
                            line = f.readline().strip()
                            if line.upper() != "FACTOR":
                                log.warning(f"Expected 'FACTOR' at location {loc+1}, got: {line}")
                                continue
                            scale_factor = float(f.readline().strip())

                            # Read spectral matrix (nfreq x ndir)
                            data = []
                            for _ in range(nfreq):
                                row = f.readline()
                                if not row:
                                    raise ValueError("Unexpected EOF while reading spectrum matrix")
                                row_vals = [float(x) for x in row.split()]
                                if len(row_vals) != ndir:
                                    raise ValueError(f"Expected {ndir} directions, got {len(row_vals)}")
                                data.append(row_vals)

                            spec2d = np.array(data) * scale_factor

                            # If this is the first location → save and stop
                            if loc == locnum:
                                dtheta = np.deg2rad(np.abs(np.diff(dirs).mean()))
                                spectrum_1d = np.sum(spec2d, axis=1) * dtheta
                                found = True
                                break
                        break  # stop after found target time

            if not found:
                log.warning(f"⚠ Target time {target_time} not found in file.")
                return

            # ----------------------------------------------------------------
            # Plot 1D spectrum
            # ----------------------------------------------------------------
            plt.figure(figsize=(7, 5))
            plt.plot(freqs, spectrum_1d, "-o", color="blue", markersize=4)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Spectral density (m²/Hz)")
            plt.title(f"SWAN 1D Spectrum\n{stationnm} @ {target_time}")
            plt.grid(True)

            out_file = output_dir / f"{config['project_name']}_{stationnm}_{target_time}_spec1D.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close()

            log.info(f"✅ 1D spectrum saved: {out_file}")

    except Exception as e:
        log.error(f"❌ Failed to read or plot SWAN spectrum: {e}", exc_info=True)

def validate_swan_spectrum_2d(config, log, target_time, stationnm):
    """
    Plot and save 2D SWAN spectrum (frequency vs direction) for a specific location and time.

    Parameters
    ----------
    config : dict
        Contains:
          paths.case_dir       : directory with SWAN outputs
          paths.case_post_dir  : output directory for figures
          project_name         : SWAN project name (used in filenames)
    log : logging.Logger
        Logger instance.
    target_time : str
        Target timestamp in format '%Y%m%d.%H%M%S'
    stationnm : str
        Station name (used in output filename).
    """

    swan_spc_file = Path(config["paths"]["case_dir"], f"{config['project_name']}.spec")
    output_dir = Path(config["paths"]["case_post_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading SWAN spectrum file: {swan_spc_file}")

    # Map station name → location index
    if stationnm.upper() == "ROTTNEST":
        locnum = 0
    else:
        locnum = 0  # default to first

    try:
        with open(swan_spc_file, "r") as f:
            # ---- Skip general header ----
            for _ in range(20):
                f.readline()

            # ---- Frequencies ----
            nfreq = int(f.readline().strip().split()[0])
            freqs = np.array([float(f.readline().strip()) for _ in range(nfreq)])

            # ---- Directions ----
            f.readline()
            ndir = int(f.readline().strip().split()[0])
            dirs = np.array([float(f.readline().strip()) for _ in range(ndir)])

            # ---- Skip QUANT, VaDens, unit, exception value ----
            for _ in range(5):
                f.readline()

            found = False
            spec2d = None

            # ----------------------------------------------------------------
            # Loop through each time step block
            # ----------------------------------------------------------------
            while True:
                line = f.readline()
                if not line:
                    break  # EOF

                line = line.strip().split()[0]
                if len(line) >= 15 and line[8] == ".":  # time line
                    try:
                        step_time = datetime.strptime(line, "%Y%m%d.%H%M%S")
                    except Exception:
                        continue

                    if step_time == datetime.strptime(target_time, "%Y%m%d.%H%M%S"):
                        log.info(f"✅ Found matching time step: {step_time}")

                        # Loop through locations (assume 12)
                        for loc in range(12):
                            line = f.readline().strip()
                            if line.upper() != "FACTOR":
                                log.warning(f"Expected 'FACTOR' at location {loc+1}, got: {line}")
                                continue
                            scale_factor = float(f.readline().strip())

                            # Read spectral matrix
                            data = []
                            for _ in range(nfreq):
                                row = f.readline()
                                if not row:
                                    raise ValueError("Unexpected EOF while reading spectrum matrix")
                                row_vals = [float(x) for x in row.split()]
                                if len(row_vals) != ndir:
                                    raise ValueError(f"Expected {ndir} directions, got {len(row_vals)}")
                                data.append(row_vals)

                            if loc == locnum:
                                spec2d = np.array(data) * scale_factor
                                found = True
                                break
                        break  # stop after found time

            if not found or spec2d is None:
                log.warning(f"⚠ Target time {target_time} not found in file.")
                return

            # ----------------------------------------------------------------
            # Plot 2D spectrum
            # ----------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(8, 6))

            # Convert to degrees 0–360 (ensure increasing order for pcolormesh)
            dirs_plot = (dirs + 360) % 360
            if np.any(np.diff(dirs_plot) < 0):
                sort_idx = np.argsort(dirs_plot)
                dirs_plot = dirs_plot[sort_idx]
                spec2d = spec2d[:, sort_idx]

            pcm = ax.pcolormesh(dirs_plot, freqs, spec2d, shading="auto", cmap="turbo")
            cbar = plt.colorbar(pcm, ax=ax, label="Energy density (m²/Hz/°)")

            ax.set_xlabel("Direction (°)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"SWAN 2D Spectrum\n{stationnm} @ {target_time}")

            plt.tight_layout()

            out_file = output_dir / f"{config['project_name']}_{stationnm}_{target_time}_spec2D.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            log.info(f"✅ 2D spectrum saved: {out_file}")

    except Exception as e:
        log.error(f"❌ Failed to read or plot SWAN 2D spectrum: {e}", exc_info=True)

def validate_swan_spectrum_1d_obs(config, log, target_time, stationnm):
    """
    Plot and compare SWAN 1D spectrum with observed spectrum nearest to target_time.

    Parameters
    ----------
    config : dict
        Contains SWAN model paths and settings.
    log : logging.Logger
        Logger instance.
    target_time : datetime
        Target timestamp to extract (UTC or model time).
    stationnm : str
        Station name.
    obs_dir : str or Path
        Directory containing observed spectra files named like '2021-11-01T02h43'.
    """

    # ----------------------------------------------------------------------
    # Read SWAN model spectrum (same as your working function)
    # ----------------------------------------------------------------------
    target_time = datetime.strptime(target_time, "%Y%m%d.%H%M%S")
    swan_spc_file = Path(config["paths"]["case_dir"], f"{config['project_name']}.spec")
    output_dir = Path(config["paths"]["case_post_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dir = Path(config["paths"]["obsdata_dir"],stationnm,target_time.strftime("%Y%m"))

    log.info(f"Reading SWAN spectrum file: {swan_spc_file}")

    if stationnm.upper() == "ROTTNEST":
        locnum = 0

    try:
        with open(swan_spc_file, "r") as f:
            for _ in range(20):
                f.readline()

            nfreq = int(f.readline().strip().split()[0])
            freqs = np.array([float(f.readline().strip()) for _ in range(nfreq)])

            f.readline()
            ndir = int(f.readline().strip().split()[0])
            dirs = np.array([float(f.readline().strip()) for _ in range(ndir)])

            for _ in range(5):
                f.readline()

            found = False
            spectrum_1d = None

            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip().split()[0]
                if len(line) >= 15 and line[8] == ".":
                    try:
                        step_time = datetime.strptime(line, "%Y%m%d.%H%M%S")
                    except Exception:
                        continue

                    if step_time == target_time:
                        log.info(f"✅ Found matching SWAN time step: {step_time}")
                        for loc in range(12):
                            line = f.readline().strip()
                            if line.upper() != "FACTOR":
                                continue
                            scale_factor = float(f.readline().strip())

                            data = []
                            for _ in range(nfreq):
                                row_vals = [float(x) for x in f.readline().split()]
                                data.append(row_vals)
                            spec2d = np.array(data) * scale_factor
                            if loc == locnum:
                                dtheta = np.deg2rad(np.abs(np.diff(dirs).mean()))
                                spectrum_1d = np.sum(spec2d, axis=1) * dtheta
                                m0 = np.trapz(spectrum_1d, freqs)                        # integrate over frequency
                                Hs = 4 * np.sqrt(m0)
                                log.info(f"SWAN_Hs from spectrum: {Hs}")
                                found = True
                                break
                        break

            if not found:
                log.warning(f"⚠ Target time {target_time} not found in SWAN file.")
                return

    except Exception as e:
        log.error(f"❌ Failed to read SWAN spectrum: {e}", exc_info=True)
        return

    # ----------------------------------------------------------------------
    # Locate and read observed spectrum file nearest to target_time
    # ----------------------------------------------------------------------
    obs_files = sorted(obs_dir.glob("*-*-*T*h*.dat"))  # flexible pattern
    if not obs_files:
        log.warning(f"No observed spectrum files found in {obs_dir}")
        return

    # Parse filenames to datetimes
    obs_times = []
    for fpath in obs_files:
        try:
            t = datetime.strptime(fpath.stem, "%Y-%m-%dT%Hh%M")
            obs_times.append((t, fpath))
        except:
            continue

    if not obs_times:
        log.warning("No valid observed filenames found.")
        return

    # Find closest in time
    obs_times = [
        (pd.Timestamp(t).tz_localize("Etc/GMT-8").tz_convert("UTC").tz_localize(None), fpath)
        for t, fpath in obs_times
    ]

    diffs = [abs((t - target_time).total_seconds()) for t, _ in obs_times]
    nearest_idx = int(np.argmin(diffs))
    obs_time, obs_file = obs_times[nearest_idx]
    log.info(f"📘 Closest observed file: {obs_file.name} (Δt = {diffs[nearest_idx]/60:.1f} min)")

    # ----------------------------------------------------------------------
    # Read observed file
    # ----------------------------------------------------------------------
    with open(obs_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    header = lines[0].split()
    nfreq, ndir = int(header[1]), int(header[2])

    freqs_obs = np.array([float(x) for x in " ".join(lines[2:2 + ((nfreq-1)//10 + 1)]).split()])
    start_idx = 2 + ((nfreq-1)//10 + 1)
    dirs = np.array([float(x) for x in " ".join(lines[start_idx:start_idx + ((ndir-1)//10 + 1)]).split()])

    matrix = np.zeros((nfreq, ndir))
    energy_start = start_idx + ((ndir - 1)//10 + 1)
    line_idx = energy_start

    for i in range(nfreq):
        energy_vals = []
        # 9 full lines × 10 values
        for _ in range(9):
            energy_vals.extend([float(v) for v in lines[line_idx].split()])
            line_idx += 1
        # 1 line with the last single value
        energy_vals.extend([float(v) for v in lines[line_idx].split()])
        line_idx += 1

        matrix[i, :] = energy_vals[:ndir]

    matrix[matrix <= -4.0] = np.nan
    energy = 10 ** matrix



    # Convert to 1D spectrum
    dtheta = np.deg2rad(np.abs(np.diff(dirs).mean()))
    obs_spec1d = np.nansum(energy, axis=1) * dtheta
    m0 = np.trapz(obs_spec1d, freqs_obs)                        # integrate over frequency
    Hs_obs = 4 * np.sqrt(m0)
    log.info(f"obs_hs from spectrum: {Hs_obs}")
    # ----------------------------------------------------------------------
    # Normalize spectra to peak = 1
    # ----------------------------------------------------------------------
    if spectrum_1d is not None:
        spectrum_1d_norm = spectrum_1d / np.nanmax(spectrum_1d)
    else:
        spectrum_1d_norm = None

    if obs_spec1d is not None:
        obs_spec1d_norm = obs_spec1d / np.nanmax(obs_spec1d)
    else:
        obs_spec1d_norm = None

    # # ----------------------------------------------------------------------
    # # Plot comparison
    # # ----------------------------------------------------------------------
    # plt.figure(figsize=(7, 5))
    # plt.plot(freqs, spectrum_1d, "-o", color="blue", markersize=4, label="SWAN model")
    # plt.plot(freqs_obs, obs_spec1d, "-s", color="red", markersize=3, label=f"Obs ({obs_time:%Y-%m-%d %H:%M})")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Spectral density (m²/Hz)")
    # plt.title(f"1D Spectrum Comparison\n{stationnm} @ {target_time}")
    # plt.legend()
    # plt.grid(True)

    # out_file = output_dir / f"{config['project_name']}_{stationnm}_{target_time}_spec1D_compare.png"
    # plt.savefig(out_file, dpi=300, bbox_inches="tight")
    # plt.close()

    # log.info(f"✅ Comparison plot saved: {out_file}")

    # ----------------------------------------------------------------------
    # Plot comparison
    # ----------------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(freqs, spectrum_1d_norm, "-o", color="blue", markersize=4, label="SWAN model (normalized)")
    plt.plot(freqs_obs, obs_spec1d_norm, "-s", color="red", markersize=3, label=f"Obs (normalized)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized spectral density")
    plt.title(f"Normalized 1D Spectrum Comparison\n{stationnm} @ {target_time}")
    plt.legend()
    plt.grid(True)

    out_file = output_dir / f"{config['project_name']}_{stationnm}_{target_time}_UTC_spec1D_compare_normalized.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"✅ Comparison plot saved: {out_file}")

def validate_swan_spectrum_2d_obs(config, log, target_time, stationnm):
    """
    Plot 2D spectra of SWAN model and observed data in two subplots.

    Parameters
    ----------
    config : dict
        Contains SWAN paths/settings and obsdata_dir:
            - paths.case_dir
            - paths.case_post_dir
            - project_name
            - paths.obsdata_dir
    log : logging.Logger
        Logger instance
    target_time : str
        Target time as "YYYYMMDD.HHMMSS"
    stationnm : str
        Station name
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    target_dt = datetime.strptime(target_time, "%Y%m%d.%H%M%S")
    swan_spc_file = Path(config["paths"]["case_dir"], f"{config['project_name']}.spec")
    output_dir = Path(config["paths"]["case_post_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = Path(config["paths"]["obsdata_dir"], stationnm, target_dt.strftime("%Y%m"))

    # ------------------- Read SWAN spectrum -------------------
    log.info(f"Reading SWAN spectrum: {swan_spc_file}")
    spec2d_model = None
    freqs_model, dirs_model = None, None

    try:
        with open(swan_spc_file, "r") as f:
            for _ in range(20): f.readline()
            nfreq = int(f.readline().strip().split()[0])
            freqs_model = np.array([float(f.readline().strip()) for _ in range(nfreq)])
            f.readline()
            ndir = int(f.readline().strip().split()[0])
            dirs_model = np.array([float(f.readline().strip()) for _ in range(ndir)])
            dirs_model = (dirs_model + 360) % 360

            for _ in range(5): f.readline()

            locnum = 0 if stationnm.upper() == "ROTTNEST" else 0
            while True:
                line = f.readline()
                if not line: break
                line = line.strip().split()[0]
                if len(line) >= 15 and line[8] == ".":
                    try:
                        step_time = datetime.strptime(line, "%Y%m%d.%H%M%S")
                    except: continue
                    if step_time == target_dt:
                        for loc in range(12):
                            line = f.readline().strip()
                            if line.upper() != "FACTOR": continue
                            scale_factor = float(f.readline().strip())
                            data = []
                            for _ in range(nfreq):
                                row_vals = [float(x) for x in f.readline().split()]
                                data.append(row_vals)
                            spec2d_model = np.array(data) * scale_factor
                            if loc == locnum: break
                        break
            if np.any(np.diff(dirs_model) < 0):
                sort_idx = np.argsort(dirs_model)
                dirs_model = dirs_model[sort_idx]
                spec2d_model = spec2d_model[:, sort_idx]
    except Exception as e:
        log.error(f"Failed to read SWAN spectrum: {e}", exc_info=True)
        return

    # ------------------- Read observed spectrum -------------------
    obs_files = sorted(obs_dir.glob("*-*-*T*h*.dat"))
    if not obs_files:
        log.warning(f"No observed spectrum files found in {obs_dir}")
        return
    # closest in time
    obs_times = []
    for fpath in obs_files:
        try: obs_times.append((datetime.strptime(fpath.stem, "%Y-%m-%dT%Hh%M"), fpath))
        except: continue
    obs_times = [
        (pd.Timestamp(t).tz_localize("Etc/GMT-8").tz_convert("UTC").tz_localize(None), fpath)
        for t, fpath in obs_times
    ]
    diffs = [abs((t - target_dt).total_seconds()) for t, _ in obs_times]
    nearest_idx = int(np.argmin(diffs))
    obs_time, obs_file = obs_times[nearest_idx]
    log.info(f"Closest observed file: {obs_file.name} (Δt={diffs[nearest_idx]/60:.1f} min)")

    with open(obs_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    header = lines[0].split()
    nfreq_obs, ndir_obs = int(header[1]), int(header[2])
    freqs_obs = np.array([float(x) for x in " ".join(lines[2:2 + ((nfreq_obs-1)//10 + 1)]).split()])
    start_idx = 2 + ((nfreq_obs-1)//10 + 1)
    dirs_obs = np.array([float(x) for x in " ".join(lines[start_idx:start_idx + ((ndir_obs-1)//10 + 1)]).split()])

    matrix = np.zeros((nfreq_obs, ndir_obs))
    line_idx = start_idx + ((ndir_obs-1)//10 + 1)
    for i in range(nfreq_obs):
        energy_vals = []
        for _ in range(9):
            energy_vals.extend([float(v) for v in lines[line_idx].split()])
            line_idx += 1
        energy_vals.extend([float(v) for v in lines[line_idx].split()])
        line_idx += 1
        matrix[i, :] = energy_vals[:ndir_obs]
    matrix[matrix <= -4.0] = np.nan
    spec2d_obs = 10 ** matrix

    # ------------------- Plot 2D spectra -------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    X_model, Y_model = np.meshgrid(dirs_model, freqs_model)
    pcm1 = axs[0].pcolormesh(X_model, Y_model, spec2d_model, shading='auto', cmap='viridis')
    axs[0].set_xlabel("Direction (deg)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_title(f"SWAN 2D Spectrum\n{stationnm} @ {target_dt}")
    fig.colorbar(pcm1, ax=axs[0], label="Energy (m²/Hz/deg)")

    X_obs, Y_obs = np.meshgrid(dirs_obs, freqs_obs)
    pcm2 = axs[1].pcolormesh(X_obs, Y_obs, spec2d_obs, shading='auto', cmap='viridis')
    axs[1].set_xlabel("Direction (deg)")
    axs[1].set_title(f"Observed 2D Spectrum\n{stationnm} @ {obs_time}_UTC")
    fig.colorbar(pcm2, ax=axs[1], label="Energy (m²/Hz/deg)")

    out_file = output_dir / f"{config['project_name']}_{stationnm}_{target_time}_spec2D_compare.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"✅ 2D comparison plot saved: {out_file}")