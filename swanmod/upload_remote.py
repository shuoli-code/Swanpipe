from pathlib import Path
import xarray as xr

def run(config):
    wind_path = Path(config["paths"]["wind_nc"])
    work_dir = Path(config["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(wind_path)
    print(f"Wind file loaded: {wind_path.name}, variables = {list(ds.data_vars)}")

    # Your existing wind interpolation logic here
    out_file = work_dir / "wind_for_swan.dat"
    print(f"Exported wind data to {out_file}")
