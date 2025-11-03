# swanmod/config_loader.py
import yaml
from pathlib import Path
from datetime import datetime

def load_config(config_path="/home/azureuser/project/SWANpipeline/config.yaml", save_updated=True):
    """
    Load and modify SWAN configuration file.

    - Automatically updates 'work_dir' using project name and run period
    - Creates directory if it doesn't exist
    - Optionally saves an updated YAML file (config_updated.yaml)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    # Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- Extract key info ---
    project = config.get("project_name", "SWANProject")
    run_period = config.get("run_period", {})
    start = datetime.fromisoformat(run_period["start"])
    end = datetime.fromisoformat(run_period["end"])

    # Format new name
    start_str = start.strftime("%Y%m%d%H%M")
    end_str = end.strftime("%Y%m%d%H%M")

    # --- Update work_dir ---
    base_work_dir = Path(config["paths"]["work_dir"])
    base_output_dir = Path(config["paths"]["out_dir"])
    base_post_dir = Path(config["paths"]["post_dir"])
    new_work_dir = base_work_dir / f"{project}_{start_str}_{end_str}"
    new_output_dir = base_output_dir / f"{project}_{start_str}_{end_str}"
    new_post_dir = base_post_dir / f"{project}_{start_str}_{end_str}"
    # update hotfile dir if use last run hot file
    if bool(config["run_period"]["hotstart"]):
        config["paths"]["hotstart_dir"] = config["paths"]["case_dir"]

    config["paths"]["case_dir"] = str(new_work_dir)
    config["paths"]["case_out_dir"] = str(new_output_dir)
    config["paths"]["case_post_dir"] = str(new_post_dir)
    # Create folder
    new_work_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Updated case_dir → {new_work_dir}")

    # --- Optionally save updated config ---
    if save_updated:
        updated_path = config_path.parent / "config.yaml"
        with open(updated_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        print(f"💾 Saved updated config → {updated_path}")

    return config


if __name__ == "__main__":
    # Quick check (for manual run)
    cfg = load_config("config.yaml")
    print("✅ Config loaded successfully:")
    print(yaml.dump(cfg, sort_keys=False))
