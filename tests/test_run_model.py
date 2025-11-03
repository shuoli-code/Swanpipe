# tests/test_run_model.py
from swanmod import run_model
import yaml
from pathlib import Path

def test_prepare_case():
    print("🧩 Testing SWAN run_model module...")

    # Path to your YAML config file
    config_path = Path("/home/o2hpc/Project/SWANpipeline/config.yaml")

    # Read YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run SWAN preparation
    run_model.prepare_case(config)

    # Basic checks
    work_dir = Path(config["paths"]["work_dir"])
    swn_file = work_dir / "INPUT.swn"
    assert work_dir.exists(), "Run folder should be created"
    assert swn_file.exists(), ".swn file should be generated"

    print(f"✅ Test complete. SWAN run folder and .swn file created at {work_dir}")

# # Run test directly
# if __name__ == "__main__":
#     test_prepare_case()
