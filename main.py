import yaml
from pathlib import Path
from swanmod import (
    prepare_wind,
    prepare_boundary,
    run_model,
    postprocess_output,
    config_loader,
    utilities
)


def main():

    # ---- Step 0: update config file ----
    print("➡️ updating config file...")
    config = config_loader.load_config()

    # Dynamically generate log filename from config
    project_name = config["project_name"]
    start_time = config["run_period"]["start"].replace(":", "-").replace(" ", "_")
    end_time = config["run_period"]["end"].replace(":", "-").replace(" ", "_")
    log_filename = f"{project_name}_{start_time}_to_{end_time}.log"
    
    # ✅ Create logger ONCE here
    log = utilities.setup_logger(
        name=f"swan.{project_name}",
        log_file=log_filename,
        logdir="logs",
        level="INFO",
        console=True
    )

    # ---- Step 1: Wind Preparation ----
    if bool(config["run_period"]["use_wind"]) and bool(config["run_period"]["use_original_wind"]):
        print("➡️ Preparing wind input...")
        wind_file, windstr, windend = prepare_wind.Barrac2_slice(config, log)
    elif bool(config["run_period"]["use_wind"]) and bool(config["run_period"]["use_correct_wind_single"]):
        wind_file, windstr, windend = prepare_wind.correct_barrac2_with_bom_single(config, log)
    elif bool(config["run_period"]["use_wind"]) and bool(config["run_period"]["use_correct_wind_multi"]):
        wind_file, windstr, windend = prepare_wind.correct_barrac2_with_bom_multi(config, log)
    else:
        wind_file = Path('/rrrrrr/')
        windstr, windend = '20220101.000000', '20220101.000000'

    # ---- Step 2: Boundary Preparation ----
    if bool(config["run_period"]["use_bound"]):
        print("➡️ Preparing boundary conditions...")
        prepare_boundary.extract_boundary_data(config, log)

    # ---- Step 3: Generate SWAN input (.swn, etc.) ----
    print("➡️ Generating SWAN control files...")
    run_model.prepare_case(config, log, wind_file.name, windstr, windend)

    # ---- Step 4: Post-process outputs ----
    print("➡️ Postprocessing model output...")

    postprocess_output.postprocess_field_plot(config,log, target_time='2021-11-09 23:00')

    postprocess_output.validate_swan_vs_obs(config, log, "Rottnest")

    postprocess_output.compare_swan_bom_wind(config, log, "ROTTNEST")

    postprocess_output.validate_swan_spectrum_1d(config, log,target_time='20211109.230000', stationnm="Rottnest")

    postprocess_output.validate_swan_spectrum_2d(config, log,target_time='20211109.230000', stationnm="Rottnest")

    postprocess_output.validate_swan_spectrum_1d_obs(config, log, target_time='20211109.230000', stationnm="Rottnest")

    postprocess_output.validate_swan_spectrum_2d_obs(config, log, target_time='20211109.230000', stationnm="Rottnest")
    print("\n✅ SWAN pipeline completed successfully.")

if __name__ == "__main__":
    main()
# # Each submodule can have its own test section:
# if __name__ == "__main__":
#     import yaml
#     with open("config.yaml") as f:
#         config = yaml.safe_load(f)
#     prepare_case(config)

