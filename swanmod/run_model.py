# modules/run_model.py
from pathlib import Path
from datetime import datetime
from swanmod import utilities
# from .utilities import setup_logger
import subprocess
import shutil  # at the top of your file
import time

# log = setup_logger(name="swan.run_model", log_file="swan.log", logdir="logs")


def prepare_case(config, log, wind_file, windstr, windend):
    """
    Prepare SWAN run directory:
      1. Copy base case
      2. Generate .swn control file
    """
    base_dir = config["paths"]["base_case"]
    case_dir = config["paths"]["case_dir"]

    # ------------------------------
    # Remove previous output folder if exists
    # ------------------------------
    output_folder = Path(case_dir) / f"{config["project_name"]}_output"
    if output_folder.exists() and output_folder.is_dir():
        log.info(f"Removing existing folder: {output_folder}")
        shutil.rmtree(output_folder)

    log.info(f"Preparing SWAN run folder: {case_dir}")
    utilities.copy_base_case(base_dir, case_dir)
    # copy hot start file if needed
    if bool(config["run_period"]["hotstart"]):
        shutil.copy(Path(config["paths"]["hotstart_dir"], f"{config["project_name"]}restartnew"), Path(config["paths"]["case_dir"], f"{config["project_name"]}restartold"))

    # Write .swn control file
    write_swn(config, log, wind_file, windstr, windend)
    # After preparing input, run SWAN
    run_swan_locally(config, log)

# def run_swan_locally(config):
#     case_dir = Path(config["paths"]["case_dir"])
#     cmd = ["time", "./swanrun", "-input", "Westport", "-omp", "88"]

#     log.info(f"Running SWAN locally: {' '.join(cmd)} in {case_dir}")
#     try:
#         start = time.time()
#         result = subprocess.run(cmd, cwd=case_dir, capture_output=True, text=True, check=True)
#         log.info(result.stdout)
#         end = time.time()
#         print(f"SWAN runtime: {(end - start)/60:.2f} minutes")
#     except subprocess.CalledProcessError as e:
#         log.error(f"SWAN run failed:\n{e.stderr}")
#         raise
def run_swan_locally(config, log):
    """
    Runs a SWAN simulation locally using an existing logger.
    """

    case_dir = Path(config["paths"]["case_dir"])
    start_time = config["run_period"]["start"].replace(":", "-").replace(" ", "_")
    end_time = config["run_period"]["end"].replace(":", "-").replace(" ", "_")

    cmd = ["/usr/bin/time", "-p", "./swanrun", "-input", f"{config["project_name"]}", "-omp", "88"]
    log.info(f"Starting SWAN for project: {config["project_name"]}")
    log.info(f"Simulation period: {start_time} → {end_time}")
    log.info(f"Working directory: {case_dir}")
    log.info(f"Command: {' '.join(cmd)}")

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=case_dir,
            capture_output=True,
            text=True,
            check=True
        )
        end = time.time()
        runtime_min = (end - start) / 60
        log.info(f"SWAN finished successfully in {runtime_min:.2f} minutes.")

    except subprocess.CalledProcessError as e:
        end = time.time()
        log.error(f"SWAN failed after {((end - start) / 60):.2f} minutes.")
        log.error(f"Error: {e.stderr.strip()}")

        raise


def write_swn(config, log, wind_file, windstr, windend):
    """
    Generate SWAN .swn file from config and boundary_list.txt.
    """
    start = datetime.fromisoformat(config["run_period"]["start"])
    end = datetime.fromisoformat(config["run_period"]["end"])
    wdt_hr = config["run_period"]["wind_dt_hours"]
    outdt_hr = config["run_period"]["output_dt_hours"]

    case_dir = Path(config["paths"]["case_dir"])
    swn_path = case_dir / f'{config["project_name"]}.swn'
    boundary_list_file = case_dir / 'boundary_list.txt'
    wind_file_path = case_dir / wind_file

    log.info("Preparing SWAN .swn file")

    # ------------------------------
    # Read boundary list (if exists)
    # ------------------------------
    if boundary_list_file.exists() and bool(config["run_period"]["use_bound"]):
        with open(boundary_list_file, "r") as f:
            boundary_lines = f.readlines()
        boundary_block = (
            "BOUN SHAPE JONSWAP 2.0 PEAK DSPR DEGREES\n"
            "!BOUN SHAPE PM PEAK DSPR DEGREES\n"
            "BOU SIDE 1 CCW VARiable FILE"
        )
        for i, line in enumerate(boundary_lines):
            if i == 0:
                # first line already starts after "FILE"
                boundary_block += " " + line
            else:
                boundary_block += " " + line
        log.info(f"Included {len(boundary_lines)} boundary files from boundary_list.txt")
    else:
        log.warning("No boundary_list.txt found or not enabled")
        boundary_block = (
            "! BOUN SHAPE JONSWAP 0.02 PEAK DSPR DEGREES\n"
            "! (no open boundary forcing provided)\n"
        )

    if wind_file_path.exists() and bool(config["run_period"]["use_wind"]):
        wind_block = (
            f"INPGRID WIND REG 338532.62917465915 6403976.356402224 0.0 22 26 3758.396647814071 4434.554447237328 EXC -999 NONSTAT {windstr} {wdt_hr} HR {windend}\n"
            f"READINP WIND 1. '{wind_file}' 3 0 0 0 FREE"
        )
        log.info(f"Included {wind_file} wind file")
    else:
        log.warning("No wind file found or not enabled")
        wind_block = (
            "! INPGRID WIND REG 143.125 -32.49 0.0 21 25 0.04 0.04 EXC -999\n"
            "! READINP WIND 1. 'XXXNOWIND' 3 0 0 0 FREE"
        )
    
    # ---hot start setup
    if bool(config["run_period"]["nexthotstart"]):
        hotout_block = (
            f"HOTFile '{config["project_name"]}restartnew' UNFORMATTED"
        )
    else:
        hotout_block = ""

    if bool(config["run_period"]["hotstart"]):
        hotin_block = (
            f"INITial HOTStart SINGle '{config["project_name"]}restartold' UNFormatted"  
        )
    else:
        hotin_block = ""
    # ------------------------------
    # Construct .swn text
    # ------------------------------
    lines = f"""$*************************HEADING************************
PROJ '{config['project_name']}' 'wwm'
$********************MODEL INPUT*************************
!
SET DEPMIN 0.50 NAUTical
MODE NONST TWOD
!
!CGRID UNSTRUC CIRCLE 36 0.0376 1. 36
CGRID UNSTRUC CIRCLE 64 0.025 0.58 36
READGRID UNSTRUC ADCIRC
!
{wind_block}
!
{boundary_block}
!
{hotin_block}
GEN3
WCAPping
!GEN3  ST6  U10PROXY 33.0
!SSWELL
!
FRICTION
BREAKING
QUAD
TRIAD
!PROP BSBT
!
POINTS 'BUOYS' FILE   'CS.loc'
TABLE  'BUOYS' HEAD   '{config["project_name"]}.tab' HS HSWELL TM01 TM02 DSPR DIR PDIR RTP TPS DIST DEP BOTLev WIND WATLEV PTHSIGN PTRTP PTDIR PTWLEN PTDSPR OUT {start.strftime("%Y%m%d.%H%M%S")} {outdt_hr} HR
SPEC 'BUOYS' SPEC2D ABS '{config["project_name"]}.spec' OUT {start.strftime("%Y%m%d.%H%M%S")} {outdt_hr} HR
SPEC 'BUOYS' SPEC1D ABS '{config["project_name"]}_1d.spec' OUT {start.strftime("%Y%m%d.%H%M%S")} {outdt_hr} HR
!
NUM ACCUR 0.01  0.02  0.02  98.5  NONSTAT  100  CSigma  0.3  &  CTheta  0.3
BLOCK 'COMPGRID' NOHEAD '{config["project_name"]}.vtk' BOTLEV HS TM01 DIR PDIR TPS DSPR WATLEV WIND OUT {start.strftime("%Y%m%d.%H%M%S")} {outdt_hr} HR
!
TEST 1,0
COMPUTE NONSTAT {start.strftime("%Y%m%d.%H%M%S")} 300 SEC {end.strftime("%Y%m%d.%H%M%S")}
{hotout_block}
STOP
!
"""

    utilities.write_text_file(swn_path, lines)
    log.info(f"SWAN input file written → {swn_path}")


# -------------------------------------------------------
# CLI runner block
# -------------------------------------------------------
# if __name__ == "__main__":

    # # Prepare SWAN case
    # prepare_case(config)