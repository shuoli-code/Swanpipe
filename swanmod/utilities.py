# modules/utilities.py
import shutil
from datetime import datetime
from pathlib import Path
import logging

def copy_base_case(base_dir, work_dir):
    """Copy base SWAN setup into working directory."""
    base = Path(base_dir)
    work = Path(work_dir)
    if work.exists():
        print(f"⚠️  {work} exists — contents may be overwritten.")
    else:
        work.mkdir(parents=True)
    for item in base.glob("*"):
        dest = work / item.name
        if item.is_file():
            shutil.copy(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
    print(f"📂 Copied base case from {base_dir} → {work_dir}")

def timestamp():
    """Return a timestamp string for logging or file naming."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_text_file(path, lines):
    """Write list of lines to a text file."""
    with open(path, "w") as f:
        # f.writelines([line + "\n" for line in lines])
        f.write(lines)  # write entire string at once
    print(f"💾 Wrote {path}")

def setup_logger(name="swan", log_file="swan.log", logdir="logs", level=logging.INFO, console=True):
    """
    Sets up a logger that writes both to file and optionally to console.

    Parameters
    ----------
    name : str
        Logger name (e.g., 'swan.run_model').
    log_file : str
        Log file name.
    level : logging level
        Logging level, e.g., logging.INFO, logging.DEBUG.
    console : bool
        Whether to also show logs in console.

    Returns
    -------
    logging.Logger
    """

    # Ensure logs directory exists
    log_dir = Path(logdir)
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / log_file

    # Create logger (singleton pattern)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if function is called multiple times
    if not logger.handlers:

        # File handler
        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(level)

        # Console handler
        if console:
            ch = logging.StreamHandler()
            ch.setLevel(level)
        else:
            ch = None

        # Formatter (for both)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh.setFormatter(formatter)
        if ch:
            ch.setFormatter(formatter)

        # Add handlers
        logger.addHandler(fh)
        if ch:
            logger.addHandler(ch)

    return logger