To run the whole pipeline:

python main.py


To run just one step manually (for debugging):

python -m swanmod.prepare_wind


Or modify each submodule to include:

if __name__ == "__main__":
    from yaml import safe_load
    config = safe_load(open("config.yaml"))
    run(config)

Each submodule can have its own test section:

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    prepare_case(config)


Then run any single step:

python -m swanmod.run_model



export PYTHONPATH=/home/o2hpc/Project/SWANpipeline:$PYTHONPATH
pytest	Run all tests
pytest -v	Verbose output
pytest -k run_model	Run only tests matching “run_model”
pytest -s	Show print/log output
pytest --maxfail=1 --disable-warnings -q	Stop after first failure
pytest tests/test_run_model.py::test_prepare_run_folder	Run a single test function


sudo $(which python) main.py

to run background:
tmux new -s swanrun
sudo $(which python) main.py

detach safely with:
Ctrl + B, then D

tmux attach -t swanrun

Session remains alive until you:

Type exit, or

Run tmux kill-session -t swanrun