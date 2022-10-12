# PythonProject

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install pip-tools `pip3 install pip-tools`
- Update dev requirements: `pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in --upgrade`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
`pip-compile --output-file=requirements.txt requirements.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`

# Assignments

- Assignment 1 located in 'src' folder with the name of 'hw1.py'
- Assignment 2 located in 'src/hw2' folder with the name of 'hw2.py'
- Assignment 3 located in 'src/hw3' folder
