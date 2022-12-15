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
- Assignment 4 located in 'src/hw4' folder
  - There are still many improvements needed, however, I've tried my best in my ability.
- Midterm located in 'src/midterm' folder
  - In Assignment 4, I didn't fully understand the "Difference with Mean of Response" concept, updated it in this assignment.
  - The main_midterm.py contains the main function that allows the project to run.
- Assignment 5 located in 'src/hw5' folder
  - the precommit is fixed with the modifcation to the yaml file.
- Assignment 6: docker-compose.yml and Dockerfile located in the general folder, and the scripts are in 'src/hw6' folder.
- Final project is using the hw5.sql and hw5.py files to generate report
  - the report includes hw4 and mideterm analyzer for all predictor, but the model score are reflecting the best score only.
  - the report will be generated in 'src/hw5' folder, and the plots will be generated in 'src/hw5/output' folder
