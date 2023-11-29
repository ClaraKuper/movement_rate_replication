# Project Install and Analysis Replication

This is a test project, made for learning how to:

- Clone a project from git and set up a working environment to run it
- Read and understand code from the project "[rapid manual inhibition][rmi_link]"
- Replicate a movement rate analysis from the above project

## Steps to follow:
### Installation
1. Clone this git repository on your local machine
2. Recreate the conda environment

2.1 From your console/terminal, navigate to the folder with the environment.yml file

2.2 Recreate the conda environment by entering "$ conda env create --name choose_your_env_name_here --file environment.yml"

2.3 Activate the conda environment (conda activate your_chosen_name)

3. Install the src directory as a package:

3.1 From your terminal, navigate to the right folder (the folder that contains the setup.py file)

3.2 From your terminal, run the install command "$ pip install -e ."

### Analysis Replication
You will need the right dataset to replicate the analysis. Ask [Clara][clara_git_link] to send you a download link.
1. Place the data in the folder "data"
2. Open the jupyter notebook in "scripts"
3. Follow the steps in the notebook 


[rmi_link]: https://github.com/ClaraKuper/rapid_movement_inhibition
[clara_git_link]: https://github.com/ClaraKuper