
## Instructions to run the django project

1. clone the repository

2. download the data archive e.g. `simulation_data.tar.xz` (shared separately) and extract it into the folder `datastore` -- that should create a subfolder named `csv`

3. create a conda environment: `conda create -n microgrid python==3.12.5 pip` and activate it (`conda activate microgrid`)

5. install the requirements.txt file: `pip install -r requirements.txt` (note this might need adjusting depending on the OS)

6. 


## Instructions to run the training code

1. download the data archive `simulation_data.tar.xz` (shared separately) and extract it into the folder `datastore` -- that should create a subfolder named `csv`

2. install the requirements.txt file (ideally in a virual environment or conda environment)

3. run the `resample.py` file on the csv data to preprocess

<!-- 2. install the following libraries (ideally in a virtual environment or conda environment): -->
<!-- `pip install pandas seaborn PySide2 openpyxl xlsxwriter` -->

<!-- 3. to run the test code `python rl_training/run_test.py` -->

Apart from the csv files, everything needed for the RL training should be in the `rl_training` subfolder. All the other subfolders are for the django web app, and can be ignored.



The main files are the following:

The `train_agent_sb.py` where you will find:
- an agent that uses the stable baselines3 TD3 model
- a 'CustomFeaturesExtractor' that adds convolutional layers to process the forecast data

The `utils.py` file contains:
- a class 'SolarBatteryEnv`, which follows the convention of Gymnasium environments;
- a class `Battery` which contains the battery parameters

The `run_agent.py` file:
- runs a pretrained agent on the environment for evaluation

The `models` folder contains different trained models:

- `model_1` was trained for over 6500 streaks, and contains a features extractor with one conv layer.

- `model_2` was trained for over 6500 streaks, and contains a feautres extractor with two conv layers.

- `model_3` was trained for over 6500 streaks. It contains the same architecture as `model_2`, but incorporates simulated bookings in the training as well.

<!-- The main file is `run_test.py` in this file you will find:
- a class `SolarBatteryEnv`, which loosely follows the convention of Gymnasium environments; however the data types are currently based on pandas dataframes, rather than Gymnasium data types, so some adjustment might be required
- two basic agent classes: `NoBatteryAgent` and `AlwaysChargeAgent` these were created just for testing the environment before having access to an actual RL agent
- a function `run_iteration` which demonstrates how to call the environments methods -->

<!-- At the end of the file there is a loop that calls run_iteration 30 times. In this loop, you will see some references to a `plan` function -- please ignore these, as this was just to compare the agent operation to a linear programming planner/solver. -->

The following subfolders in `non-rl_attempts` were used in the development of this agent but are not relevant for training/running the current agent:
- `dqn`: This contains implementations and resources related to Deep Q-Network (DQN), a popular RL algorithm. It includes code for training and evaluating a DQN model.
- `mountain_car`: This subfolder probably includes code and resources for the Mountain Car environment, an OpenAI environment. It contains scripts for training RL agents in this environment.
- `OR_planner`: This subfolder contains resources related to an Operations Research (OR) planner, which involves optimization algorithms and planning strategies used in conjunction with our environment.
- `supervised_learning`: This subfolder includes code and resources for supervised learning tasks. It contains scripts for training and evaluating supervised learning models, which was used as part of the RL training process to finetune the network architecture.

Future suggestions for training the model:
- `model_3` is interesting in that it incorporates bookings for each episode. I would try retraining where bookings are incorporated for some episodes but not all (maybe 50%).
- Consider adding a booking forecast to the observation input. This means that the agent would treat the booking forecast as a seperate input.