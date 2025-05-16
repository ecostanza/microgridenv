
## Instructions to run setup and run an example script

1. clone the repository

2. download the data archive e.g. `simulation_data.tar.xz` (shared separately) and extract it into the folder `datastore` -- that should create a subfolder named `csv`

3. create a conda environment: `conda create -n microgrid python==3.12.5 pip` and activate it (`conda activate microgrid`)

5. install the requirements.txt file: `pip install -r requirements.txt` (note this might need adjusting depending on the OS)

6. run `python run_ac_agent.py` and you should see:
```
Number of streaks: 58080
Loaded 2194 consumption files
Loaded 2194 peak files
Loaded 10848 carbon intensity files -- 217 strides
```


## Some Information about the Code

The main files are the following:

The `utils.py` file contains:
- a class `SolarBatteryEnv`, which follows the convention of Gymnasium environments;
- a class `Battery` which contains the battery parameters

The `no_rl_agents.py` file contains:
- a class `NoBatteryAgent` representing a simple agent that never uses the battery
- a class `AlwaysChargeAgent` representing a simple agent that always uses the battery, and ignores the forecast information

The `run_ac_agent.py` file:
- runs the always `AlwaysChargeAgent` on the environment for testing

