# coding:utf-8

# This file is part of the Microgrid Environment project.
#
# Copyright (C) 2025 Enrico Costanza at University College London
# with contributions from Malak Ramadan at University College London
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils import SolarBatteryEnv
from no_rl_agents import AlwaysChargeAgent, NoBatteryAgent

simulate_bookings = True

# bio stands for "booking index offset"
if simulate_bookings:
    bio = 1
else:
    bio = 0

# save the results in a folder
out_dir = Path('out/always_charge')
# create the folder if it doesn't exist
out_dir.mkdir(parents=True, exist_ok=True)

# instantiate the environment
env = SolarBatteryEnv(simulate_bookings=simulate_bookings)

# instantiate a simple agent
agent = AlwaysChargeAgent(timestep_minutes=15, battery=env._battery, simulate_bookings=simulate_bookings)
# agent = NoBatteryAgent()

# crate a list to store the results
all_results = []
columns = ['consumption', 'generation', 'import_price', 'export_price', 'hourofday', 'hourofweek', 'battery']
# if simulate_bookings is True, add bookings to the columns (at the start)
if simulate_bookings:
    columns.insert(0, 'bookings')

total_rewards = []

num_episodes = 10
for episode in range(num_episodes):
    test_episode_results = {
        'consumption': [],
        'generation': [],
        'battery_soc': [],
        'action': [],
        'reward': [],
        'import_price': [],
        'export_price': []
    }
    if simulate_bookings:
        test_episode_results['bookings'] = []
    done = False
    episode_reward = 0

    # use the episode number as part of the seed, so that the results are reproducible
    obs, _ = env.reset(seed=5 * episode) 
    # iterate through the episode
    while not done:
        # get the action from the agent
        action = agent.get_action(obs)
        # apply the action and step the environment
        next_obs, reward, done, truncated, info = env.step(action)
        # accumulate the reward
        episode_reward += reward

        # if not done, store the results
        if not done:
            test_episode_results['consumption'].append(obs[0+bio][0])
            test_episode_results['generation'].append(obs[1+bio][0])
            test_episode_results['import_price'].append(obs[2+bio][0])
            test_episode_results['export_price'].append(obs[3+bio][0])
            test_episode_results['battery_soc'].append(obs[6+bio][0])
            test_episode_results['action'].append(action[0])
            test_episode_results['reward'].append(reward)
            if simulate_bookings:
                test_episode_results['bookings'].append(obs[0][0])
        
        # update the observation
        obs = next_obs
    
    print(f'Episode {episode} reward: {episode_reward}')
    
    # convert the results to a dataframe for plotting
    results = pd.DataFrame(test_episode_results)   
    value_vars = [
            'generation', 
            'consumption', 
            'battery_soc', 
            'action', 'reward', 'import_price', 'export_price']
    # add bookings to the value_vars if simulate_bookings is True
    if simulate_bookings:
        value_vars = ['bookings'] + value_vars
    
    # reshape the results for plotting
    melted_results = pd.melt(
        results, id_vars=[], 
        value_vars=value_vars)
    g = sns.FacetGrid(melted_results, row="variable", sharey=True, aspect=4)

    # save in out folder  
    g = g.map(plt.plot, "value")
    g.savefig(out_dir / f'test_episode_{episode}.pdf')
    plt.close()


env.close()

