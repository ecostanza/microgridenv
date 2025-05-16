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


import numpy as np
import pandas as pd

class NoBatteryAgent():
    def __init__(self) -> None:
        pass

    def get_action(self, observation):
        return np.array([0.0])

class AlwaysChargeAgent():
    def __init__(self, timestep_minutes, battery, simulate_bookings=False) -> None:
        self._timestep_minutes = timestep_minutes
        
        self._battery = battery

        self._simulate_bookings = simulate_bookings
        # bio is the booking index offset
        self._bio = 0
        if self._simulate_bookings:
            self._bio = 1

    def get_action(self, observation):
        # generation = observation['forecast'].iloc[0]['generation']
        # consumption = observation['forecast'].iloc[0]['consumption']
        # b = observation['battery']
        # available_charge = b.capacity * (b.soc - b.min_soc)
        # available_capacity = b.capacity * (b.max_soc - b.soc)
        consumption = observation[0+self._bio][0]
        generation = observation[1+self._bio][0]

        battery_soc = observation[6+self._bio][0]
        available_charge = self._battery.capacity * (battery_soc - self._battery.min_soc)
        available_capacity = self._battery.capacity * (self._battery.max_soc - battery_soc)
        
        action = 0.0
        if generation > consumption:
            surplus = generation - consumption
            action = min(surplus, self._battery.max_charging_rate / (60.0/self._timestep_minutes))
            action = min(action, available_capacity)
        elif generation < consumption:
            deficit = consumption - generation
            action = min(deficit, self._battery.max_discharging_rate / (60.0/self._timestep_minutes))
            action = -min(action, available_charge)
        return np.array([action])

