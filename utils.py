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

import math
import random
from pathlib import Path

import pandas as pd
import numpy as np
import gymnasium 
from gymnasium import spaces
import matplotlib.pyplot as plt

plotting = False

battery_params = {
    'max_charging_rate': 3.6,
    'max_discharging_rate':	2,
    'capacity':	5,
    'charging_eff':	0.05,
    'discharging_eff':	0.05,
    'min_soc':	0.1,
    'max_soc':	0.95,
    'init_soc':	0.1
}

class Battery:
    def __init__(self, 
                 max_charging_rate: float, 
                 max_discharging_rate: float, 
                 capacity: float, 
                 charging_eff: float, 
                 discharging_eff: float, 
                 min_soc: float, 
                 max_soc: float, 
                 init_soc: float):
        self.max_charging_rate = max_charging_rate
        self.max_discharging_rate = max_discharging_rate
        self.capacity = capacity
        self.charging_eff = charging_eff
        self.discharging_eff = discharging_eff
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.init_soc = np.random.uniform(min_soc, max_soc)
        # self.soc = self.init_soc
        # random initial state of charge
        self.soc = np.random.uniform(min_soc, max_soc)
    
    def __str__(self):
        return f'Battery(capacity={self.capacity})'

timestep_minutes = 15
streak_days = 14
annotation_days=7
coordination_days = streak_days - annotation_days
panel_size = 3.0

# min_price = 0.24
# max_price = min_price * 10 # 98 p/kWH

data_dir = Path('datastore/csv')
out = Path('out')

# create the folder out, if it does not exist
out.mkdir(exist_ok=True)



class SolarBatteryEnv(gymnasium.Env):
    
    def __init__(self, simulate_bookings=False, bookings_rate=1.0,
                    min_price=0.24, max_price=2.40, price_carbon_factor=1.0,
                    export_price_factor=1.0
                 ):
        super(SolarBatteryEnv, self).__init__()

        self._simulate_bookings = simulate_bookings
        self._bookings_rate = bookings_rate
        # self._bio is the booking index offset -- if bookings are simulated, 
        # we need to offset the index of the observations by 1
        self._bio = 0
        if self._simulate_bookings:
            self._bio = 1
        
        self._min_price = min_price
        self._max_price = max_price
        self._price_carbon_factor = price_carbon_factor
        self._export_price_factor = export_price_factor

        self._delta_price = self._max_price - self._min_price

        self.render_mode = None

        self._solar_folder = data_dir / 'solar_generation_resampled'
        self._consumption_folder = data_dir / 'electricity_consumption_resampled'

        self._peaks_folder = data_dir / 'peaks'

        self._streaks = pd.read_csv(data_dir / 'streaks_lut.csv')
        self._streaks['start'] = pd.to_datetime(self._streaks['start'])
        self._max_streaks = self._streaks['lut'].max()
        print(f'Number of streaks: {self._max_streaks}')

        # get the pricing data (based on carbon intensity)
        # look for how many strides are available for the duration of the 
        # consumption stride and then select one randomly aligning the start 
        # in terms of hour of the day
        # the data is in half hour steps
        # self._carbon_intensity_folder = data_dir / 'carbon_intensity_resampled'
        # carbon_intensity_files = sorted(list(self._carbon_intensity_folder.glob('*.csv')))
        # self._min_carbon_intensity = 100
        # self._max_carbon_intensity = 0
        # for f in carbon_intensity_files:
        #     df = pd.read_csv(f)
        #     max_value = df['forecast'].max()
        #     min_value = df['forecast'].min()
        #     self._min_carbon_intensity = min(self._min_carbon_intensity, min_value)
        #     self._max_carbon_intensity = max(self._max_carbon_intensity, max_value)


        # self._carbon_intensity_start = pd.to_datetime(carbon_intensity_files[0].stem.replace('carbon_intensity_', '').replace('Z', ''))
        # carbon_intensity_end = pd.to_datetime(carbon_intensity_files[-1].stem.replace('carbon_intensity_', '').replace('Z', ''))
        # print('carbon_intensity_start:', self._carbon_intensity_start)

        # carbon_intensity_days = int(math.floor((carbon_intensity_end - self._carbon_intensity_start).days))

        # # we subtract 1 from the number of days because 
        # # we want to align the start of the carbon intensity stride
        # # with the start of the consumption stride so we loose some data
        # self._carbon_intensity_strides = carbon_intensity_days - coordination_days - 1

        # Preload consumption data
        self._consumption_data = {}
        consumption_files_list = self._consumption_folder.glob("*.csv")
        # consumption_files_list = list(self._consumption_folder.glob("*.csv"))
        # consumption_files_list = consumption_files_list[:100]
        for file in consumption_files_list:
            # example file: datastore/csv/electricity_consumption_resampled/2023-02-23-99.csv
            # date, sensor_id = file.stem.split('-')
            parts = file.stem.split('-')
            sensor_id = parts[3]
            date = parts[:3]
            date = '-'.join(date)
            self._consumption_data[(date, sensor_id)] = pd.read_csv(file, parse_dates=['time']).set_index('time')
        print(f'Loaded {len(self._consumption_data)} consumption files')

        if self._simulate_bookings:
            # Preload peaks data
            self._peaks_data = {}
            peak_files_list = self._peaks_folder.glob("*.csv")
            # consumption_files_list = list(self._consumption_folder.glob("*.csv"))
            # consumption_files_list = consumption_files_list[:100]
            for file in peak_files_list:
                # example file: datastore/csv/electricity_consumption_resampled/2023-02-23-99.csv
                # sensor_id, date = file.stem.split('_')
                parts = file.stem.split('-')
                sensor_id = parts[3]
                date = parts[:3]
                date = '-'.join(date)
                curr_peaks = pd.read_csv(file, parse_dates=['time']).set_index('time')

                # TODO: currently only ~ 0.5 rate is used if _bookings_rate < 1.0
                # fix this by approximating the booking rate more accurately
                if self._bookings_rate < 1.0:
                    # add only 50% of the bookings 
                    # add a column with the diff in value
                    curr_peaks['diff'] = curr_peaks['value'].diff()
                    curr_peaks['segment'] = (curr_peaks['diff'].abs() > 0) & (curr_peaks['value'] > 0)
                    # convert gap info to 0 and 1
                    curr_peaks['segment'] = curr_peaks['segment'].apply(lambda x: 1 if x else 0)
                    # "fill down"
                    curr_peaks['segment'] = curr_peaks['segment'].cumsum()
                    # set segment to 0 if the value is 0
                    curr_peaks['segment'] = curr_peaks['segment'].where(curr_peaks['value'] > 0, 0)
                    # curr_peaks.to_csv('test.csv', index=True)
                    # set value to zero if segment is even
                    curr_peaks['value'] = curr_peaks['value'].where(curr_peaks['segment'] % 2 == 1, 0)
                    
                    curr_peaks = curr_peaks.drop(columns=['diff', 'segment'])
                
                self._peaks_data[(date, sensor_id)] = curr_peaks

            print(f'Loaded {len(self._peaks_data)} peak files')

        # Preload solar forecast data
        solar_lut = pd.read_csv(data_dir / 'solar_generation_lut.csv')
        self._solar_data = {}
        # unique_files = set()

        for _, row in solar_lut.iterrows():
            date = row['date']
            to_load = row['replacement']
            file = self._solar_folder / f'{to_load}.csv'
            self._solar_data[date] = pd.read_csv(file, parse_dates=['time'])

            # change 'time' column if needed
            if date != to_load:
                date_dt = pd.to_datetime(date)
                to_load_dt = pd.to_datetime(to_load)
                delta = date_dt - to_load_dt
                self._solar_data[date]['time'] = self._solar_data[date]['time'] + delta
                # print(f'Changed time for {date} to {to_load}')

            # unique_files.add(to_load)
        # print(f'Loaded {len(unique_files)} solar forecast files')

        self._carbon_intensity_folder = data_dir / 'carbon_intensity_resampled'
        self._min_carbon_intensity = 100
        self._max_carbon_intensity = 0

        # Preload carbon intensity data
        
        self._carbon_intensity_data = {}
        for file in self._carbon_intensity_folder.glob("*.csv"):
            timestamp = file.stem.replace("carbon_intensity_", "").replace("Z", "")
            self._carbon_intensity_data[timestamp] = pd.read_csv(file, parse_dates=['request_time', 'datetime'])
            max_value = self._carbon_intensity_data[timestamp]['forecast'].max()
            min_value = self._carbon_intensity_data[timestamp]['forecast'].min()
            self._min_carbon_intensity = min(self._min_carbon_intensity, min_value)
            self._max_carbon_intensity = max(self._max_carbon_intensity, max_value)

        self._delta_carbon_intensity = self._max_carbon_intensity - self._min_carbon_intensity

        carbon_intensity_files = sorted(list(self._carbon_intensity_folder.glob('*.csv')))
        self._carbon_intensity_start = pd.to_datetime(carbon_intensity_files[0].stem.replace('carbon_intensity_', '').replace('Z', ''))
        carbon_intensity_end = pd.to_datetime(carbon_intensity_files[-1].stem.replace('carbon_intensity_', '').replace('Z', ''))
        # print('carbon_intensity_start:', self._carbon_intensity_start)

        carbon_intensity_days = int(math.floor((carbon_intensity_end - self._carbon_intensity_start).days))

        # we subtract 1 from the number of days because 
        # we want to align the start of the carbon intensity stride
        # with the start of the consumption stride so we loose some data
        self._carbon_intensity_strides = carbon_intensity_days - coordination_days - 1

        print(f'Loaded {len(self._carbon_intensity_data)} carbon intensity files -- {self._carbon_intensity_strides} strides')

        self._days_ahead = 2
        self._time_ahead = self._days_ahead * 24 * 4

        self._battery = Battery(**battery_params)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,self._time_ahead), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7+self._bio,self._time_ahead), dtype=np.float32)

    def _get_obs(self):
        '''
        Get the observation for the current timestep in form:
        [
            bookings, # (if present)
            consumption,
            generation,
            import_price,
            export_price,
            hourofday,
            hourofweek,
            battery 
        ]
        '''
        curr_dt = self._start + pd.DateOffset(minutes=timestep_minutes*self._current_step)

        c_idx = self._consumption['time'] == curr_dt
        sf_idx = self._solar_forecast['time'] == curr_dt
        c_i_idx = self._carbon_intensity['time'] == curr_dt

        timestamps = pd.date_range(curr_dt, periods=self._time_ahead, freq=f'{timestep_minutes}min')

        sf = self._solar_forecast[sf_idx].drop(columns=['time'])
        sf = sf.drop_duplicates(subset=['forecast_time'])
        sf = sf.set_index('forecast_time').reindex(timestamps).fillna(0)
        sf = sf['value'].to_numpy()  # Solar forecast as NumPy
        
        # generate consumption forecast for days_ahead days based on average_days
        consumption_forecast = np.array([])  # Start with an empty NumPy array
        curr_qh = curr_dt.minute / 15 + curr_dt.hour * 4
        # the first day we need data from curr_dt.hour to the end of the day
        for i in range(self._days_ahead+1):
            weekday = 'weekend'
            if (curr_dt + pd.Timedelta(days=i)).weekday() < 5:
                weekday = 'weekday'
            if i == 0:
                # first day
                day_consumption = self._average_days.loc[curr_qh:, weekday].to_numpy()
            
            elif i == self._days_ahead:
                # last day
                day_consumption = self._average_days.loc[:curr_qh, weekday].iloc[:-1].to_numpy()
            else:
                # middle days
                day_consumption = self._average_days.loc[:, weekday].to_numpy()
            consumption_forecast = np.concatenate((consumption_forecast, day_consumption))
        
        # set the first value to be the actual consumption, rather than the forecast
        try:
            value = self._consumption[c_idx]['value']
            if value.empty:
                consumption_forecast[0] = 0.0
            else:
                consumption_forecast[0] = self._consumption[c_idx]['value'].iloc[0]
        except ValueError :
            # consumption_forecast.iloc[0, 0] = 0
            consumption_forecast[0] = 0.0
        
        if self._simulate_bookings:
            # add simulated bookings
            # the _peaks array corresponds to coordination, and _current_step is relative to coordination
            peak_start_idx = self._current_step
            # we have 48h forecast in quarter-hourly intervals
            peak_end_idx = peak_start_idx + 48 * 4
            current_peaks = self._peaks[peak_start_idx:peak_end_idx]['value'].to_numpy()
            if len(current_peaks) < 48 * 4:
                # pad with zeros
                current_peaks = np.pad(current_peaks, (0, (48 * 4) - len(current_peaks)), mode='constant', constant_values=0.0)

        curr_carbon_intensity = self._carbon_intensity.loc[c_i_idx, 'value']
        
        try:
            # use numpy for import and export price
            import_price_forecast = np.array([self.import_price_forecast.loc[curr_carbon_intensity.index].values[0]], dtype=np.float32)
            # make import price 2x the export price
            # import_price_forecast = import_price_forecast * 2
            export_price_forecast = np.array([self.export_price_forecast.loc[curr_carbon_intensity.index].values[0]], dtype=np.float32)
            # pad to match the forecast state
            import_price_forecast = np.pad(import_price_forecast, (0, self._time_ahead - len(import_price_forecast)), mode='constant', constant_values=0.0)
            export_price_forecast = np.pad(export_price_forecast, (0, self._time_ahead - len(export_price_forecast)), mode='constant', constant_values=0.0)
        except IndexError:
            print(f'IndexError from price forecast')
            # print('curr_carbon_intensity:', curr_carbon_intensity)
            import_price_forecast = np.zeros(self._time_ahead)
            export_price_forecast = np.zeros(self._time_ahead)

        # Extract hour of day and hour of week as NumPy arrays
        hourofday = np.array([timestamps[0].hour], dtype=np.float32)
        hourofweek = np.array([timestamps[0].dayofweek * 24 + hourofday[0]], dtype=np.float32)

        hourofday = np.pad(hourofday, (0, self._time_ahead - len(hourofday)), mode='constant', constant_values=0.0)
        hourofweek = np.pad(hourofweek, (0, self._time_ahead - len(hourofweek)), mode='constant', constant_values=0.0)

        battery = np.array([self._battery.soc], dtype=np.float32)
        # # pad the battery state to match the forecast state
        battery = np.pad(battery, (0, self._time_ahead - len(battery)), mode='constant', constant_values=0.0)

        forecasts = [consumption_forecast, sf, import_price_forecast, export_price_forecast, hourofday, hourofweek, battery]
        for array in forecasts:
            if array.shape[0] < self._time_ahead:
                array = np.pad(array, (0, self._time_ahead - array.shape[0]), mode='edge')

        # turn the 7 lists into arrays
        # obs = np.asarray([consumption_forecast[:self._time_ahead], sf[:self._time_ahead], import_price_forecast[:self._time_ahead], export_price_forecast[:self._time_ahead], hourofday[:self._time_ahead], hourofweek[:self._time_ahead], battery[:self._time_ahead]], dtype=np.float32)
        if self._simulate_bookings:
            obs = np.asarray([
                current_peaks, 
                consumption_forecast[:self._time_ahead], 
                sf[:self._time_ahead], 
                import_price_forecast[:self._time_ahead], 
                export_price_forecast[:self._time_ahead], 
                hourofday[:self._time_ahead], 
                hourofweek[:self._time_ahead], 
                battery[:self._time_ahead]
            ], dtype=np.float32)
        else:
            obs = np.asarray([
                consumption_forecast[:self._time_ahead], 
                sf[:self._time_ahead], 
                import_price_forecast[:self._time_ahead], 
                export_price_forecast[:self._time_ahead], 
                hourofday[:self._time_ahead], 
                hourofweek[:self._time_ahead], 
                battery[:self._time_ahead]
            ], dtype=np.float32)
        
        obs = np.nan_to_num(obs)

        return obs
    
    def generate_prediction(self, start, sensor_id):
        # get the consumption data for the training period
        # we'll use this for 'training' -- actually just the average weekday and weekend day, at least for now..
        # Use preloaded consumption data
        training_consumption = []
        for d in range(annotation_days):
            curr = (start + pd.DateOffset(days=d)).date()
            key = (str(curr), str(sensor_id))
            if key in self._consumption_data:
                consumption = self._consumption_data[key]
                training_consumption.append(consumption)
            else:
                print(f"Missing consumption data for {key}")

        training_consumption = pd.concat(training_consumption)

        # make a copy of training_consumption
        # training_consumption_copy = training_consumption.copy()

        # calculate the average hour by hour
        hourly_training_consumption = training_consumption.resample('h').mean()

        # split the data by weekday and weekend
        hourly_training_consumption['weekday'] = hourly_training_consumption.index.weekday
        hourly_training_consumption['hour'] = hourly_training_consumption.index.hour
        hourly_training_consumption['weekend'] = hourly_training_consumption['weekday'] >= 5
        self._average_days = hourly_training_consumption.groupby(['weekend', 'hour'])['value'].mean().unstack(level=0)
        self._average_days.columns = ['weekday', 'weekend']
        # convert the index to timedeltaindex
        self._average_days.index = pd.to_timedelta(self._average_days.index, unit='h')

        # TODO: run a low pass filter on both columns of self._average_days
        # from utils import filter_consumption
        # from simulate_bookings import filter_consumption
        # filter the data
        # mav_weekday = filter_consumption(self._average_days['weekday'].to_numpy(), use_moving_average=True, window_size=4)
        # fil_weekday = filter_consumption(self._average_days['weekday'].to_numpy(), use_moving_average=False)
        # mav_weekend = filter_consumption(self._average_days['weekend'].to_numpy(), use_moving_average=True, window_size=4)
        # fil_weekend = filter_consumption(self._average_days['weekend'].to_numpy(), use_moving_average=False)

        # self._average_days['weekday'] = filter_consumption(self._average_days['weekday'].to_numpy(), use_moving_average=False)
        # self._average_days['weekend'] = filter_consumption(self._average_days['weekend'].to_numpy(), use_moving_average=False)
        
        # # plot the original data and the filtered version
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        # plt.sca(ax1)
        # plt.plot(range(len(self._average_days['weekday'])), self._average_days['weekday'], label='original')
        # plt.plot(range(len(self._average_days['weekday'])), mav_weekday, label='moving average')
        # plt.plot(range(len(self._average_days['weekday'])), fil_weekday, label='filtered')
        # plt.legend()


        # plt.sca(ax2)
        # plt.plot(range(len(self._average_days['weekend'])), self._average_days['weekend'], label='original')
        # plt.plot(range(len(self._average_days['weekend'])), mav_weekend, label='moving average')
        # plt.plot(range(len(self._average_days['weekend'])), fil_weekend, label='filtered')

        # plt.savefig(f'{self._streak_id}_average_days.pdf')

        # resample to 15 minute intervals
        self._average_days = self._average_days.resample(f'{timestep_minutes}min').ffill()
        self._average_days = self._average_days.reset_index(drop=True)
        #  replicate last row 3 times
        last_row_index = self._average_days.index[-1]
        for i in range(3):
            last_row_index += 1
            self._average_days.loc[last_row_index] = self._average_days.iloc[-1]
        # self._average_days.to_excel(out / 'average_days_resampled.xlsx')


    def reset(self, seed=None):
        # select a random streak
        random.seed(seed)
        self._streak_id = random.randint(0, self._max_streaks - 1)

        # self._streak_id = 34038
        # self._streak_id = 12994
        # self._streak_id = 33892
        # self._streak_id = 41549
        # self._streak_id = 47533
        # self._streak_id = 6885
        # self._streak_id = 52630
        # self._streak_id = 13695

        print(f'self._streak_id: {self._streak_id}')

        selection = self._streaks[self._streaks['lut'] <= self._streak_id]
        if len(selection) == 0:
            # select first row
            selection = self._streaks.iloc[[0]]
            self._streak_id = selection['lut'].values[0]

        # select the next row in the self._streaks dataframe
        row = self._streaks.iloc[len(selection)]
        # try:
        #     # select the next row in the self._streaks dataframe
        #     row = self._streaks.iloc[len(selection)]
        #     print('all good')
        # except IndexError:
        #     print('IndexError!')
        #     truncated = True
        #     terminated = True
        #     reward = 0
        #     info = {
        #         'energy_import': 0,
        #         'energy_export': 0
        #     }
        #     obs = np.zeros((8, self._time_ahead), dtype=np.float32)
        #     return obs, reward, terminated, truncated, info

        hh_offset = int(self._streak_id - row['lut'] + row['hh_steps'])
        # TODO why 30? EC: because it's 30 minutes -- is it correct?
        offset = pd.DateOffset(minutes=hh_offset * 30)

        sensor_id = row['sensor_id']

        start = row['start'] + offset

        self.generate_prediction(start, sensor_id)

        # get the solar generation and consumption data for all dates needed
        all_solar_forecasts = []
        coordination_consumption = []
        coordination_peaks = []
        
        for d in range(annotation_days, streak_days):
            curr = start + pd.DateOffset(days=d)

            # find the solar forecast file from the preloaded data
            solar_data = self._solar_data.get(str(curr.date()))
            # TODO: remove the following as it should not be needed
            if solar_data is None:
                all_dates = list(self._solar_data.keys())
                # TODO: change this so that it does not take into account the year, but just day and month
                closest_date = min(all_dates, key=lambda x: abs(pd.to_datetime(x) - curr))
                solar_data = self._solar_data.get(str(closest_date))
                print(f'filling solar gap for {curr.date()}, closest date: {closest_date}')

            all_solar_forecasts.append(solar_data)
            
            consumption = self._consumption_data.get((str(curr.date()), str(sensor_id)))
            coordination_consumption.append(consumption)

            if self._simulate_bookings:
                peaks = self._peaks_data.get((str(curr.date()), str(sensor_id)))
                coordination_peaks.append(peaks)

        self._solar_forecast = pd.concat(all_solar_forecasts, ignore_index=True)
        self._solar_forecast['time'] = self._solar_forecast['time'].dt.tz_localize(None)
        # arbitrary scaling factor
        self._solar_forecast['value'] /= 1000
        self._solar_forecast['value'] *= panel_size
        self._solar_forecast['forecast_time'] = self._solar_forecast['time'] + (self._solar_forecast['minutes_offset'] * pd.Timedelta('1 min'))
        self._solar_forecast = self._solar_forecast.drop(columns=['minutes_offset'])

        self._consumption = pd.concat(coordination_consumption)
        self._consumption = self._consumption.reset_index()
        self._consumption['time'] = self._consumption['time'].dt.tz_localize(None)

        if self._simulate_bookings:
            self._peaks = pd.concat(coordination_peaks)
            self._peaks = self._peaks.reset_index()
            self._peaks['time'] = self._peaks['time'].dt.tz_localize(None)

        # ------

        coordination_start = start + pd.DateOffset(days=annotation_days)

        # get a random carbon intensity stride
        # c_i_offset = random.randint(0, self._carbon_intensity_strides)
        # select a carbon intensity stride in a way that is reproducible
        c_i_offset = self._streak_id % self._carbon_intensity_strides
        # print(f'c_i_offset: {c_i_offset}')

        carbon_intensity_delta = pd.DateOffset(days=c_i_offset)

        start_hour = coordination_start.hour
        start_minute = coordination_start.minute

        carbon_intensity_stride_start = self._carbon_intensity_start + carbon_intensity_delta
        carbon_intensity_stride_start = carbon_intensity_stride_start.replace(hour=start_hour, minute=start_minute)

        # store the data in a dataframe and use the same format as the solar and consumption data
        # the carbon intensity data is in half hour steps, but later we'll resample it to 15 minute steps
        all_carbon_intensity = []
        carbon_intensity_steps = coordination_days * 24 * 2

        # TODO: why +1 ?
        for i in range(carbon_intensity_steps+1):
            carbon_intensity_curr = carbon_intensity_stride_start + pd.DateOffset(minutes=i*30)
            curr = coordination_start + pd.DateOffset(minutes=i*30)

            carbon_intensity_fname =  carbon_intensity_curr.strftime('carbon_intensity_%Y-%m-%dT%H-%M-%SZ')

            timestamp = carbon_intensity_fname.replace("carbon_intensity_", "").replace("Z", "")
            carbon_intensity_fpath = self._carbon_intensity_data.get(str(timestamp))
            # TODO: remove the following as it should not be needed
            if carbon_intensity_fpath is None:
                print('filling carbon gap')
                all_dates = list(self._carbon_intensity_data.keys())
                closest_date = min(all_dates, key=lambda x: abs(pd.to_datetime(x) - pd.to_datetime(timestamp)))
                carbon_intensity_fpath = self._carbon_intensity_data.get(str(closest_date))
                if carbon_intensity_fpath is None:
                    # pick a random carbon intensity date
                    carbon_intensity_fpath = random.choice(list(self._carbon_intensity_data.values()))
                    print(f"Missing carbon intensity data for {timestamp}")

            df = carbon_intensity_fpath
            # rename columns
            df = df.rename(columns={'request_time': 'time', 'offset': 'minutes_offset', 'forecast': 'value'})
            df = df.drop(columns=['datetime', 'actual'])

            df['time'] = curr

            df = df[df['minutes_offset'] >= 0]

            if not df.empty:
                all_carbon_intensity.append(df)

        self._carbon_intensity = pd.concat(all_carbon_intensity)
        self._carbon_intensity['offset'] = self._carbon_intensity['minutes_offset'] * pd.Timedelta('1 min')
        self._carbon_intensity['forecast_time'] = self._carbon_intensity['time'] + self._carbon_intensity['offset']
        self._carbon_intensity = self._carbon_intensity.drop(columns=['offset', 'index'])
        self._carbon_intensity = self._carbon_intensity.drop_duplicates()
        self._carbon_intensity = self._carbon_intensity.set_index(['time', 'minutes_offset'])

        # resample 
        self._carbon_intensity = self._carbon_intensity.unstack(level=[1])
        self._carbon_intensity = self._carbon_intensity.resample(f'{timestep_minutes}min').ffill()
        self._carbon_intensity = self._carbon_intensity.stack(level=[1], future_stack=True)

        self._carbon_intensity = self._carbon_intensity.reset_index()
        self._carbon_intensity = self._carbon_intensity.set_index(['forecast_time', 'time'])

        # sort self._carbon_intensity by the index
        self._carbon_intensity = self._carbon_intensity.sort_index()

        try:
            self._carbon_intensity = self._carbon_intensity.unstack(level=[1])
        except ValueError as e:
            print('dropna on carbon_intensity')
            # print(self._carbon_intensity)
            self._carbon_intensity.dropna(axis='index', how='any', inplace=True)
            # self._carbon_intensity.fillna(value=0, inplace=True)
            self._carbon_intensity = self._carbon_intensity.unstack(level=[1])

        self._carbon_intensity = self._carbon_intensity[self._carbon_intensity.index.notnull()]

        self._carbon_intensity = self._carbon_intensity.resample(f'{timestep_minutes}min').ffill()
        self._carbon_intensity = self._carbon_intensity.stack(level=[1], future_stack=True)

        self._carbon_intensity = self._carbon_intensity.dropna(axis='index', how='any')
        # self._carbon_intensity = self._carbon_intensity.fillna(value=0)
        self._carbon_intensity = self._carbon_intensity.reset_index()
        
        self._carbon_intensity['offset'] = self._carbon_intensity['forecast_time'] - self._carbon_intensity['time']
        self._carbon_intensity['minutes_offset'] = self._carbon_intensity['offset'] / pd.Timedelta('1 min')

        self._carbon_intensity = self._carbon_intensity[self._carbon_intensity['minutes_offset'] >= 0].reset_index()
        # print(self._carbon_intensity)

        self._carbon_intensity = self._carbon_intensity[self._carbon_intensity['minutes_offset'] < self._days_ahead * 24 * 60]

        self.import_price_forecast = (
            self._min_price + self._delta_price * (
                self._carbon_intensity['value'] - self._min_carbon_intensity
                ) / self._delta_carbon_intensity)
        self.export_price_forecast = self.import_price_forecast / self._export_price_factor

        self._start = start + pd.DateOffset(days=annotation_days)
        self._end = start + pd.DateOffset(days=streak_days)

        self._steps = int((self._end - self._start) / pd.Timedelta(f'{timestep_minutes} min'))

        self._current_step = 0


        # reset the battery
        self._battery.soc = self._battery.init_soc

        info = {}
        # return the initial state
        obs = self._get_obs()
        return obs, info

    @property
    def steps(self):
        return self._steps

    def step(self, action):
        # print(f'Step: {self._current_step}/{self._steps}')
        # print(f'action: {action}')
        # if action is array, take the first element
        if isinstance(action, np.ndarray):
            action = self.scale_action(action[0])
            # action = action[0]
        truncated = False
        terminated = self._current_step > self._steps - 1
        reward = 0
        info = {
            'energy_import': 0,
            'energy_export': 0
        }
        if terminated:
            # print("current step", self._current_step)
            # print("steps:", self._steps)
            # create dummy obs to return
            # obs = np.zeros((7, self._time_ahead), dtype=np.float32)
            obs = np.zeros((7+self._bio, self._time_ahead), dtype=np.float32)
            return obs, reward, terminated, truncated, info
        
        # apply the action
        # positive action: charge the battery
        # negative action: discharge the battery
        available_charge_capacity = (self._battery.max_soc - self._battery.soc)
        available_discharge_capacity = (self._battery.soc - self._battery.min_soc)

        charge = 0
        penalty = 0

        if action < 0:
            if action >= -available_discharge_capacity:
                charge = action
            # if the action would exceed the battery capacity, apply a penalty
            if (self._battery.soc + action) < self._battery.min_soc:
                penalty = 0.1

        elif action > 0:
            if action <= available_charge_capacity:
                charge = action
                charge = charge * (1 - self._battery.charging_eff)  # Apply charging efficiency
            # if the action would exceed the battery capacity, apply a penalty
            if (self._battery.soc + action) > self._battery.max_soc:
                penalty = 0.1


        self._battery.soc += charge

        observation = self._get_obs() 

        # calculate the reward
        # reward = -cost of importing power + income from selling power
        # self._bio is the booking index offset -- if bookings are there, then everyhting is shifted by 1
        consumption = observation[0+self._bio][0]
        generation = observation[1+self._bio][0]
        import_price = observation[2+self._bio][0]
        export_price = observation[3+self._bio][0]

        energy_import = max(consumption + (charge * self._battery.capacity) - generation, 0)
        energy_export = max(generation - consumption - (charge * self._battery.capacity), 0)

        demand = max(consumption - generation, 0)
        surplus = max(generation - consumption, 0)

        balance = energy_export * export_price - energy_import * import_price
        balance_without_battery = surplus * export_price - demand * import_price
        # balance = round(balance, 10)  # Round balance to a specific number of decimal places
        # balance_without_battery = round(balance_without_battery, 10)
        reward = balance - balance_without_battery
        if charge == 0:
            reward = 0
        # reward -= penalty


        info = {
            'energy_import': energy_import,
            'energy_export': energy_export
        }
        self._current_step += 1

        return observation, reward, terminated, truncated, info

    def scale_action(self, action):
        charge_rate = (self._battery.max_charging_rate / (60.0/timestep_minutes)) / self._battery.capacity
        discharge_rate = (self._battery.max_discharging_rate / (60.0/timestep_minutes)) / self._battery.capacity
        range = charge_rate - (-discharge_rate)
        action = action * range / 2 + (-discharge_rate + (0.5 * range))
        return action
    
    def step_dqn(self, action):
        # print(f'Step: {self._current_step}/{self._steps}')
        truncated = False
        terminated = self._current_step > self._steps - 1
        reward = 0
        info = {
            'energy_import': 0,
            'energy_export': 0
        }
        if terminated:
            # print("current step", self._current_step)
            # print("steps:", self._steps)
            return {}, reward, terminated, truncated, info


        observation = self._get_obs()
        consumption = observation[0+self._bio][0]
        generation = observation[1+self._bio][0]
        # apply the action
        # positive action: charge the battery
        # negative action: discharge the battery
        available_charge_capacity = (self._battery.max_soc - self._battery.soc)
        charge_rate = (self._battery.max_charging_rate / (60.0/timestep_minutes)) / self._battery.capacity
        available_charge_capacity = min(available_charge_capacity, charge_rate )

        available_discharge_capacity = (self._battery.soc - self._battery.min_soc)
        discharge_rate = (self._battery.max_discharging_rate / (60.0/timestep_minutes)) / self._battery.capacity
        available_discharge_capacity = min(available_discharge_capacity, discharge_rate)

    
        charge = 0.0
        penalty = 0.0
        if action == 0:
            # dont charge
            charge = 0.0
        # charge using solar
        elif action == 1:
            surplus_solar = max(generation - consumption, 0)
            charge = min(surplus_solar, available_charge_capacity)
            if self._battery.soc == self._battery.max_soc:
                penalty = abs(charge) * 10
        # charge using grid (charge full capacity)
        elif action == 2:
            charge = available_charge_capacity
            if self._battery.soc == self._battery.max_soc:
                penalty = abs(charge) * 10
        # # discharge and sell (discharge max capacity)
        elif action == 3:
            charge = -available_discharge_capacity
            if self._battery.soc == self._battery.min_soc:
                penalty = abs(charge) * 10
        # # discharge and power load (only discharge if generation is less than consumption)
        elif action == 4:
            surplus_demand = max(consumption - generation, 0)
            charge = -min(surplus_demand, available_discharge_capacity)
            if self._battery.soc == self._battery.min_soc:
                penalty = abs(charge) * 10

        # Update SOC (State of Charge)
        self._battery.soc += charge
        # clip soc
        self._battery.soc = np.clip(self._battery.soc, self._battery.min_soc, self._battery.max_soc)
        # update obs
        observation[6+self._bio][0] = self._battery.soc
        

        # observation = self._get_obs() 

        # calculate the reward
        # reward = -cost of importing power + income from selling power
        # consumption = observation[0][0]
        # generation = observation[1][0]
        energy_import = max(consumption + (charge * self._battery.capacity) - generation, 0)
        energy_export = max(generation - consumption - (charge * self._battery.capacity), 0)
        import_price = observation[2+self._bio][0]
        export_price = observation[3+self._bio][0]
        reward = energy_export * export_price - energy_import * import_price
        reward = reward - penalty

        info = {
            'energy_import': energy_import,
            'energy_export': energy_export
        }
        self._current_step += 1

        return observation, reward, terminated, truncated, info
    


def saveResults(df, out_fname):
    writer = pd.ExcelWriter(out_fname, engine="xlsxwriter")

    df = df.round(2)
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name="Sheet1", index=True)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]

    # Get the dimensions of the dataframe.
    (max_row, max_col) = df.shape

    for excel_col in range(1, max_col+1):
        # Apply a conditional format to the required cell range.
        # print(col, all_summaries.columns[col])
        df_col = excel_col - 1
        worksheet.conditional_format(
            1, excel_col, max_row, excel_col, 
            {
                "type": "data_bar",
                "bar_solid": True,
                "max_type": "num",
                "max_value": df.iloc[:, df_col].max() * 2,
                "min_type": "num",
                "min_value": min(0, df.iloc[:, df_col].min() * 2),
            })
        # print(f"Column {df_col}, {df.columns[df_col]}, min: {df.iloc[:, df_col].min()}, max: {df.iloc[:, df_col].max()}")
    worksheet.set_column(0, 0, 17.5)
    worksheet.set_column(1, max_col, 16)
    worksheet.freeze_panes(1, 0)  # Freeze the first row.
    worksheet.set_zoom(120)

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()



# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
