import pandas as pd
import numpy as np
from tqdm import tqdm 

class InflowOutflowModel:

    def __init__(self, window_size=5, area=20):
        self.window_size = window_size
        self.area = area

    @staticmethod
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    @staticmethod
    def calculate_inflow(height_diff, outflow, time_diff_seconds, area=20):
        volume_variation = area * height_diff
        net_volume_change = volume_variation + outflow * time_diff_seconds
        inflow_rate = net_volume_change / time_diff_seconds
        return inflow_rate

    def fit(self, df):
        inflow_rates = []
        for i in tqdm(range(1, len(df))):
            height_diff = df.loc[i, "height"] - df.loc[i - 1, "height"]
            outflow = df.loc[i, "outflow"]
            time_diff = (df.loc[i, "time"] - df.loc[i - 1, "time"])
            time_diff_seconds = time_diff.total_seconds()

            inflow_rate = self.calculate_inflow(height_diff, outflow, time_diff_seconds, self.area)
            inflow_rates.append(inflow_rate)

        inflow_rates = [None] + inflow_rates
        df["inflow"] = inflow_rates

        smoothed_inflow = self.moving_average(df["inflow"].dropna().values, self.window_size)
        smoothed_inflow = np.concatenate(([None] * (self.window_size), smoothed_inflow))

        df["smoothed_inflow"] = smoothed_inflow
        return df[self.window_size:].reset_index(drop=True)
