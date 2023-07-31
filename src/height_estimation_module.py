import pandas as pd
import numpy as np

class EMAInflowOutflowModel:

    def __init__(self, alpha=0.2, area=20):
        self.alpha = alpha
        self.area = area

    @staticmethod
    def exponential_moving_average(data, alpha):
        ema = [data[0]]
        for i in range(1, len(data)):
            ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
        return np.array(ema)

    def fit(self, df):
        smoothed_inflow_ema = self.exponential_moving_average(df["inflow"].dropna().values, self.alpha)
        # Change the concatenation here:
        smoothed_inflow_ema = np.concatenate(([None] * (len(df) - len(smoothed_inflow_ema)), smoothed_inflow_ema))
        df["smoothed_inflow_ema"] = smoothed_inflow_ema
        return df


    def simulate_height(self, df):
        simulated_heights_ema = []
        
        for i in range(len(df)):
            if i == 0 or df.loc[i, "smoothed_inflow_ema"] is None:
                simulated_heights_ema.append(df.loc[i, "height"])
                continue
            
            inflow = df.loc[i, "smoothed_inflow_ema"]
            outflow = df.loc[i, "outflow"]
            time_diff = (df.loc[i, "time"] - df.loc[i - 1, "time"])
            time_diff_seconds = time_diff.total_seconds()
            
            net_volume_change = (inflow - outflow) * time_diff_seconds
            height_change = net_volume_change / self.area
            new_height = simulated_heights_ema[-1] + height_change
            simulated_heights_ema.append(new_height)

        df["simulated_height_ema"] = simulated_heights_ema
        return df

