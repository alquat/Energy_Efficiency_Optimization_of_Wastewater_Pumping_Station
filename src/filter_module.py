import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

class DataFilter():
    
    def __init__(self, forward_stability_condition: int = 3):
        self.forward_stability_condition = forward_stability_condition
    
    def wref_stability(self, df):
        df["same_wref_flag"] = df['wref'].eq(df['wref'].shift(1))
        df['temp'] = (~df['same_wref_flag']).cumsum()
        return df.groupby('temp')
    
    def __create_empty_dataset(self):
        data = {
            "time": [],
            "height": [],
            "inflow": [],
            "outflow": [],
            "pump1_rpm": [],
            "pump1_power": [],
            "pump4_rpm": [],
            "pump4_power": [],
            "wref": [],
            "time_delta_sec": []}
        return data


    def fit_transform(self, df: pd.DataFrame):
        grouped = self.wref_stability(df)
        filtered_df = self.__create_empty_dataset() 

        for group_id in tqdm(range(1, len(grouped) + 1)):
            g = grouped.get_group(group_id)

            if len(g) > self.forward_stability_condition:
                filtered_df['time'].append(g.index[0])
                filtered_df['time_delta_sec'].append((g.index[-1] - g.index[0]).total_seconds())
                filtered_df['height'].append(np.median(g['height']))
                filtered_df['outflow'].append(np.median(g['outflow']))
                filtered_df['pump1_power'].append(np.median(g['pump1_power'] * g['pump1_status']))
                filtered_df['pump1_rpm'].append(np.median(g['pump1_rpm'] * g['pump1_status']))
                filtered_df['pump4_power'].append(np.median(g['pump4_power'] * g['pump4_status']))
                filtered_df['pump4_rpm'].append(np.median(g['pump4_rpm'] * g['pump4_status']))
                filtered_df['wref'].append(np.median(g['wref']))
                #filtered_df['inflow'].append(np.median(g['inflow_est']))

        df_filtered = pd.DataFrame(filtered_df)

        print(f'The compression factor of the filter is: {len(df)/len(df_filtered):.2f}')


        return df_filtered

