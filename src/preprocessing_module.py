import pandas as pd 
import numpy as np 
import logging

logging.getLogger().setLevel(logging.INFO)

class Preprocessig_Toolbox():

    """A toolbox for preprocessing data from a water pumping system.

    Parameters:
        dataframe_size (float): The fraction of the data to load from the pickle file.
        dataframe_path (str): The path to the pickle file containing the data.

    Attributes:
        dataframe_size (float): The fraction of the data to load from the pickle file.
        dataframe_path (str): The path to the pickle file containing the data.
        data (pandas.DataFrame): The preprocessed data.

    Methods:
        load_dataframe(): Load the data from the pickle file and filter it based on the specified size.
        get_speed_ref(): Calculate the reference speed of the water pumps based on the water level.
        add_pumps_status(): Add binary variables indicating whether each pump is running or not.
        transform_dataframe(): Apply all preprocessing steps to the data."""


    def __init__(self, dataframe_size:float, dataframe_path:str):

        """
        Initialize the Preprocessing_Toolbox object.

        Parameters:
            dataframe_size (float): The fraction of the data to load from the pickle file.
            dataframe_path (str): The path to the pickle file containing the data.
        """
                
        self.dataframe_size = dataframe_size
        self.dataframe_path = dataframe_path
        self.data = self.transform_dataframe()

    def load_dataframe(self):

        """
        Load the data from the pickle file and filter it based on the specified size.

        Returns:
            pandas.DataFrame: The filtered data.
        """

        self.COLS = ['height', 'outflow',
                'pump1_speed', 'pump1_rpm', 'pump1_power',
                'pump4_speed', 'pump4_rpm', 'pump4_power', 
                'overflow', "pump1_status", "pump4_status"]
        dataframe = pd.read_pickle(self.dataframe_path)
        N = len(dataframe)
        START_INDEX = 3459
        self.data = dataframe.iloc[START_INDEX:int(self.dataframe_size*N)]
        #self.data["outflow"] = self.data["outflow"]/3600
        #self.data["height"] = self.data["height"]/100
        return self.data.set_index("time")

    def get_speed_ref(self):
        
        """
        Calculate the reference speed of the water pumps based on the water level.

        Returns:
            pandas.DataFrame: The data with the "wref" column added.
        """
        
        self.data["wref"] = np.nan
        SL0 =  (self.data["height"] > 0) & (self.data["height"] <= 150)
        SL1 =  (self.data["height"] > 150) & (self.data["height"] <= 175)
        SL2 =  (self.data["height"] > 175) & (self.data["height"] <= 200)
        SL3 =  (self.data["height"] > 200) & (self.data["height"] <= 215)
        self.data.loc[SL0, "wref"] = 50
        self.data.loc[SL1, "wref"] = 0.4*self.data.loc[SL1, "height"] + 30
        self.data.loc[SL2, "wref"] = 1.4*self.data.loc[SL2, "height"] - 160
        #self.data.loc[SL3, "wref"] = 2.3*self.data.loc[SL0, "height"] - 401
        return self.data

    def add_pumps_status(self):
            
        """
        Add binary variables indicating whether each pump is running or not.

        Returns:
            pandas.DataFrame: The data with the "pump1_status" and "pump4_status" columns added.
        """
        STATUS_SLICE_P1 = self.data["pump1_rpm"] > 0
        STATUS_SLICE_P4 = self.data["pump4_rpm"] > 0
        self.data["pump1_status"] = 0
        self.data["pump4_status"] = 0
        self.data.loc[STATUS_SLICE_P1, "pump1_status"] = 1
        self.data.loc[STATUS_SLICE_P4, "pump4_status"] = 1
        
        return self.data[self.COLS]

    def transform_dataframe(self):
        """
        Apply all preprocessing steps to the data.

        Returns:
            pandas.DataFrame: The preprocessed data.
        """
        logging.info("loading data...")
        # Load the data from the pickle file and filter it based on the specified size
        self.data = self.load_dataframe()
        # Add binary variables indicating whether each pump is running or not
        self.data = self.add_pumps_status()
        # Calculate the reference speed of the water pumps based on the water level
        self.data = self.get_speed_ref()
        # Return the preprocessed data
        return self.data
