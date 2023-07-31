import os 
import pandas as pd
import logging


from src.preprocessing_module import Preprocessig_Toolbox
from src.filter_module import DataFilter

def load_dataset(DATAFRAME_PATH, DATAFRAME_SIZE, apply_filter=False):

    COLS = ['time', 'height', 'outflow',
        'pump1_rpm', 'pump1_power',
        'pump4_rpm', 'pump4_power']
    
    if apply_filter:
        logging.info("filtering the dataset...")
        processing = Preprocessig_Toolbox(dataframe_size = DATAFRAME_SIZE, dataframe_path= DATAFRAME_PATH)
        filter = DataFilter(forward_stability_condition = 3)
        df_filtered = filter.fit_transform(processing.data)
        logging.info("done!")
        return df_filtered
    else:
        logging.info("filter precedure skipped...")
        logging.info("loading raw dataset...")
        df = pd.read_pickle(DATAFRAME_PATH)
        df["height"] = df["height"]/100
        df["outflow"] = df["outflow"]/3600
        logging.info("raw dataset loaded")
        return df[:int(DATAFRAME_SIZE*len(df))][COLS]

def load_surrogate_model(SURROGATE_MODEL_PATH):
    with open(SURROGATE_MODEL_PATH + 'scaler_X_mpc_surrogate.pkl', 'rb') as f:
        scaler_X = pickle.load(f)

    with open(SURROGATE_MODEL_PATH + 'scaler_Y_mpc_surrogate.pkl', 'rb') as g:
        scaler_Y = pickle.load(g)

    model = tf.keras.models.load_model(SURROGATE_MODEL_PATH + 'mpc_surrogate.h5')
    logging.info("Models and scalers uploaded")
    return scaler_X, scaler_Y, model
