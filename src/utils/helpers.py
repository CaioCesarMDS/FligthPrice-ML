import pandas as pd
import numpy as np

def convert_duration(value):
    if pd.isna(value):
        return np.nan
    hours, minutes = 0, 0
    parts = value.strip().lower().split()
    for part in parts:
        if 'h' in part:
            hours = int(part.replace('h', ''))
        elif 'm' in part:
            minutes = int(part.replace('m', ''))
    return hours * 60 + minutes

def parse_arrival_time(value):
    try:
        return pd.to_datetime(value, format="%H:%M %d %b")
    except:
        try:
            return pd.to_datetime(value, format="%H:%M")
        except:
            return pd.NaT
