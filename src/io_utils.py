import numpy as np
import pandas as pd
import matplotlib as plt


def load_data() -> pd.DataFrame:
    df = pd.read_csv("..\\data\\health_study_dataset.csv")
    
    return df

df = load_data()