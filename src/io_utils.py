import pandas as pd



def load_data() -> pd.DataFrame:
    df = pd.read_csv("..\\data\\health_study_dataset.csv")
    
    return df

df = load_data()