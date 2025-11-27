from io_utils import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from metrics import regression_bp_age
#from viz import 


class HealthAnalyzer:
    def __init__(self):
        
        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        """Läser in datasetet från CSV-filen och returnerar en DataFrame."""        
        df = pd.read_csv("..\\data\\health_study_dataset.csv")
        return df
    
    def descriptive_stats(self) -> pd.DataFrame:
        """
        Räkna ut medel, median, min och max för: age, weight, height, systolic_bp, cholesterol.
        """
        columns = ['age', 'weight', 'height', 'systolic_bp', 'cholesterol']
        
        stats = self.df[columns].agg(['mean', 'median', 'min', 'max']).transpose()
        stats = stats.rename(columns={'mean': 'Mean', 'median': 'Median', 'min': 'Min', 'max': 'Max'})
        
        return stats.round(2)

 

    def average_blood_pressure(self):
        """Beräknar medelvärdet av systolic_bp."""
        
        systolic_bp = self.df['systolic_bp'].dropna()
        systolyc_bp_mean =  np.mean(systolic_bp)
        return systolyc_bp_mean
            

    def plot_age_vs_bp(self):

        age = self.df['age'].dropna()
        systolic_bp = self.df['systolic_bp'].dropna()
        """Ritar scatter-plot mellan ålder och blodtryck."""
        plt.scatter(age, systolic_bp)
        plt.xlabel("Ålder")
        plt.ylabel("Systoliskt blodtryck")
        plt.title("Ålder vs Systoliskt blodtryck")
        plt.show()

    
    def avg_weight_by_gender(self):
        avg_weights = self.df.groupby("sex")["weight"].mean()
        return avg_weights.round(2)

    def plot_avg_weight_by_gender(self):
        avg_weights = self.df.groupby("sex")["weight"].mean()
        plt.bar(avg_weights.index, avg_weights.values)
        plt.xlabel("Kön")
        plt.ylabel("Genomsnittlig vikt (kg)")
        plt.title("Genomsnittlig vikt per kön")
        plt.show()
       

    def regression_bp_age(self):
        X = self.df[['age']].values        # Oberoende variabel (ålder)
        y = self.df['systolic_bp'].values  # Beroende variabel (blodtryck)

        model = LinearRegression()
        model.fit(X, y)

        # Prediktion
        self.df['pred_bp'] = model.predict(X)

        # Visa resultat
        print("Intercept:", round(model.intercept_, 2))
        print("Slope:", round(model.coef_[0], 2))
        print("\nModel: BP = age * slope + intercept")

        # Plot
        plt.figure(figsize=(8,5))
        plt.scatter(self.df['age'], self.df['systolic_bp'], label='Data')
        plt.plot(self.df['age'], self.df['pred_bp'], linewidth=2, label='Regression line')
        plt.xlabel("Ålder")
        plt.ylabel("Systoliskt blodtryck")
        plt.title("Linjär regression: blodtryck baserat på ålder")
        plt.legend()
        plt.show()

    



