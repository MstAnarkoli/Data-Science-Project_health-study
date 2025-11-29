from io_utils import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
            

    def plot_cholesterol_vs_bp(self):

        cholesterol = self.df['cholesterol'].dropna()
        systolic_bp = self.df['systolic_bp'].dropna()
        """Ritar scatter-plot mellan ålder och blodtryck."""
        plt.scatter(cholesterol, systolic_bp)
        plt.xlabel("Cholesterol")
        plt.ylabel("Systoliskt blodtryck")
        plt.title("Cholesterol vs Systoliskt blodtryck")
        plt.show()

    
    def avg_weight_by_gender(self):
        avg_weights = self.df.groupby("sex")["weight"].mean()
        return avg_weights.round(2)

    def plot_avg_weight_by_gender(self):
        avg_weights = self.df.groupby("sex")["weight"].mean()
        plt.bar(avg_weights.index, avg_weights.values, color=['pink', 'lightblue'], width = 0.5)
        plt.xlabel("Kön")
        plt.ylabel("Genomsnittlig vikt (kg)")
        plt.title("Genomsnittlig vikt per kön")
        plt.show()
       
 
    def regression_bp_age(self):
        X = self.df[['age']].to_numpy()        # Oberoende variabel (ålder)
        y = self.df['systolic_bp'].to_numpy()  # Beroende variabel (blodtryck)

        model = LinearRegression()
        model.fit(X, y)
        

        # Prediktion
        y_hat = model.predict(X)
        

        # Visa resultat
        print("Intercept:", round(model.intercept_, 2))
        print("Slope:", round(model.coef_[0], 2))
        print("\nModel: BP = age * slope + intercept")

        # Plot
        plt.figure(figsize=(8,5))
        plt.scatter(self.df['age'], self.df['systolic_bp'], label='Data')
        plt.plot(self.df['age'], y_hat, linewidth=1, color='black', label='Regression line')
        plt.xlabel("Ålder")
        plt.ylabel("Systoliskt blodtryck")
        plt.title("Linjär regression: blodtryck baserat på ålder")
        plt.legend()
        plt.show()



        # Residual model
        
        residuals = y - y_hat
        self.df['residuals'] = residuals

       #Residual plot       
        plt.figure(figsize=(8,5))
        plt.scatter(y_hat, residuals, label='Residuals')
        plt.axhline(0, color='black', linewidth=2)   # horizontal reference line
        plt.xlabel("Förutsagt blodtryck")
        plt.ylabel("Residual (y - y_hat)")
        plt.title("Residualer vs. förutsagt värde")
        plt.legend()
        plt.show()


    def pca_model(self):
        self.df['sex'] = self.df['sex'].map({'F': 0, 'M': 1})
        self.df['smoker'] = self.df['smoker'].map({'No': 0, 'Yes': 1})

        # Välj numeriska kolumner för PCA
        features = ['age', 'sex', 'height', 'weight', 'systolic_bp', 'cholesterol', 'smoker']
        X = self.df[features]

        # Standardisera
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA (2 komponenter för visualisering)
        pca2 = PCA(n_components=2)
        principal_components = pca2.fit_transform(X_scaled)

        # Create a DataFrame for plotting
        pca2_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca2_df['disease'] = self.df['disease']  # för färgläggning

        # Scatter plot att visualisera kluster/mönster
        plt.figure(figsize=(8,6))
        colors = ['blue' if d == 0 else 'red' for d in pca2_df['disease']]
        plt.scatter(pca2_df['PC1'], pca2_df['PC2'], c=colors)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA - De två första komponenterna (färgade efter sjukdom)')
        # Lägg till Legend manuellt
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Frisk',
                markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Sjuk',
                markerfacecolor='red', markersize=10),
        ]

        plt.legend(handles=legend_elements, loc='upper right')

        plt.show()

        # PCA:3 komponenter för 3D-plot
        pca3 = PCA(n_components=3)
        pca3_result = pca3.fit_transform(X_scaled)

        # Skapa PCA dataframe
        pca3_df = pd.DataFrame({
            'PC1': pca3_result[:,0],
            'PC2': pca3_result[:,1],
            'PC3': pca3_result[:,2],
            'disease': self.df['disease']
        })

        # 3D scatter plot
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        # Färger för sjukdomar
        colors = ['blue' if d==0 else 'red' for d in pca3_df['disease']]

        ax.scatter(
            pca3_df['PC1'], 
            pca3_df['PC2'], 
            pca3_df['PC3'], 
            c=colors, 
            s=50,
            label=None
        )

        # Lägg till Legend manuellt
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Frisk',
                markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Sjuk',
                markerfacecolor='red', markersize=10),
        ]

        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("3D PCA Plot (Färgad av sjukdom))")

        plt.show()


    def corr_heatmap(self, columns):
        C = self.df[columns].corr()
            
        plt.figure(figsize=(6,5))
        plt.imshow(C, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar(fraction=0.05, pad=0.05)
        plt.title('Korrelationsmatris')
        plt.xticks(range(len(columns)), columns, rotation=45, ha='right')
        plt.yticks(range(len(columns)), columns)
        plt.tight_layout(); 
        plt.show()
            

    def  disease_gender(self):
        df = self.df.copy()

        # Visa sjukdom som numerisk (0/1)
        df["disease"] = df["disease"].astype(int)
        df["sex"] = df["sex"].map({0: "F", 1: "M"})

        # Sjukdomsincidens per kön
        incidence = df.groupby("sex")["disease"].mean()
        print("Sjukdomsincidens (%):")
        print((incidence * 100).round(2))

        # Antal sjukdomar per kön:
        counts = df.groupby(["sex", "disease"]).size().unstack(fill_value=0)
        counts = counts.rename(columns={0: "Frisk", 1: "Sjuk"})
        print("\nAntal sjukdomar per kön::")
        print(counts)

        # Visualisera resultatet
        import matplotlib.pyplot as plt

        counts.plot(kind="bar", stacked=True)
        plt.title("Andel sjukdomar per kön")
        plt.xlabel("Kön")
        plt.ylabel("Antal personer")
        plt.legend(["Frisk", "Sjuk"])
        plt.show()

        # Chi-square test (kontrollerar om kön och sjukdom är relaterade)
        from scipy.stats import chi2_contingency

        chi2, p, dof, expected = chi2_contingency(counts)
        print("\nChi-square test:")
        print("Chi2 =", chi2)
        print("p-value =", p)


            



            



