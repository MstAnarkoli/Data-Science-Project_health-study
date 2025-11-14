import matplotlib.pyplot as plt
import pandas as pd

def plot_bp_histogram(df: pd.DataFrame):
    """Histogram över systolic blodtryck."""
    plt.figure(figsize=(8,5))
    plt.hist(df['systolic_bp'], bins=20, color='skyblue', edgecolor='black')
    plt.title("Histogram över blodtryck")
    plt.xlabel("Systolic BP")
    plt.ylabel("Count")
    plt.show()

def plot_weight_by_gender(df: pd.DataFrame):
    """Boxplot över vikt per kön."""
    plt.figure(figsize=(8,5))
    
    # Separate weights by gender
    males = df[df['sex']=='M']['weight']
    females = df[df['sex']=='F']['weight']
    
    plt.boxplot([males, females], labels=['M', 'F'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red'))
    
    plt.title("Boxplot över vikt per kön")
    plt.xlabel("Gender")
    plt.ylabel("Weight (kg)")
    plt.show()

def plot_smoker_proportion(df: pd.DataFrame):
    """Stapeldiagram över andelen rökare."""
    smoker_counts = df['smoker'].value_counts(normalize=True) * 100
    
    plt.figure(figsize=(6,5))
    plt.bar(smoker_counts.index, smoker_counts.values,width=0.3,edgecolor='black', color=['lightgreen', 'lightcoral'])
    plt.title("stapeldiagram över andelen rökare(%)")
    plt.ylabel("Percentage")
    plt.show()
