import pandas as pd
import numpy as np
import random
from scipy import stats



def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Räkna ut medel, median, min och max för: age, weight, height, systolic_bp, cholesterol.
    """
    columns = ['age', 'weight', 'height', 'systolic_bp', 'cholesterol']
    
    stats = df[columns].agg(['mean', 'median', 'min', 'max']).transpose()
    stats = stats.rename(columns={'mean': 'Mean', 'median': 'Median', 'min': 'Min', 'max': 'Max'})
    
    return stats


def calculate_disease_proportion(df: pd.DataFrame) -> pd.DataFrame:
    # personer  som har sjukdomen
    has_disease = df[df["disease"] == 1]

    # personer  som har inte sjukdomen
    no_disease = df[df["disease"] == 0]

    proportion = df["disease"].mean()      
    print(f"Andelen personer  som har sjukdomen: {proportion:.3f} ({proportion*100:.1f}%)")  
          
    
          

def simulate_disease_proportion(seed=42, n_people=1000, disease_prob=0.059):
    np.random.seed(seed)
    
    # Simulera sjukdomsstatus: 1 = har sjukdomen, 0 = har inte sjukdomen
    disease_status = np.random.binomial(n=1, p=disease_prob, size=n_people)
    
    # Beräkna andelen personer med sjukdomen
    proportion_with_disease = np.mean(disease_status)
    
    return proportion_with_disease

def confidence_inteval(df: pd.DataFrame) -> pd.DataFrame:
    
    sys_bp = df['systolic_bp']   
    n = len(sys_bp)
    mean_sys_bp = sys_bp.mean()
    std_sys_bp = sys_bp.std(ddof=1)  
    se = std_sys_bp / np.sqrt(n) 

    # z-värde för 95% confidence interval
    z = 1.96

    # Confidence interval med normal approximation

    lower_bound = mean_sys_bp - z * se
    upper_bound = mean_sys_bp + z * se

    
    print("\n\nConfidence interval for the mean value of systolic_bp with normal approximation (95% CI)\n")
    print(f"Mean systolic_bp: {mean_sys_bp:.2f}")
    print(f"Standard deviation: {std_sys_bp:.2f}")
    print(f"Standard error: {se:.2f}\n\n")
    print(f"95% CI: ({lower_bound:.2f}, {upper_bound:.2f})\n\n")

def confidence_inteval_bootstrap(df: pd.DataFrame, confidence = 0.95) -> pd.DataFrame:
    sys_bp = df['systolic_bp']   
    n = len(sys_bp)
    true_mean = sys_bp.mean()

    n_bootstrap = 1000
    bootstrap_means = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        bootstrap_sample = np.random.choice(sys_bp, size=n, replace=True)
        bootstrap_means[b] = np.mean(bootstrap_sample)

    alpha = (1 - confidence)/2

    lower_bound_bootstrap = np.percentile(bootstrap_means, 100*alpha)
    upper_bound_bootstrap= np.percentile(bootstrap_means, 100*(1 - alpha))
    b_means = np.mean(bootstrap_means)

    
    print("\n\nConfidence interval för medelvärdet av systoliskt_bp med bootstrap (95% CI)\n")
    print(f"True Mean systolic_bp: {true_mean:.2f}\n\n")
    print(f"Bootstrap Mean systolic_bp: {b_means:.2f}\n\n")
    print(f"95% CI: ({lower_bound_bootstrap:.2f}, {upper_bound_bootstrap:.2f})\n\n")

def hypothesis_testing(df: pd.DataFrame,) -> pd.DataFrame:

    #Hypotesprövning med bootstrap:
    print("\n\nTesta hypotesen: ”Rökare har högre medel-blodtryck än icke-rökare.\n")
    print("Null hypotesen = Rökare har högre medel-blodtryck än icke-rökare.\n")
    print("\n\nAlternative hypotesen: Rökare har lika eller lägre medel-blodtryck än icke-rökare.\n")

    smokers = df[df["smoker"] == "Yes"]["systolic_bp"].dropna().values
    nonsmokers = df[df["smoker"] == "No"]["systolic_bp"].dropna().values

    obs_diff = smokers.mean() - nonsmokers.mean()
    combined = np.concatenate([smokers, nonsmokers])

    n_bootstrap = 10000
    bootstrap_diffs = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        np.random.shuffle(combined)
        boot_smokers = combined[:len(smokers)]
        boot_nonsmokers = combined[len(smokers):]
        bootstrap_diffs[b] = (boot_smokers.mean() - boot_nonsmokers.mean())

    p_value_boot = np.mean(bootstrap_diffs >= obs_diff)

    print("Medel systoliskt blodtryck för rökare: {:.2f}".format(smokers.mean()))
    print("Medel systoliskt blodtryck för icke-rökare: {:.2f}".format(nonsmokers.mean()))
    print("Medel systoliskt blodtryck för rökare med bootstrap: {:.2f}".format(boot_smokers.mean()))
    print("Medel systoliskt blodtryck för icke-rökare med bootstrap: {:.2f}".format(boot_nonsmokers.mean()))
    print(f"Observerad skillnad: {obs_diff:.4f}")
    print(f"P-value med bootstrap: {p_value_boot:.4f}")
   
    
    
