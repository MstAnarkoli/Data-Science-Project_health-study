from io_utils import load_data
import pandas as pd
import numpy as np
from metrics import descriptive_stats,calculate_disease_proportion, hypothesis_testing, simulate_disease_proportion,confidence_inteval_bootstrap, confidence_inteval
from viz import plot_bp_histogram, plot_weight_by_gender, plot_smoker_proportion


def main():
    df = load_data()  

    print("\nRäkna ut medel, median, min och max för: age, weight, height, systolic_bp, cholesterol.\n")
    
    summary = descriptive_stats(df)
    print(f"{summary}\n\n")
    

    histogram = plot_bp_histogram(df)
    box_plot = plot_weight_by_gender(df)
    bar_diagram = plot_smoker_proportion(df)
    
    
    calculate_disease_proportion(df)
    
    #sim_prop = calculate_disease_proportion(df.to_dict('records'))
    #print(f"Calculated disease proportion: {sim_prop:.3f} ({sim_prop*100:.1f}%)")


    sim_prop_np = simulate_disease_proportion()
    print(f"Simulated disease proportion: {sim_prop_np:.3f} ({sim_prop_np*100:.1f}%)")

    confidence_inteval(df)
    confidence_inteval_bootstrap(df)
    hypothesis_testing(df)
    

if __name__ == '__main__':
    main()