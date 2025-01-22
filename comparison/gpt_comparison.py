import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pearsonr, kendalltau

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Caricamento dei file JSON
json1_file = "datasets/pc_usr_avg.json"  # human
json2_file = "pc_gpt4.json"  # llm

json1 = load_json_file(json1_file)
json2 = load_json_file(json2_file)

# Estrazione dei punteggi
scores_json1 = [item['score'] for item in json1]
scores_json2 = [item['score'] for item in json2]

# Calcolo delle metriche di confronto
# Kappa di Cohen richiede valori categoriali, quindi discretizziamo i punteggi
categories_json1 = [round(score * 4) for score in scores_json1]
categories_json2 = [round(score * 4) for score in scores_json2]
kappa = cohen_kappa_score(categories_json1, categories_json2)

# Spearman, Pearson e Kendall-Tau
spearman_corr, spearman_p = spearmanr(scores_json1, scores_json2)
pearson_corr, pearson_p = pearsonr(scores_json1, scores_json2)
kendall_corr, kendall_p = kendalltau(scores_json1, scores_json2)

# Output dei risultati
print("Risultati del confronto:")
print(f"Kappa di Cohen: {kappa:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
print(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
print(f"Kendall-Tau Correlation: {kendall_corr:.4f} (p-value: {kendall_p:.4f})")
