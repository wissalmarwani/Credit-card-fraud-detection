import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import DATA_PATH, BASE_DIR
from data_loader import load_data

def run_eda(df):
    plots_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Distribution des classes
    plt.figure(figsize=(6,4))
    sns.countplot(x='Class', data=df)
    plt.title('Distribution des classes (0=normal, 1=fraude)')
    plt.savefig(os.path.join(plots_dir, "class_distribution.png"))
    plt.close()

    # 2. Distribution du montant selon fraude ou pas
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.title('Montant des transactions selon la classe')
    plt.savefig(os.path.join(plots_dir, "amount_by_class.png"))
    plt.close()

    # 3. Matrice de corrélation
    plt.figure(figsize=(14,12))
    sns.heatmap(df.corr(), cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation')
    plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"))
    plt.close()

    print(f"3 graphiques sauvegardés dans {plots_dir}")

if __name__ == "__main__":
    df = load_data()
    run_eda(df)