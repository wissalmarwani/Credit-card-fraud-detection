import pandas as pd
from config import DATA_PATH
 
def load_data():
    """Charge le dataset de transactions bancaires"""
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Nombre de fraudes : {df['Class'].sum()}")
    print(f"Nombre de transactions normales : {(df['Class']==0).sum()}")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
