from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from data_loader import load_data

def preprocess_data(df):
    """
    Normalise séparément Amount et Time avec deux StandardScaler,
    puis effectue un partitionnement de données d'entraînement et test stratifié.
    """
    df = df.copy()

    # Scaling de Amount et Time (Correction du bug de réécriture de l'état du scaler)
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler_time.fit_transform(df[['Time']])
    
    df = df.drop(['Amount', 'Time'], axis=1)

    # Séparation features / target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Split train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train set : {X_train.shape[0]} lignes")
    print(f"Test set : {X_test.shape[0]} lignes")
    print(f"Fraudes dans train : {y_train.sum()}")
    print(f"Fraudes dans test : {y_test.sum()}")

    return X_train, X_test, y_train, y_test, scaler_amount, scaler_time

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler_amount, scaler_time = preprocess_data(df)