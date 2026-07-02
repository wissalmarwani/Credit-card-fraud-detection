from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from data_loader import load_data

def preprocess_data(df):
    df = df.copy()

    # Scaling de Amount et Time
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    df = df.drop(['Amount', 'Time'], axis=1)

    # Séparation features / target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Split train/test stratifié (important à cause du déséquilibre)
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

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)