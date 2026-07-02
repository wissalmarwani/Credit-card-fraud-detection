from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from data_loader import load_data
from data_preprocessing import preprocess_data
from config import RANDOM_STATE

def run_isolation_forest(X_train, X_test, y_test):
    # contamination = proportion attendue de fraudes dans les données
    contamination_rate = 0.0017

    iso_forest = IsolationForest(
        contamination=contamination_rate,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso_forest.fit(X_train)

    # -1 = anomalie détectée, 1 = normal
    raw_predictions = iso_forest.predict(X_test)
    # on convertit en 0/1 pour comparer avec y_test (1 = fraude)
    y_pred = [1 if p == -1 else 0 for p in raw_predictions]

    # score d'anomalie (plus c'est bas, plus c'est anormal)
    anomaly_scores = iso_forest.decision_function(X_test)

    print("=== Résultats Isolation Forest (non supervisé) ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))

    return iso_forest, anomaly_scores, y_pred

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model, scores, preds = run_isolation_forest(X_train, X_test, y_test)