import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from data_loader import load_data
from data_preprocessing import preprocess_data
from config import RANDOM_STATE

def train_xgboost(X_train, y_train, X_test, y_test):
    # ratio pour compenser le déséquilibre des classes
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='aucpr',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Résultats XGBoost (supervisé) ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

    return model, y_pred, y_proba

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model, y_pred, y_proba = train_xgboost(X_train, y_train, X_test, y_test)