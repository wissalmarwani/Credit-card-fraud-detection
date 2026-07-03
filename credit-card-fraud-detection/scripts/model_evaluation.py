import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from config import BASE_DIR

def evaluate_hybrid_model(y_test, y_pred, y_proba):
    """
    Évalue le modèle hybride, affiche le rapport de classification
    et génère une figure contenant la courbe ROC et la courbe Précision-Rappel.
    """
    plots_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Calcul des métriques
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\n=== EVALUATION DU MODELE HYBRIDE (XGBoost + Isolation Forest) ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))
    print(f"Précision           : {precision:.4f}")
    print(f"Rappel (Recall)     : {recall:.4f}")
    print(f"F1-Score            : {f1:.4f}")
    print(f"ROC-AUC             : {roc_auc:.4f}")
    print(f"PR-AUC (AUPRC)      : {pr_auc:.4f}")

    # Tracé des courbes ROC et Precision-Recall côte-à-côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, color='royalblue', lw=2, label=f"Hybride (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], 'k--', label="Aléatoire (AUC = 0.5000)")
    ax1.set_xlabel("Taux de Faux Positifs (FPR)")
    ax1.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax1.set_title("Courbe ROC (Receiver Operating Characteristic)")
    ax1.legend(loc="lower right")

    # 2. Courbe Precision-Recall
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ax2.plot(rec, prec, color='forestgreen', lw=2, label=f"Hybride (AUPRC = {pr_auc:.4f})")
    ax2.set_xlabel("Rappel (Recall)")
    ax2.set_ylabel("Précision (Precision)")
    ax2.set_title("Courbe Précision-Rappel (PR)")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    output_path = os.path.join(plots_dir, "hybrid_model_curves.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nGraphique combiné sauvegardé sous : {output_path}")
