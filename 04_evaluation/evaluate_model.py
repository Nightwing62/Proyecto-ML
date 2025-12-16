import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

print("Cargando dataset y modelo...")

df = pd.read_parquet(ART / "train_merged.parquet")
model = joblib.load(ART / "model_lgbm.joblib")

test_idx = joblib.load(ART / "test_idx.joblib")
y_test = joblib.load(ART / "y_test.joblib")

df_test = df.loc[test_idx]

num_cols = df_test.select_dtypes(include=["number", "bool"]).columns.drop("TARGET")
X_test = df_test[num_cols].fillna(0)

print("Shape test:", X_test.shape)

# =========================
# PREDICCIONES
# =========================
y_proba = model.predict(X_test)
y_pred = (y_proba >= 0.5).astype(int)

# =========================
# MÉTRICAS FINALES
# =========================
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"AUC TEST: {auc:.4f}")
print("\nMatriz de confusión:\n", cm)
print("\nReporte:\n", report)

print("✔ FASE 4 completada")
