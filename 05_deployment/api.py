from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict


# CONFIG
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

MODEL_PATH = ART / "model_lgbm.joblib"
FEATURES_PATH = ART / "model_features.txt"


# CARGA DEL MODELO
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    MODEL_FEATURES = [line.strip() for line in f.readlines()]


# API
app = FastAPI(
    title="Credit Default Risk API",
    description="API para evaluar riesgo de incumplimiento crediticio",
    version="1.1"
)


# INPUT
class CreditApplication(BaseModel):
    data: Dict[str, float]


# ENDPOINT
@app.post("/evaluate_risk")
def evaluate_risk(payload: CreditApplication):
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([payload.data])

        # Asegurar mismas features que el modelo
        df = df.reindex(columns=MODEL_FEATURES, fill_value=0)

        # Predict
        proba = model.predict(df)[0]

        # Decision de negocio (10-29% acepta, 30-60% pone en duda, mas del 60 rechaza)
        if proba < 0.3:
            decision = "APROBAR"
        elif proba < 0.6:
            decision = "REVISIÓN MANUAL"
        else:
            decision = "RECHAZAR"

        return {
            "probabilidad_default": round(float(proba), 4),
            "decision": decision
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
print("Fase 5: completada")
