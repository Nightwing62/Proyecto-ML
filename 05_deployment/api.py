from fastapi import FastAPI
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

model = joblib.load(ART / "model_lgbm.joblib")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = df.fillna(0)
    proba = model.predict(df)[0]
    return {"probalidad_default": float(proba)}