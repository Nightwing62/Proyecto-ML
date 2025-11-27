import pandas as pd
from pathlib import Path

#RUTA BASE
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

print("Cargando tablas parquet...")

app  = pd.read_parquet(DATA / "application_.parquet")
bureau  = pd.read_parquet(DATA / "bureau.parquet")
bb      = pd.read_parquet(DATA / "bureau_balance.parquet")
prev    = pd.read_parquet(DATA / "previous_application.parquet")
ccb     = pd.read_parquet(DATA / "credit_card_balance.parquet")
pos     = pd.read_parquet(DATA / "POS_CASH_balance.parquet")
inst    = pd.read_parquet(DATA / "installments_payments.parquet")

def agg_numeric(df, key):
    """Agrupa solo columnas numéricas por la llave dada."""
    num_df = df.select_dtypes(include=["number"])
    return num_df.groupby(df[key]).agg(["mean", "max", "min", "count"])

print("Agregando features de bureau...")
bureau_agg = agg_numeric(bureau, "SK_ID_CURR")
bureau_agg.columns = ["BUREAU_" + "_".join(col).upper() for col in bureau_agg.columns]

print("Agregando bureau_balance...")
bb_agg = agg_numeric(bb, "SK_ID_BUREAU")
bb_agg.columns = ["BB_" + "_".join(col).upper() for col in bb_agg.columns]

bureau = bureau.set_index("SK_ID_BUREAU").join(bb_agg, how="left").reset_index()
bureau2 = agg_numeric(bureau, "SK_ID_CURR")
bureau2.columns = ["BUREAU2_" + "_".join(col).upper() for col in bureau2.columns]

print("Agregando previous_application...")
prev_agg = agg_numeric(prev, "SK_ID_CURR")
prev_agg.columns = ["PREV_" + "_".join(col).upper() for col in prev_agg.columns]

print("Agregando POS_CASH...")
pos_agg = agg_numeric(pos, "SK_ID_CURR")
pos_agg.columns = ["POS_" + "_".join(col).upper() for col in pos_agg.columns]


print("Agregando credit_card_balance...")
ccb_agg = agg_numeric(ccb, "SK_ID_CURR")
ccb_agg.columns = ["CCB_" + "_".join(col).upper() for col in ccb_agg.columns]

print("Agregando installments...")
inst_agg = agg_numeric(inst, "SK_ID_CURR")
inst_agg.columns = ["INST_" + "_".join(col).upper() for col in inst_agg.columns]

print("Uniendo todo con application...")

df = app.set_index("SK_ID_CURR") \
        .join(bureau_agg, how="left") \
        .join(bureau2,   how="left") \
        .join(prev_agg,  how="left") \
        .join(pos_agg,   how="left") \
        .join(ccb_agg,   how="left") \
        .join(inst_agg,  how="left") \
        .reset_index()

print("Guardando dataset final...")
df.to_parquet(ART / "train_merged.parquet")

print("✔ Listo. Archivo creado en /artifacts/train_merged.parquet")
