import pandas as pd
from pathlib import Path
import gc

# RUTAS
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

print("Cargando tablas parquet...")

app  = pd.read_parquet(DATA / "application_.parquet")
bureau = pd.read_parquet(DATA / "bureau.parquet")
bb = pd.read_parquet(DATA / "bureau_balance.parquet")
prev = pd.read_parquet(DATA / "previous_application.parquet")
pos = pd.read_parquet(DATA / "POS_CASH_balance.parquet")
ccb = pd.read_parquet(DATA / "credit_card_balance.parquet")
inst = pd.read_parquet(DATA / "installments_payments.parquet")


# FUNCIONES 

def agg_numeric(df, key, stats=("mean",)):

    num_df = df.select_dtypes(include="number")
    return num_df.groupby(df[key]).agg(stats)


# fUNCION EXPLICITA PARA MANEJAR EL USO DE MEMORIA (LIGHTGBM NO NECESITA FLOAT64).
def reduce_memory(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df



# FEATURE ENGINEERING


print("Agregando features de bureau...")
bureau_agg = agg_numeric(bureau, "SK_ID_CURR", stats=("mean", "max"))
bureau_agg.columns = ["BUREAU_" + "_".join(col).upper() for col in bureau_agg.columns]
bureau_agg = reduce_memory(bureau_agg)

print("Agregando bureau_balance...")
bb_agg = agg_numeric(bb, "SK_ID_BUREAU", stats=("mean", "count"))
bb_agg.columns = ["BB_" + "_".join(col).upper() for col in bb_agg.columns]
bb_agg = reduce_memory(bb_agg)

print("Uniendo bureau + bureau_balance...")
bureau = bureau.set_index("SK_ID_BUREAU").join(bb_agg, how="left")
bureau2 = agg_numeric(bureau, "SK_ID_CURR", stats=("mean",))
bureau2.columns = ["BUREAU2_" + "_".join(col).upper() for col in bureau2.columns]
bureau2 = reduce_memory(bureau2)

del bureau, bb, bb_agg
gc.collect()

print("Agregando previous_application...")
prev_agg = agg_numeric(prev, "SK_ID_CURR", stats=("mean",))
prev_agg.columns = ["PREV_" + "_".join(col).upper() for col in prev_agg.columns]
prev_agg = reduce_memory(prev_agg)

del prev
gc.collect()

print("Agregando POS_CASH...")
pos_agg = agg_numeric(pos, "SK_ID_CURR", stats=("mean",))
pos_agg.columns = ["POS_" + "_".join(col).upper() for col in pos_agg.columns]
pos_agg = reduce_memory(pos_agg)

del pos
gc.collect()

print("Agregando credit_card_balance...")
ccb_agg = agg_numeric(ccb, "SK_ID_CURR", stats=("mean",))
ccb_agg.columns = ["CCB_" + "_".join(col).upper() for col in ccb_agg.columns]
ccb_agg = reduce_memory(ccb_agg)

del ccb
gc.collect()

print("Agregando installments_payments...")
inst_agg = agg_numeric(inst, "SK_ID_CURR", stats=("mean", "sum"))
inst_agg.columns = ["INST_" + "_".join(col).upper() for col in inst_agg.columns]
inst_agg = reduce_memory(inst_agg)

del inst
gc.collect()


# MERGE FINAL


print("Uniendo todo con application...")

df = app.set_index("SK_ID_CURR")

for agg_df in [
    bureau_agg,
    bureau2,
    prev_agg,
    pos_agg,
    ccb_agg,
    inst_agg,
]:
    df = df.join(agg_df, how="left")

df = reduce_memory(df).reset_index()

print("Shape final:", df.shape)

# GUARDADO

print("Guardando dataset final...")
df.to_parquet(ART / "train_merged.parquet")

print("Listo. Archivo creado en /artifacts/train_merged.parquet")

print("FASE 2: Terminada")
