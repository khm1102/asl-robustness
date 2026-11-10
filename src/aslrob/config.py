from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_PARQUET = DATA / "hf_v03.parquet"
PROC = DATA / "processed"
RESULTS = ROOT / "results"

# decisions/0003: 정적 + 혼동쌍(U/V) 포함 10개
LETTERS = list("ABDFILUVWY")
ALL_LETTERS = [chr(ord("A") + i) for i in range(26)]
SIZE_CNN = 96
SIZE_MP = 256
