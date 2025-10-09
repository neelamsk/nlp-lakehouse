import os
from typing import Optional
from pyspark.sql import DataFrame, SparkSession

LAKE_ACCOUNT = os.getenv("LAKE_ACCOUNT", "nlplakeadls001")
CONTAINER = os.getenv("LAKE_CONTAINER", "nlp")

def abfss_path(layer: str, relpath: str = "") -> str:
    base = f"abfss://{CONTAINER}@{LAKE_ACCOUNT}.dfs.core.windows.net/{layer}"
    return f"{base.rstrip('/')}/{relpath.lstrip('/')}" if relpath else base

def read_delta(spark: SparkSession, layer: str, relpath: str) -> DataFrame:
    return spark.read.format("delta").load(abfss_path(layer, relpath))

def write_delta(df: DataFrame, layer: str, relpath: str, mode: str = "append", partitionBy: Optional[str] = None) -> None:
    writer = df.write.format("delta").mode(mode)
    if partitionBy:
        writer = writer.partitionBy(partitionBy)
    writer.save(abfss_path(layer, relpath))
