from pyspark.sql import functions as F

def basic_clean(df, input_col: str = "text", output_col: str = "text_clean"):
    cleaned = (df
        .withColumn(output_col, F.lower(F.col(input_col)))
        .withColumn(output_col, F.regexp_replace(output_col, r"[^\w\s]", " "))
        .withColumn(output_col, F.regexp_replace(output_col, r"\s+", " "))
        .withColumn(output_col, F.trim(F.col(output_col)))
    )
    return cleaned
