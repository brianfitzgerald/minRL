import os
import pandas as pd

eval_results = []
for file in os.listdir("eval_results"):
    df = pd.read_parquet(f"eval_results/{file}")
    eval_results.append(df)

eval_results = pd.concat(eval_results)
