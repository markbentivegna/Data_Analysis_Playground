import pandas as pd

def generate_csv_file(df, survived, filename):
    df["Survived"] = survived
    df[["PassengerId", "Survived"]].to_csv(filename, index=False)