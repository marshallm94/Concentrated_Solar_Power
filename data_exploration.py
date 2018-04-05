import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = pd.read_csv("data/concentrating_solar_projects_20180404.csv", encoding='latin-1')

    for col in data.columns:
        print(col, pd.isnull(data[col]).sum()/len(pd.isnull(data[col])))

    data.groupby("Status").count()['ProjectID']

    operational = data[data['Status'] == "Operational"]

    for col in operational.columns:
        print(col, pd.isnull(operational[col]).sum()/len(pd.isnull(operational[col])))

    tech = data.groupby("Technology").count()['ProjectID']

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.bar(tech.index, tech)
    plt.xticks(rotation=65)
    plt.show()
