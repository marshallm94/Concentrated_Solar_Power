import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def multi_class_scatter_plot(arr_1, arr_2, y):
    import matplotlib.colors as colors
    color_list = list(colors.cnames.keys())
    classes = np.unique(y)

    fig, ax = plt.subplots(figsize=(8,6))
    for i in classes:
        color = np.random.choice(color_list)
        mask = y == i
        ax.scatter(arr_col_idx_1[mask], arr_col_idx_2[mask], c=color, label=f"{i}")

    ax.legend()
    plt.show()

def multi_class_bar_plot(arr_1, y, color_list):
    import matplotlib.colors as colors
    # color_list = list(colors.cnames.keys())
    classes = np.unique(arr_1)
    x_axis = np.linspace(0, len(classes), len(classes))
    fig, ax = plt.subplots(figsize=(8,6))
    for x, i in enumerate(classes):
        # color = np.random.choice(color_list)
        print(len(classes), len(color_list))
        color = color_list[x]
        mask = arr_1 == i
        ax.bar(x_axis[x],height=y[x], color=color, label=f"{i}")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("data/concentrating_solar_projects_20180404.csv", encoding='latin-1')

    for col in data.columns:
        print(col, pd.isnull(data[col]).sum()/len(pd.isnull(data[col])))

    data.groupby("Status").count()['ProjectID']

    operational = data[data['Status'] == "Operational"]

    for col in operational.columns:
        print(col, pd.isnull(operational[col]).sum()/len(pd.isnull(operational[col])))

    tech = data.groupby("Technology").count()['ProjectID']

    status_tech_data = data.groupby(['Status','Technology']).count()['ProjectID']
    status = []
    tech = []
    for i in status_tech_data.index:
        status.append(i[0])
        tech.append(i[1])
    status = np.array(status)
    tech = np.array(tech)
    count = np.array([i for i in status_tech_data])

    multi_class_bar_plot(status, count, ['red','green','blue','cyan','yellow'])
