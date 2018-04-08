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

def multi_class_bar_plot(arr_1, y, title, color_list=False):
    classes = np.unique(arr_1)
    if color_list:
        pass
    if not color_list:
        color_list = np.random.choice(list(colors.cnames.keys()), len(classes))

    x_axis = np.linspace(0, len(classes), len(classes))
    fig, ax = plt.subplots(figsize=(8,6))
    for x, i in enumerate(classes):

        color = color_list[x]
        mask = arr_1 == i
        ax.bar(x_axis[x],height=y[x], color=color, label=f"{i}")

    plt.xticks([])
    ax.legend()
    plt.suptitle(title, fontsize=18, fontweight="bold")
    plt.show()

def horizontal_bar_plot(arr_1, y, title, color_list=False):
    classes = np.unique(arr_1)

    if color_list:
        pass
    if not color_list:
        color_list = np.random.choice(list(colors.cnames.keys()), len(classes))

    desc_idx = np.argsort(y)[::-1]
    classes = classes[desc_idx]
    y = y[desc_idx]

    y_axis = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8,6))
    for x in range(len(classes)):
        color = color_list[x]
        ax.barh(y_axis[x], y[x], color=color, label="{}".format(classes[x]))

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=16)
    plt.suptitle(title, fontsize=18, fontweight="bold")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # read data
    csp_proj = pd.read_csv("data/csp_proj_20180404.csv", encoding='latin-1')
    # (160, 91)
    pv_locs = pd.read_csv("data/pv_plant_locs_20180407.csv")
    # (100, 20)
    coal_locs = pd.read_csv("data/coal_plant_locs_20180407.csv")
    # (100, 14)
    csp_locs = pd.read_csv("data/csp_plant_locs_20180407.csv")
    # (29, 10)
    ng_locs = pd.read_csv("data/ngcc_plant_locs_20180407.csv")
    # (100, 14)




    # for col in csp_proj.columns:
    #     print(col, pd.isnull(csp_proj[col]).sum()/len(pd.isnull(csp_proj[col])))

    csp_proj.groupby("Status").count()['ProjectID']

    operational = csp_proj[csp_proj['Status'] == "Operational"]

    # for col in operational.columns:
    #     print(col, pd.isnull(operational[col]).sum()/len(pd.isnull(operational[col])))

    tech = csp_proj.groupby("Technology").count()['ProjectID']

    tech_csp_proj = csp_proj.groupby(['Technology']).count()['ProjectID']
    tech = []
    for i in tech_csp_proj.index:
        tech.append(i)
    tech = np.array(tech_csp_proj.index)
    count = np.array([i for i in tech_csp_proj])

    multi_class_bar_plot(tech, count, "Operation CSP Technologies", ['red','green','blue','darkred','darkblue','darkgreen'])

    horizontal_bar_plot(tech, count, "Operation CSP Technologies", ['red','green','blue','darkred','darkblue','darkgreen'])
