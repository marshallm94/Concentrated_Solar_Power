import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def multi_class_scatter_plot(arr_1, arr_2, y, color_list=False):
    classes = np.unique(y)
    if color_list:
        pass
    if not color_list:
        color_list = np.random.choice(list(colors.cnames.keys()), len(classes))

    fig, ax = plt.subplots(figsize=(8,6))
    for i in classes:
        color = np.random.choice(color_list)
        mask = y == i
        ax.scatter(arr_1[mask], arr_2[mask], c=color, label=f"{i}")

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

def horizontal_bar_plot(arr_1, y, title, color_list=False, savefig=False):
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
    if not savefig:
        plt.show()
    else:
        plt.savefig(savefig)

def count_nans(df, verbose=True):
    """
    Calculates NaN percentages per column in a pandas DataFrame.

    Parameters:
        df: (Pandas DataFrame)
        verbose: (Boolean) Prints column names and NaN percentage if True

    Output:
        col_nans: List containing tuples of column names and percentage NaN for that column.
    """
    col_nans = []
    for col in df.columns:
        percent_nan = pd.isnull(df[col]).sum()/len(pd.isnull(df[col]))
        col_nans.append((col, percent_nan))
        if verbose:
            print("{} | {:.2f}% NaN".format(col, percent_nan*100))

    return col_nans

if __name__ == "__main__":
    # read data
    csp_proj = pd.read_csv("../data/csp_proj_20180404.csv", encoding='latin-1')
    # (160, 91)
    pv_locs = pd.read_csv("../data/pv_plant_locs_20180407.csv")
    # (100, 20)
    coal_locs = pd.read_csv("../data/coal_plant_locs_20180407.csv")
    # (100, 14)
    csp_locs = pd.read_csv("../data/csp_plant_locs_20180407.csv")
    # (29, 10)
    ng_locs = pd.read_csv("../data/ngcc_plant_locs_20180407.csv")
    # (100, 14)

    # exploration
    csp_proj_nans = count_nans(csp_proj, verbose=False)

    operational = csp_proj[csp_proj['Status'] == "Operational"]
    under_construction = csp_proj[csp_proj['Status'] == "Under construction"]
    under_dev = csp_proj[csp_proj['Status'] == "Under development"]

    # operational CSP projects
    csp_operational_tech = operational.groupby(['Technology']).count()['ProjectID']
    tech = np.array(csp_operational_tech.index)
    count = np.array([i for i in csp_operational_tech])

    horizontal_bar_plot(tech, count, "Operation CSP Technologies", ['darkblue','mediumblue','lightblue'], "../images/operational_csp_technologies.png")

    # under construction CSP projects
    csp_construction_tech = under_construction.groupby(['Technology']).count()['ProjectID']
    tech = np.array(csp_construction_tech.index)
    count = np.array([i for i in csp_construction_tech])

    horizontal_bar_plot(tech, count, "CSP Technologies Under Construction", ['tomato','darkorange','orange'], "../images/under_construction_csp_technologies.png")

    # under development CSP projects
    csp_under_dev_tech = under_dev.groupby(['Technology']).count()['ProjectID']
    tech = np.array(csp_under_dev_tech.index)
    count = np.array([i for i in csp_under_dev_tech])

    horizontal_bar_plot(tech, count, "CSP Technologies Under Development", ['darkseagreen','seagreen','mediumseagreen','darkcyan', 'red'], "../images/under_development_csp_technologies.png")
