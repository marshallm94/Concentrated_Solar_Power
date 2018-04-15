import pandas as pd
import numpy as np
import os

def build_master_csv(root, filename):
    """
    Reads all files in filepath specified by root parameter and
    concatenates into one Pandas DataFrame, which is then saved as a
    separate file specified by filename parameter

    Parameters:
        root: (str) The relative or absolute path to the directory containing all files desired to be concatenated.
        filename: (str) The filename where the final data object should be saved.

    Returns:
        None
    """
    all_files = os.listdir(path=root)
    all_files.remove("header.csv")
    col_names = list(pd.read_csv("../data/solar_measurements/header.csv"))
    for x, file in enumerate(all_files):
        current_file = root + file
        if x == 0:
            df = pd.read_csv(current_file, header=None, names=col_names)
        else:
            current_df = pd.read_csv(current_file, header=None, names=col_names)
            df = pd.concat([df, current_df], axis=0)

        print(f"Final DataFrame Shape: {df.shape}")

    df.to_csv(filename)
    print(f"\nAll files concatenated to one data object and saved to {filename}\nProcess Complete.")

if __name__ == "__main__":

    build_master_csv("../data/solar_measurements/", "../data/ivanpah_measurements.csv")

    df = pd.read_csv("../data/ivanpah_measurements.csv")
