import os
import pickle
import matplotlib.pyplot as plt
from eda import *
from modeling_base import *


def get_pickle_files(dirpath):
    '''
    Reads all pickle files in dirpath in as dictionaries and returns a
    list of dictionaries.

    Parameters:
    ----------
    dirpath : (str)
        The absolute or relative path to the directory in which the files
        are stored.

    Returns:
    ----------
    dicts : (list)
        List of dictionaries
    '''
    dicts = []
    for pickle_file in os.listdir(dirpath):
        filepath = dirpath + pickle_file
        with open(filepath, 'rb') as f:
            current = pickle.load(f)
            dicts.append(current)

    return dicts


def format_dict_for_plot(lst, key_identifiers):
    '''
    Creates a dictionary from a list of dictionaries in order to be passed
    to a plotting function. (Used to deconstruct list of dictionaries output
    from get_pickle_files())

    Parameters:
    ----------
    lst : (list)
        List of dictionaries
    key_identifiers : (list)
        List of strings (len(2)). Each element should be part of the key for all
        the values that you would like in the final dictionary

    Returns:
    ----------
    out : (dict)
        Dictionary with keys equal to those keys within lst that have
        key_identifier in their key, and values equal to the values of
        those keys
    '''
    total = {}
    for dictionary in results:
        for k, v in dictionary.items():
            for lower_dict_key, lower_dict_value in dictionary[k].items():
                if key_identifiers[0] in lower_dict_key or key_identifiers[1] in lower_dict_key:
                    out_key = k + " " + lower_dict_key
                    total[out_key] = lower_dict_value
    return total


def results_error_plot(error_dict, model_colors, base_colors, title, xlab, ylab, savefig=False):
    '''
    Plots multiple error arraysthe errors of two model against each other

    Parameters:
    ----------
    error_dict : (dict)
        A dictionary where the keys are the names of the error arrays
        (i.e 'Linear Regression Error') and the values are an array_like
        (array/list) sequence of errors
    model_colors : (list)
        List of strings with length equal to number of keys in error_dict
        divided by 2
    base_colors : (list)
        List of strings with length equal to the number of keys in error_dict
        divided by 2
    title : (str)
        The title for the plot
    xlab : (str)
        Label for x-axis
    ylab : (str)
        Label for y-axis
    savefig : (bool/str)
        If False default, image will be displayed and not saved. If the
        user would like the image saved, pass the filepath as string to
        which the image should be saved.

    Returns:
    ----------
    None
    '''
    fig, ax = plt.subplots(figsize=(12,8))
    model_counter = 0
    base_counter =0

    for name, array in error_dict.items():
        broken_name = name.split()
        if 'Persistence' in broken_name:
            ax.plot(array, c=base_colors[base_counter])
            base_counter += 1
        else:
            ax.plot(array, c=model_colors[model_counter])
            model_counter += 1
    plt.xticks(range(0,12), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'])
    plt.xlabel(xlab, fontweight='bold', fontsize=19)
    plt.ylabel(ylab, fontweight='bold', rotation=0, fontsize=19)
    ax.tick_params(axis='both', labelcolor='black', labelsize=15.0)
    ax.yaxis.set_label_coords(-0.105,0.5)
    plt.suptitle(title, fontweight='bold', fontsize=21)
    if savefig:
        plt.savefig(savefig)
    plt.show()


def separate_dict(units, parent_dict):
    '''
    Used to pull associated keys out of a dictionary. Keys in parent_dict
    must be separated by underscores. All keys that have the str specified
    by units will be returned with their associated values in the output
    dictionary

    Parameters:
    ----------
    units : (str)
        An identifier that is in the keys of every key you would like
        separated out from parent_dict
    parent_dict : (dict)
        The parent dictionary with multiple keys

    Returns:
    ----------
    out : (dict)
        Dictionary of key value pairs that have units in the key name
    '''
    out = {}
    for k, v in parent_dict.items():
        words = k.split("_")
        if units in words:
            out[k] = v
    return out


def dict_plot(dict, model_color_map, base_model_color_map, title, xlab, ylab, savefig=False):
    '''
    Creates colormaps for each list of values in dict and passes through to
    results_error_plot() (helper function)

    Parameters:
    ----------
    error_dict : (dict)
        A dictionary where the keys are the names of the error arrays
        (i.e 'Linear Regression Error') and the values are an array_like
        (array/list) sequence of errors
    model_colors_map : (str)
        Valid string specifying a seaborn color pallete
    base_model_color_map : (str)
        Valid string specifying a seaborn color pallete
    title : (str)
        The title for the plot
    xlab : (str)
        Label for x-axis
    ylab : (str)
        Label for y-axis
    savefig : (bool/str)
        If False default, image will be displayed and not saved. If the
        user would like the image saved, pass the filepath as string to
        which the image should be saved.

    Returns:
    ----------
    None
    '''
    model_color_map = sns.color_palette(model_color_map,len(dict.keys())//2)
    base_model_color_map = sns.color_palette(base_model_color_map,len(dict.keys())//2)
    results_error_plot(dict, model_color_map, base_model_color_map, title, xlab, ylab, savefig)


def create_mean_pairs(dictionary):
    '''
    Takes as input a dictionary whose values are lists of even length (i.e. 24)
    and returns a dictionary with the same keys. The values of the output
    dictionary will be the mean of every pair of numbers in the input dictionary
    value lists.

    Example:
        [1]: input_dict = {'key1': [10, 20, 30, 40, 50, 60],
                           'key2': [70, 80, 90, 100, 110, 120]}
        [2]: output_dict = create_mean_pairs(input_dict)
        [3]: output_dict = {'key1': [15.0, 35.0, 55.0],
                            'key2': [75.0, 95.0, 115.0]}

    Parameters:
    ----------
    dictionary : (dict)
        Dictionary with array like values of even length

    Returns:
    ----------
    out : (dict)
        Dictionary with pair-wise means of input dictionary values
    '''
    out = {}
    for k in dictionary.keys():
        out[k] = []
        cache = []
        for v in dictionary[k]:
            if len(cache) == 1:
                v = np.mean((cache[0], v))
                out[k].append(v)
                cache = []
            elif len(cache) == 0:
                cache.append(v)
    return out


if __name__ == "__main__":

    results = get_pickle_files("../pickle_results/")
    total = format_dict_for_plot(results, ['RMSE', 'MAE'])
    units = ['month','week','day','hour']
    for unit in units:
        time_dict = separate_dict(unit, total)
        plot_dict = create_mean_pairs(time_dict)
        title = unit.capitalize() + "s"
        dict_plot(plot_dict, 'Greens', "Reds", f"MAE & RMSE Over Multiple {title}", "Month", r"$\frac{Watts}{Meter^2}$")

    years = separate_dict('year', total)
    new_years = create_mean_pairs(years)


    dict_plot(new_years, 'Greens', "Reds", "MAE & RMSE Over Multiple Years", "Month", r"$\frac{Watts}{Meter^2}$", "../images/boostrapped_nn_errors.png")
