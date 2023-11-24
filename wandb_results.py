import os
import math
import glob
import datetime
import pytz
import json
import argparse

import pandas as pd 
import wandb


def filter_runs(runs, group_filter=None, jt_filter=None, config_filter=None, results_filter=None, include_timestamp=False, show_keys=False):
    """ filter wandb api runs by values - any None filters are ignored

    :group_filter:      Wandb run group name to retrieve
    :jt_filter:         Wandb job type name to retrieve
    :config_filter:     list of config paramters to filter by
    :results_filter:    list of results metrics to filter by
    :include_timestamp: include timestamp of runs
    :show_keys:         print the config and summary keys to console on the first valid run

    returns filtered results as a pandas DataFrame
    """

    output = []

    for run in runs:
        if (run.group == group_filter or group_filter == "") and (run.job_type == jt_filter or jt_filter == ""):

            run_params = {}
            run_params['group'] = run.group
            run_params['job_type'] = run.job_type
            run_params['run_name'] = run.name

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config = {k: v for k, v in run.config.items()
                if not k.startswith('_')}

            if config_filter and config_filter[0].lower() != "different":
                for config_param in config_filter:
                    run_params[config_param] = config[config_param]
            else:
                run_params.update(config)   # All

            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files 
            summary = run.summary._json_dict

            if show_keys:
                print("Config keys: ", config.keys())
                print("Summary keys: ", summary.keys())
                show_keys = False

            if results_filter:
                valid_result = False
                for summary_param in results_filter:
                    if summary_param in summary.keys():
                        value = summary[summary_param]
                        # if 'accuracy' in summary_param:
                        #     value = round(value, 2)
                        # elif 'f1_score' in summary_param:
                        #     value = round(value, 3)
                        run_params[summary_param] = value
                        valid_result = True
                if valid_result:
                    if include_timestamp:
                        run_params['timestamp'] = summary['_timestamp']
                    output.append(run_params)   # Only add if results_filter is specified and found (if not found, likely a different run type or not finished)
            else:
                run_params.update(summary)  # All
                output.append(run_params)

    if len(output) == 0:
        raise Exception("No runs found for given filters. Please check valid values are passed.")

    df = pd.DataFrame(output)

    if config_filter and config_filter[0].lower() == 'different':
        df = remove_constant_columns(df)
    return df


def remove_constant_columns(df):
    """ Remove columns where all values are the same.

    :df:     (pandas.DataFrame): The input DataFrame.

    returns pandas.DataFrame: A new DataFrame without the constant columns.
    """
    # Initialize an empty list to store constant columns
    constant_columns = []

    for col in df.columns:
        # Convert column values to their string representations
        str_values = df[col].apply(str)

        # Get the set of unique string representations
        unique_str_values = set(str_values)

        # If there's only one unique value, it means all values in the column are the same
        if len(unique_str_values) == 1:
            constant_columns.append(col)

    # Drop constant columns
    return df.drop(columns=constant_columns)


def get_columns_mapper(df):
    """ Alter column names from variable names to better titles """

    columns_mapper = {}
    for col in df.columns:
        renamed_col = col.replace('/', ' ')
        renamed_col = renamed_col.replace('_', ' ')
        renamed_col = renamed_col.title()   # Capitalize each word

        # .title() method messes up LR and NN capitalization
        renamed_col = renamed_col.replace('Lr', 'LR')
        renamed_col = renamed_col.replace('Nn', 'NN')
        columns_mapper[col] = renamed_col
    return columns_mapper


def main():
    opt = parse_opt()
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(opt.project)

    df = filter_runs(runs,
                     group_filter=opt.group_filter,
                     jt_filter=opt.jt_filter,
                     config_filter=opt.config_filter,
                     results_filter=opt.results_filter,
                     include_timestamp=opt.include_timestamp,
                     show_keys=opt.show_keys)

    if opt.sort_by:
        df = df.sort_values(by=opt.sort_by, ascending=False)

    if opt.best_metric:
        df = df.nlargest(opt.n_largest, opt.best_metric)

    # Round values
    for col in df.columns:
        if 'accuracy' in col and isinstance(df[col][0], float):
            df[col] = df[col].round(2)
        elif 'f1_score' in col and isinstance(df[col][0], float):
            df[col] = df[col].round(3)
        elif 'lr' in col and isinstance(df[col][0], float):
            df[col] = df[col].round(6)
        elif 'ssim' in col.lower() and isinstance(df[col][0], float):
            df[col] = df[col].round(4)
        elif 'psnr' in col.lower() and isinstance(df[col][0], float):
            df[col] = df[col].round(3)
        elif 'sam' in col.lower() and isinstance(df[col][0], float):
            df[col] = df[col].round(3)
        elif 'ergas' in col.lower() and isinstance(df[col][0], float):
            df[col] = df[col].round(3)
        elif 'timestamp' in col.lower() and isinstance(df[col][0], float):
            df[col] = df[col].round(0)

    # Convert variable names into nicer looking titles
    columns_mapper = get_columns_mapper(df)
    df = df.rename(columns=columns_mapper)

    df.to_csv(opt.save_file)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='au-phd/Cow-ID', help='Wandb project name.')
    #parser.add_argument('--group_filter', nargs="+", type=str, default=[], help="Group(s) to filter runs by.")
    parser.add_argument('--group_filter', type=str, default="", help="Group to filter runs by. Default no filter")
    parser.add_argument('--jt_filter', type=str, default="", help="Job type to filter runs by. Default no filter")
    parser.add_argument('--config_filter', nargs="+", type=str, default=[], help="Config (training parameters) to add to output. Default all. Can pass 'different' which will identify and keep columns containing different values (i.e. not the same parameter for each)")
    parser.add_argument('--results_filter', nargs="+", type=str, default=[], help="Result(s) metrics to add to output. Default all.")
    parser.add_argument('--save_file', type=str, default='results.csv', help='Output file to save results to.')
    parser.add_argument('--best_metric', type=str, default="", help="Evaluation metric to return n_largest results based on. Default not applied.")
    parser.add_argument('--n_largest', type=int, default=5, help='Number of best_metric runs to return (if best_metric is specified).')
    parser.add_argument('--show_keys', action='store_true', help='Print the config and summary keys to console on the first valid run.')
    parser.add_argument('--sort_by', nargs="+", type=str, default=[], help="Sort results by column name. Multiple columns can be specified. Default not applied.")
    parser.add_argument('--include_timestamp', action='store_true', help='Include timestamp of runs.')
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    main()
