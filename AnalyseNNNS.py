#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Maria Laura Gabriel Kuniyoshi.
"""
import argparse
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

if __name__ == '__main__':
    MESSAGE = """Calculates summary scores and produces boxplots from a table
    with NeoNatal Neurobehavioral Scale results. """
    parser = argparse.ArgumentParser(description=MESSAGE)
    parser.add_argument("file",
                        type=str,
                        help="Name of the file containing the NNNS raw data.")
    parser.add_argument("--output_filename",
                        type=str,
                        help="Name on the output files.",
                        default="")
    parser.add_argument("--file_type",
                        type=str,
                        help="Type of the input. Default is xlsx.",
                        choices=['xlsx', 'csv'],
                        default="xlsx")
    parser.add_argument("--boxplot",
                        type=bool,
                        help="Produces boxplot images. Default is True",
                        default=True)
    args = parser.parse_args()


#############################################################################
# Importing the file and detecting missing/invalid data.


def import_excel(file):
    """
    Import the data as dataframe.

    Input: name of the excel file, according to the template.
    Output: pandas dataframe.
    """
    df = pd.read_excel(file,
                       header=0,
                       index_col=[0, 1, 2],
                       skiprows=[1, 2, 3, 4])

    # Remove useless columns.
    df = df.drop(1, level=1, axis=0)
    df = df.reset_index(level=0, drop=True)
    df = df.dropna(axis=1, how='all')

    # Convert data types.
    df = df.astype('Int16', errors='ignore')

    return df


def import_csv(file):
    """
    Import the data as dataframe.

    Input: name of the csv file, according to the template.
    Output: pandas dataframe.
    """
    df = pd.read_csv(file, header=0)

    def convers(x):
        try:
            return int(x)
        except:
            return x

    df.iloc[:, 0] = df.iloc[:, 0].apply(convers)
    df = df.set_index(list(df.columns)[0:2])

    # Remove useless columns.
    df = df.drop(1, level=0, axis=0)
    df = df.dropna(axis=1, how='all')

    # Convert data types.
    df = df.astype('Int16', errors='ignore')

    return df


def check_range(df):
    """
    Check if the values are accoring to the description from Lester 2004,
    The Neonatal Intensive Care Unit Network Neurobehavioral Scale Procedures.

    If it is not accordingly, substitute for NaN. 
    """

    valid_scores = {1: [1, 2, 3, 4, 5, 6],
                    2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 95, 96, 97, 98, 99],
                    3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 95, 96, 97, 98, 99],
                    4: [1, 2, 3, 4, 5, 6, 7, 8, 9, 95, 96, 97, 98, 99],
                    5: [1, 2, 3, 4, 5, 98, 99],
                    6: [1, 2, 3, 4, 98, 99],
                    7: [],
                    '7a': [1, 2, 0, np.nan],
                    '7b': [1, 2, 0, np.nan],
                    '7c': [1, 2, 0, np.nan],
                    '7d': [1, 2, 0, np.nan],
                    8: [1, 2, 3, 98, 99],
                    9: [1, 2, 3, 4, 5, 6, 7, 8, 9, 95, 96, 97, 98, 99],
                    10: [1, 2, 3, 4, 98, 99],
                    11: [1, 2, 3, 4, 98, 99],
                    12: [1, 2, 3, 4, 98, 99],
                    13: [1, 2, 3, 4, 5, 98, 99],
                    14: [1, 2, 3, 4, 5, 98, 99],
                    15: [1, 2, 3, 4, 5, 98, 99],
                    16: [1, 2, 3, 4, 5, 6, 98, 99],
                    17: [1, 2, 3, 4, 98, 99],
                    18: [1, 2, 3, 4, 5, 98, 99],
                    19: [1, 2, 3, 4, 5, 98, 99],
                    20: [1, 2, 3, 4, 5, 98, 99],
                    21: [1, 2, 3, 4, 5, 98, 99],
                    22: [1, 2, 3, 4, 5, 6, 7, 98, 99],
                    23: [1, 2, 3, 4, 98, 99],
                    24: [1, 2, 3, 4, 5, 98, 99],
                    25: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 , 98, 99],
                    26: [1, 2, 3, 98, 99],
                    27: [1, 2, 3, 4, 5, 6, 7, 98, 99],
                    28: [1, 2, 3, 4, 5, 6, 98, 99],
                    29: [1, 2, 3, 4, 98, 99],
                    30: [1, 2, 3, 4, 98, 99],
                    31: [1, 2, 0, 98, 99],
                    32: [1, 2, 3, 4, 5, 6, 98, 99],
                    33: [1, 2, 3, 4, 5, 6, 7, 8, 9, 97, 98, 99],
                    34: [1, 2, 3, 4, 5, 6, 7, 8, 9, 97, 98, 99],
                    35: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98, 99],
                    36: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98, 99],
                    37: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98, 99],
                    38: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98, 99],
                    39: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98, 99],
                    40: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98, 99],
                    41: [1, 2, 3, 4, 5, 6, 98, 99],
                    42: [1, 2, 3, 4, 98, 99],
                    43: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 98, 99],
                    44: [1, 2, 3, 4, 98, 99],
                    45: [1, 2, 3, 4, 5, 98, 99],
                    46: [],
                    '46a': [1, 2, 0, np.nan],
                    '46b': [1, 2, 0, np.nan],
                    '46c': [1, 2, 0, np.nan],
                    '46d': [1, 2, 0, np.nan],
                    '46e': [1, 2, 0, np.nan],
                    '46f': [1, 2, 0, np.nan],
                    '46g': [1, 2, 0, np.nan],
                    '46h': [1, 2, 0, np.nan],
                    '46i': [1, 2, 0, np.nan],
                    47: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98],
                    48: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 98],
                    49: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98],
                    50: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 98, 99],
                    51: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    52: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    53: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98],
                    54: [1, 2, 3, 4, 5, 6, 98],
                    55: [1, 2, 3, 4, 5, 6, 98],
                    56: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    57: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98],
                    58: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    59: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    60: [1, 2, 3, 4, 5, 6, 7, 8, 9, 98],
                    61: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    62: [1, 2, 3, 4, 5, 6],
                    63: [1, 2, 3, 4, 5, 6],
                    64: [1, 2, 3, 4, 5, 6],
                    65: [1, 2, 3, 4, 5, 6],
                    66: [1, 2, 0, np.nan],
                    67: [1, 2, 0, np.nan],
                    68: [1, 2, 0, np.nan],
                    69: [1, 2, 0, np.nan],
                    70: [1, 2, 0, np.nan],
                    71: [1, 2, 0, np.nan],
                    72: [1, 2, 0, np.nan],
                    73: [1, 2, 0, np.nan],
                    74: [1, 2, 0, np.nan],
                    75: [1, 2, 0, np.nan],
                    76: [1, 2, 0, np.nan],
                    '77a': [1, 2, 0, np.nan],
                    '77b': [1, 2, 0, np.nan],
                    78: [1, 2, 0, np.nan],
                    79: [1, 2, 0, np.nan],
                    80: [1, 2, 0, np.nan],
                    81: [1, 2, 0, np.nan],
                    82: [1, 2, 0, np.nan],
                    83: [1, 2, 0, np.nan],
                    84: [1, 2, 0, np.nan],
                    85: [1, 2, 0, np.nan],
                    86: [1, 2, 0, np.nan],
                    87: [1, 2, 0, np.nan],
                    88: [1, 2, 0, np.nan],
                    89: [1, 2, 0, np.nan],
                    90: [1, 2, 0, np.nan],
                    91: [1, 2, 0, np.nan],
                    92: [1, 2, 0, np.nan],
                    93: [1, 2, 0, np.nan],
                    94: [1, 2, 0, np.nan],
                    95: [1, 2, 0, np.nan],
                    96: [1, 2, 0, np.nan],
                    97: [1, 2, 0, np.nan],
                    98: [1, 2, 0, np.nan],
                    99: [1, 2, 0, np.nan],
                    100: [1, 2, 0, np.nan],
                    101: [1, 2, 0, np.nan],
                    102: [1, 2, 0, np.nan],
                    103: [1, 2, 0, np.nan],
                    104: [1, 2, 0, np.nan],
                    105: [1, 2, 0, np.nan],
                    106: [1, 2, 0, np.nan],
                    107: [1, 2, 0, np.nan],
                    108: [1, 2, 0, np.nan],
                    109: [1, 2, 0, np.nan],
                    110: [1, 2, 0, np.nan],
                    111: [1, 2, 0, np.nan],
                    112: [1, 2, 0, np.nan],
                    113: [1, 2, 0, np.nan],
                    114: [1, 2, 0, np.nan],
                    115: [1, 2, 0, np.nan]}

    for i in df.iterrows():
        row = i[1].astype('int', errors='ignore')
        variable = i[0]

        if variable[0] in [7, 46]:
            continue

        valid_value = row.isin(valid_scores[variable[0]])
        invalid = list(valid_value[valid_value is False].index)

        for baby in invalid:
            print(f'Invalid or missing score for the variable {variable[0]}, sample {baby}')
            df.loc[variable, baby] = np.nan
    print('Replaced all invalid or missing scores with NaN')
    return df


def import_data(extension, file):
    if extension == 'xlsx':
        df =  import_excel(file)
    elif extension == 'csv':
        df = import_csv(file)
    else:
        print('Unsupported extension')
        sys.exit()

    df = check_range(df)

    # The ones marked as 95 - 99 are missing data. This will set them as NaN.
    df = df.mask(df.isin([95, 96, 97, 98, 99]))

    return df

##############################################################################
# Calculate scores


def binarify(df, row_name, condition_pos, condition_neg):
    """
    Convert numerical variables into binary variables.

    df: Pandas dataframe.
    condition_pos: Condition where class is 1.
    condition_neg: Condition where class is 0.
    """
    x = df.loc[[row_name]]
    x = x.astype('object')

    np_x = np.select([eval(condition_pos), eval(condition_neg), pd.isna(x)],
                     [1                  , 0                  , np.nan    ])

    df.loc[[row_name]] = pd.DataFrame(np_x, columns=x.columns, index=x.index)

    return df


def nhabit(df):
    """Calculate habituation score."""
    items = df.loc[[2, 3, 4]]
    items = items.dropna(axis=1, how='any')
    return items.mean()


def norient(df):
    """Calculate attention score."""
    items = df.loc[[35, 36, 37, 38, 39, 40, 47]]
    items = items.dropna(axis=1, thresh=4)
    return items.mean()


def narousal(df):
    """Calculate arousal score."""
    items = df.loc[[8, 51, 53, 54, 55, 62, 63]]
    items = items.dropna(axis=1, thresh=5)
    return items.mean()


def nselfre3(df):  # Minor differences
    """Calculate regulation score."""
    items = df.loc[[25, 33, 34, 43, 47, 48, 49,
                    50, 52, 56, 57, 58, 59, 60, 61]]
    items = items.reset_index(level=1, drop=True)

    rc_items = items.T.replace({
        25: {10: 2, 11: 1},
        48: {5: 6, 6: 5, 7: 3, 8: 2, 9: 1, 10: 2},
        57: {9: np.nan},
        58: {1: np.nan, 2: 7, 4: 5, 5: 6, 6: 4, 7: 3, 8: 2, 9: 1, 10: 3, 11: 1},
        59: {2: 5, 3: 4, 4: 3, 5: 2, 6: 2, 7: 1, 8: 1, 9: 1},
        61: {6: 7, 7: 8, 8: 9, 9: 10, 10:6}
        }).T
    rc_items.loc[[52]] = 13 - rc_items.loc[[52]]
    rc_items.loc[[56]] = 10 - rc_items.loc[[56]]
    rc_items.loc[[57]] = 10 - rc_items.loc[[57]].replace(9, np.nan)

    rc_items = rc_items.dropna(axis=1, thresh=10)
    return rc_items.mean()


def nhandle(df): # Minor differences
    """Calculate handling score."""
    items = df.loc[['46a', '46b', '46c', '46d', '46e', '46f', '46g', '46h']]
    rc_items = items.replace({2: 0})
    rc_items = rc_items.dropna(axis=1, thresh=7)
    return rc_items.mean()


def nqmove(df):
    """Calculate quality of movement score."""
    items = df.loc[[8, 49, 54, 55, 56, 57]]
    items = items.reset_index(level=1, drop=True)

    rc_items = items.T.replace({
        8: {1: 2, 3: 1},
        54: {1: 4, 2: 4, 3: 4, 4: 3, 5: 2, 6: 1},
        55: {1: 2, 2: 4, 3: 4, 4: 3, 5: 2, 6: 1},
        56: {1: 9, 2: 8, 3: 7, 4: 6, 6: 4, 7: 3, 8: 2, 9: 1},
        57: {1: np.nan, 2: 8, 3: 7, 4: 6, 6: 4, 7: 3, 8: 2, 9: 1}
        }).T

    rc_items = rc_items.dropna(axis=1, thresh=4)
    return rc_items.mean()


def nhexctot(df):  # Minor differences
    """Calculate excitability score."""
    items = df.loc[[33, 34, 48, 49, 50, 51, 52,
                    53, 54, 55, 56, 57, 58, 59, 60]]

    items = items.reset_index(level=1, drop=True)
    itemsT = items.T

    cond = [[33, '(x>=1) & (x<3)', '(x>=3)'],
            [34, '(x>=1) & (x<3)', '(x>=3)'],
            [48, '(x>6)', '(x>=1) & (x<=6)'],
            [49, '(x>=1) & (x<4)', '(x>=4)'],
            [50, '(x>=1) & (x<4)', '(x>=4)'],
            [51, '(x>=7)', '(x>=1) & (x<7)'],
            [52, '(x>6)', '(x>=1) & (x<=6)'],
            [53, '(x>5)', '(x>=1) & (x<=5)'],
            [54, '(x>6)', '(x>=1) & (x<=6)'],
            [55, '(x>6)', '(x>=1) & (x<=6)'],
            [56, '(x>5)', '(x>=1) & (x<=5)'],
            [57, '(x>4)', '(x>=1) & (x<=4)'],
            [58, '(x>7)', '(x>=1) & (x<=7)'],
            [59, '(x>3)', '(x>=1) & (x<=3)'],
            [60, '(x>=1) & (x<3)', '(x>=3)']]
    for i in cond:
        items = binarify(items, *i)

    itemsT2 = items.T

    itemsT2.loc[(itemsT[50].isna()) &
               (itemsT[60].isna()) &
               (itemsT[51] > 0) &
               (itemsT[51] < 7),
               [48, 60]] = 0

    items = itemsT2.T

    return items.sum()


def nbadref1(df):  # Minor differences
    """Calculate non-optimal reflexes score."""
    items = df.loc[[10, 11, 12, 21, 22, 23, 26, 27,
                    28, 30, 32, 41, 42, 44, 45]]

    cond = [[10, '(x == 3)', 'x.isin([1, 2, 4])'],
            [11, '(x == 3)', 'x.isin([1, 2, 4])'],
            [12, 'x.isin([1, 2, 3])', '(x == 4)'],
            [21, '(x == 3)', 'x.isin([1, 2, 4, 5])'],
            [22, '(x == 3)', 'x.isin([1, 2, 4, 5, 6, 7])'],
            [23, '(x == 3)', 'x.isin([1, 2, 4])'],
            [26, '(x == 3)', 'x.isin([1, 2])'],
            [27, '(x == 4)', 'x.isin([1, 2, 3, 5, 6, 7])'],
            [28, '(x == 3)', 'x.isin([1, 2, 4])'],
            [30, '(x == 3)', 'x.isin([1, 2, 4])'],
            [32, '(x == 4)', 'x.isin([1, 2, 3, 5, 6])'],
            [41, '(x == 4)', 'x.isin([1, 2, 3, 5, 6])'],
            [42, 'x.isin([1, 2])', 'x.isin([3, 4])'],
            [44, 'x.isin([1, 2, 3])', '(x == 4)'],
            [45, '(x == 4)', 'x.isin([1, 2, 3, 5])']]
    for i in cond:
        items = binarify(items, *i)

    items = items.groupby(level=0).max()

    return items.apply((lambda x: (x == 0).sum()))


def ndeprtot(df):  # Minor differences
    """Calculate lethargy score."""
    items = df.loc[[25, 35, 36, 37, 38, 39, 40, 43,
                    47, 48, 51, 52, 54, 55, 53]]
    items = items.reset_index(level=1, drop=True)

    itemsT = items.T
    itemsT.loc[(itemsT[47].isin([np.nan, 1, 2]))
               &
               (itemsT[[35, 36, 37, 38, 39, 40]].isna().any(axis=1)),
               ['ALOR_SUB']] = 1
    itemsT['ALOR_SUB'] = itemsT['ALOR_SUB'].fillna(0)

    for i in [35, 36, 37, 38, 39, 40, 47]:
        itemsT.loc[(itemsT['ALOR_SUB'] == 1) & (itemsT[i].isna()), i] = 1
    items = itemsT.T
    items.drop(index=['ALOR_SUB'])

    cond = [[i, '(x>=1) & (x<4)', '(x>=4)'] for i in items.index[:-1]]
    cond.append([53, '(x>=1) & (x<3)', '(x>=3)'])
    for i in cond:
        items = binarify(items, *i)

    return items.sum().astype('float64')


def nasymtot(df):
    """Calculate asymmetrical reflexes score."""
    items = df.loc[[10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 23, 26, 27, 29]]
    items = items.dropna(axis=1, how='any')

    cond = [[i, '(x == 5)', 'x.isin([1, 2, 3, 4])']
            for i in [10, 11, 12, 13, 14, 15, 16, 17,
                      18, 19, 20, 21, 23, 26, 27, 29]]
    for i in cond:
        items = binarify(items, *i)

    items = items.groupby(level=0).max()

    return items.sum()


def nhypetot(df):
    """Calculate hypertonicity score."""
    items = df.loc[[5, 13, 16, 17, 18, 24, 25, 27, 28, 48]]

    cond = [[5, '(x == 5)', 'x.isin([1, 2, 3, 4])'],
            [13, '(x == 5)', 'x.isin([1, 2, 3, 4])'],
            [16, '(x == 6)', 'x.isin([1, 2, 3, 4, 5])'],
            [17, '(x == 4)', 'x.isin([1, 2, 3])'],
            [18, '(x == 5)', 'x.isin([1, 2, 3, 4])'],
            [24, '(x == 5)', 'x.isin([1, 2, 3, 4])'],
            [25, '(x == 10)', 'x.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 11])'],
            [27, '(x == 6)', 'x.isin([1, 2, 3, 4, 5, 7])'],
            [28, '(x == 5)', 'x.isin([1, 2, 3, 4, 6])'],
            [48, '(x.isin([8, 9]))', 'x.isin([1, 2, 3, 4, 5, 6, 7, 10])']]
    for i in cond:
        items = binarify(items, *i)

    items = items.groupby(level=0).max()

    return items.sum()


def nhypotot(df):
    """Calculate hypotonicity score."""
    items = df.loc[[5, 13, 16, 17, 18, 24, 25, 27, 28, 48]]

    cond = [[5, '(x == 1)', 'x.isin([2, 3, 4, 5])'],
            [13, '(x == 1)', 'x.isin([2, 3, 4, 5])'],
            [16, '(x == 1)', 'x.isin([2, 3, 4, 5, 6])'],
            [17, '(x == 1)', 'x.isin([2, 3, 4])'],
            [18, '(x == 1)', 'x.isin([1, 2, 3, 4, 5])'],
            [24, '(x == 1)', 'x.isin([1, 2, 3, 4, 5])'],
            [25, '(x == 1)', 'x.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])'],
            [27, '(x == 1)', 'x.isin([2, 3, 4, 5, 6, 7])'],
            [28, '(x == 1)', 'x.isin([1, 2, 3, 4, 5, 6])'],
            [48, '(x == 1)', 'x.isin([2, 3, 4, 5, 6, 7, 8, 9, 10])']]
    for i in cond:
        items = binarify(items, *i)

    items = items.groupby(level=0).max()

    return items.sum()


def nstress(df):
    """Calculate stress and abstinence score."""
    items = df.loc[[66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, '77a', '77b',
                    78, 79, 80, 81, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
                    95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                    108, 109, 110, 111, 112, 113, 114, 115]]
    items = items.replace(2, 0)

    items = items.dropna(axis=1, thresh=40)
    return items.mean()


##############################################################################
# Get results


def summary_scores(df):
    """
    Calculate all summary scores for the data.

    Input: pandas dataframe.
    Output: dictionary with pandas series as values.
    """
    sum_scores = {'habituation': nhabit(df),
                  'attention': norient(df),
                  'arousal': narousal(df),
                  'regulation': nselfre3(df),
                  'handling': nhandle(df),
                  'Quality of movement': nqmove(df),
                  'excitability': nhexctot(df),
                  'lethargy': ndeprtot(df),
                  'nonoptimal reflexes': nbadref1(df),
                  'assymetrical reflexes': nasymtot(df),
                  'hypertonicity': nhypetot(df),
                  'hypotonicity': nhypotot(df),
                  'stress-abstinence': nstress(df)
                  }
    return sum_scores


def summary_table(sum_scores):
    """Return summary statistics for the summary scores."""
    summary_stats = {k: v.describe() for k, v in sum_scores.items()}
    return pd.DataFrame(summary_stats)


def boxplots(sum_scores, output):
    """Produce and saves boxplot for each summary score."""
    for k, v in sum_scores.items():
        fig, ax = plt.subplots()
        ax.boxplot(list(v))
        ax.set_title(k)
        fig.savefig(f'{k}_{output}.png', format='png')


def main():
    file = args.file
    output = args.output_filename
    extension = args.file_type

    df = import_data(extension, file)
    sum_scores = summary_scores(df)
    sum_table = summary_table(sum_scores)
    sum_table.to_csv(f'summary_scores_{output}.txt', sep='\t')

    if args.boxplot is True:
        boxplots(sum_scores, output)


if __name__ == "__main__":
    main()
