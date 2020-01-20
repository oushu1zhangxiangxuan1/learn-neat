
"""
Generate new column with random or same numeric data at the left/right side of csv column.
Parameters:
    file: path to the origin_file
    output: output file path
    
    Optional:
    
    header: csv data header, split with comma, default None
    random: bool, default False
        True:generate random float data in [0,1)
        False: generate column only contains 0
    
    front: bool, default True
        True:generate column at the left side of the file.
        False: generate column at the right side of the file.

"""

import env
import ast
import argparse
import pandas as pd
import numpy as np 

from meat.utils.uuid_short import short_uuid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "Generate new column with random or same numeric data at the left/right side of csv column."
    parser.add_argument("file", help="Source file path", type=str)
    parser.add_argument("output", help="Output file path", type=str)

    parser.add_argument("--header", help="Csv data header, split with comma", type=str, default=None)
    parser.add_argument("--random", help="Whether generate random data ", type=bool, default=False)
    parser.add_argument("--front", help="Position to generate new column", type=ast.literal_eval, default=True)
    
    parser.add_argument("--debug", help="train data file absolute path", type=bool, default=False)
    parser.add_argument("--test", help="test data file absolute path", type=bool, default=False)
    
    args = parser.parse_args()
    print("args:\n{}".format(args))
    DEBUG = args.debug
    TEST  = args.test

    src = pd.read_csv(args.file, header=args.header)

    UUID = short_uuid()

    if args.random:
        col_name = "RANDOM_" + UUID
        new_data = np.random.rand(src.shape[0])
    else:
        col_name = "ZERO_" + UUID
        # new_data = np.zeros(src.shape[0], dtype=int)
        new_data = np.ones(src.shape[0], dtype=int)

    if args.front:
        src.insert(0, col_name, new_data)
        src[col_name]=new_data
    else:
        src[col_name] = new_data

    src.to_csv(args.output, header=False, index=False)
    