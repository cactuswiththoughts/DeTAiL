import argparse


# TODO Convert the data into MS Swift format
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

