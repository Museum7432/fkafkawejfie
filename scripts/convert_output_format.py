import pandas as pd
import os
import argparse



def main():
    parser = argparse.ArgumentParser(
        description="transform multivers output into the final format"
    )

    parser.add_argument("--input_file", type=str,help="path to input file")

    parser.add_argument("--output_file", type=str,help="path to output file")

    args = parser.parse_args()

    if not args.input_file:
        parser.error("input file needed!")
    if not args.input_file:
        parser.error("output file needed!")

    
    