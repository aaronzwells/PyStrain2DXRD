import pandas as pd
import os
import sys

def convert_q_units(input_file):
    """
    Reads a two-column (q, I) text file, converts q from nm^-1 to Å^-1
    by dividing by 10, and saves the result to a new file.

    Args:
        input_file (str): Path to the input file.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        sys.exit(1)

    try:
        # Read the data from the file. Assumes space-separated values.
        df = pd.read_csv(input_file, sep=r'\s+', header=None, names=['q_nm-1', 'I'])

        # Create a new DataFrame for the converted data
        df_converted = pd.DataFrame()
        df_converted['q_A-1'] = df['q_nm-1'] / 10
        df_converted['I'] = df['I']

        # Create the output filename
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_A-1{ext}"

        # Save the new DataFrame to a text file
        df_converted.to_csv(output_file, sep=' ', index=False, header=False, float_format='%.6f')
        print(f"Successfully converted file and saved to '{output_file}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    filename = "InputFiles/q_vs_I.txt"
    convert_q_units(filename)