"""
    Running the class, generating output and plotting the output.
"""
from matplotlib import pyplot as plt
from SOM import SOM
import pandas as pd
import os
from Formatter import FormatCSVFile
from pathlib import Path
import Mapper as mapper
from Tester import Tester


formatter = FormatCSVFile()
tester = Tester()

if Path("huge_merged_csv_file.csv").is_file() is False:
    formatter.format_csv_files()
    formatter.generate_merged_huge_file()

if Path("Validation Data/validation_file.csv").is_file() is False:
    formatter.format_validation_data()
    formatter.generate_validation_file()

if Path("Test Data/test_file.csv").is_file() is False:
    formatter.format_validation_data()
    formatter.generate_validation_file()

# Train SOM, if there is no weights.
if Path("weights.txt").is_file() is False:

    # Use pandas for loading data using dataframes.
    d = os.path.dirname(os.getcwd())
    file_name = "huge_merged_csv_file.csv"
    data = pd.read_csv(file_name, header=None)
    # Shuffle the data in place.
    data = data.sample(frac=1).reset_index(drop=True)

    # create SOM object
    som = SOM(7, 1)

    # Train with 100 neurons.
    som.train(data)

    # Get output grid
    grid = som.get_centroids()

    # Save the weights in a file.
    result_file = open("weights.txt", "w")
    result_file.writelines(str(grid))
    result_file.close()

# Map data to neurons.
mapped_vectors = mapper.map_vects()
tester.open_test_file()
tester.test_som(mapped_vectors)

