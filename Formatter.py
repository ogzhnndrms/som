"""
This module contains a class which performs formatting file for efficient use of memory.
"""
import sys
import fileinput
import pandas as pd


class FormatCSVFile:
    """
    This class formats csv file for memory efficiency.
    """

    def format_csv_files(self):
        """
        This function takes file and delete some part of file.

        Args:

        Return:
            valid if operation is successfull.
        """

        # Delete line numbers because
        # In python 2.7 a string allocates 37 bytes in memory
        # In this work, this situation causes waste of 48 MB memory.
        for i in range(1, 12):
            for line in fileinput.input("Activity Recognition from Single Chest-Mounted Accelerometer/" + str(i) + ".csv", inplace=True):
                # Write again same line without its line number and class.
                first_comma_index = line.find(',')
                # Added 1 because of index starts 0
                # but operation index starts from 1
                print "%s" % (line[first_comma_index + 1:-3])

    def format_validation_and_test_data(self):
        """
        This function formats validation file,
        deletes first element (until comma).
        """
        for i in range(12, 14):
            for line in fileinput.input("Validation Data/" + str(i) + ".csv", inplace=True):
                # Write again same line without its line number and class.
                first_comma_index = line.find(',')
                # Added 1 because of index starts 0
                # but operation index starts from 1
                print "%s" % (line[first_comma_index + 1:-1])

    def generate_merged_huge_file(self):
        """
        Read all files and count number of rows then split them to %80 for training
        %10 for validation, %10 for testing.

        """

        f_out = open("huge_merged_csv_file.csv", "a")
    
        for i in range(1, 12):

            f = open("Activity Recognition from Single Chest-Mounted Accelerometer/" + str(i) + ".csv")
            for line in f:
                f_out.write(line)
            f.close()

        f_out.close()

    def generate_file(self, directory=None, file_range=None, file_name=None):

        f_out = open(directory + "/" + file_name, "a")
    
        for file_index in file_range:
            f = open(directory + "/" + str(file_index) + ".csv")
            for line in f:
                f_out.write(line)
            f.close()

        f_out.close()