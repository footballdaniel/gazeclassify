from typing import List
import csv
import os.path

class CsvLoader:
    ''' A class used to extract gaze data from csv files.'''

    def __init__(self, folderpath: str): 
        '''The input of the __init__ function of this class should be the path to the folder containing data.
        Input should look like : "C:/Desk/CsvFiles".'''
        self.folderpath = folderpath

    def check_if_file_exists(self, filename: str = "timestamps.csv"):
        ''' This function is used in others to check if the file containing data exists. 
        Input should look like "filename.csv".'''
        self.filename = filename
        os.path.exists(self.folderpath + "/" + self.filename)

    def read_timestamps(self, filename: str = "timestamps.csv", column_name: str = "timestamps")-> List[float]:
        ''' This function is used to extract timestamps.
        filename should be the name of the file containing timestamps.
        column_name corresponds to the name of the column in which those timestamps are.'''
        self.check_if_file_exists(filename)
        self.column_name = column_name
        f = open(self.folderpath + "/" + self.filename)
        CSVFile = csv.reader(f)
        header = next(CSVFile)
        if self.column_name in header:
            self.timestamps = []
            for _, row in enumerate(CSVFile):
                self.timestamps.append(float(row[0]))
            return self.timestamps
        else:
            print("File does not have the right columns")

    def read_position(self, filename: str = "gaze_positions.csv", column_name_x: str = "norm_pos_x", column_name_y: str = "norm_pos_y")-> List[float]:
        ''' This function is used to extract the gaze positions along the X and Y axis.
        filename should be the path to the file containing the gaze positions data. 
        column_name_x corresponds to the name of the column in which the gaze position along the x axis is. 
        column_name_y corresponds to the name of the column in which the gaze position along the y axis is.'''
        self.check_if_file_exists(filename)
        self.column_name_x = column_name_x
        self.column_name_y = column_name_y
        f = open(self.folderpath + "/" + self.filename)
        file_handle = csv.reader(f)
        self.position_x = []
        self.position_y = []
        header = next(file_handle)
        if (self.column_name_x in header) & (self.column_name_y in header):
            for row in file_handle:
                self.position_x.append(float(row[3]))
                self.position_y.append(float(row[4]))
        else:
            print("File does not have the right columns")
        return self.position_x, self.position_y