from __future__ import annotations
from typing import Any

from matplotlib.figure import Figure  # type: ignore
from tabulate import tabulate
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore


class Results:
    ''' A class used to analyze results obtained with the GazeClassify package.'''

    def __init__(self, filepath: str) -> None:
        '''The __init__ function of this class requires the path of the csv results obtained with the gazeclassify package as input.
        This function creates the right dataframes including "self.Results" containing for every joint the pourcentage of frames where it was the closest to gaze irrelevantly of wether the gaze is located on or outside of the human shape.'''
        pd.set_option('chained_assignment', None)
        self.filepath = filepath
        self.dataframe = pd.read_csv(filepath)
        self.dataframe['frame'] = pd.to_numeric(self.dataframe['frame'], downcast='float')
        self.dataframe.sort_values(by=['frame'], inplace=True)
        self.df1 = pd.read_csv(filepath)
        self.df1.sort_values(by=['distance'], inplace=True)
        self.dfHJ = self.df1[self.df1['name'] == 'Human_Joints']
        self.dfHJdrop = self.dfHJ.drop_duplicates(subset='frame', keep="first")
        self.dfHJdrop.sort_values(by=['frame'], inplace=True)
        self.df3 = self.dfHJdrop.groupby(['joint'], as_index=False, dropna=False).agg(
            n=pd.NamedAgg(column='joint', aggfunc='count'), )
        self.total = self.df3.n.sum()
        self.df4 = pd.DataFrame(self.df3.n / self.total * 100)
        self.df4.columns = ['Percentages']
        self.Results = pd.concat([self.df3, self.df4], axis=1)
        self.Results.sort_values(by=['Percentages'], ascending=False, inplace=True)
        self.Head = self.Results.head()

    def results(self) -> Any:
        ''' This function produces a dataframe corresponding to the "results" obtained with the __init__ function.
        This function requires running the __init__ function first.'''
        return self.Results

    def table(self, tablename: str = "Table") -> str:
        ''' This function produces a table corresponding to the "results" obtained with the __init__ function.
        This function requires running the __init__ function first.'''
        self.tablename = tablename
        self.tablename = (tabulate(self.Results, self.Head, tablefmt="github", floatfmt=(".2f")))
        return self.tablename

    def piechart(self) -> Results:
        ''' This function produces pie chart corresponding to the "results" obtained with the __init__ function.
        This function requires running the __init__ function first.'''
        self.fig = plt.figure()
        self.Rounded = np.round(self.Results['Percentages'], decimals=2)
        self.pie = plt.pie(self.Rounded, labels=self.Rounded)
        plt.legend(self.pie[0], self.Results.joint, bbox_to_anchor=(1.2, 0.5), loc="center right", fontsize=10,
                   bbox_transform=plt.gcf().transFigure)
        plt.title("Pourcentages of frames for each joint where it was the closest to gaze")
        plt.show()
        return self

    def barplot(self) -> Results:
        ''' This function produces a barplot corresponding to the "results" obtained with the __init__ function.
        This function requires running the __init__ function first.'''
        plt.rcdefaults()
        self.fig, self.ax = plt.subplots()
        self.ax.barh(self.Results.joint, self.Results.Percentages, align='center', color="cyan",
                     orientation="horizontal")
        self.ax.set_yticks(self.Results.joint)
        self.ax.set_yticklabels(self.Results.joint)
        self.ax.invert_yaxis()
        self.ax.set_xlabel('Percentages')
        self.ax.set_title('Closest joint from gaze')
        plt.show()
        return self

    def timeseries_human_shape(self) -> Results:
        ''' This function represents the evolution between frames of the distance between the location of gaze and human shape.
        This function requires running the __init__ function first.'''
        self.dfHS = self.dataframe[self.dataframe['name'] == 'Human_Shape']
        self.fig = plt.figure()
        self.axHS = plt.axes()
        plt.title("Evolution of the distance between the gaze and human shape")
        plt.xlabel("Frame")
        plt.ylabel("Distance between gaze and human shape")
        self.axHS.plot(self.dfHS.frame, self.dfHS.distance, color="blue")
        plt.show()
        return self

    def piechart_human_shape(self) -> Results:
        ''' This function produces a pie chart corresponding to the pourcentages of frames where the gaze is located on or outside of human shape.
        This function requires running the __init__ function first.'''
        self.dfHS = self.dataframe[self.dataframe['name'] == 'Human_Shape']
        self.Human = self.dfHS[self.dfHS["distance"] == 0]
        self.Human_count = len(self.Human)
        self.Shape_Percentages = self.Human_count / len(self.dfHS)
        self.Shape_pie = [self.Shape_Percentages, 1 - self.Shape_Percentages]
        self.Shape_pie_rounded = np.round(self.Shape_pie, decimals=2)
        self.Shape_label = ["On human shape", "Outside"]
        self.fig = plt.figure()
        self.gaze_shape = plt.pie(self.Shape_pie_rounded, labels=self.Shape_pie_rounded)
        plt.legend(self.gaze_shape[0], self.Shape_label, bbox_to_anchor=(1.15, 0.5), loc="center right", fontsize=10,
                   bbox_transform=plt.gcf().transFigure)
        plt.title("Gaze distribution on human shape")
        plt.show()
        return self

    def timeseries_plot(self, joint: str) -> Results:
        ''' This function represents the evolution between frames of the distance between the location of gaze and the chosen joint with input.
        Possible inputs are these strings : "Left Ankle" ; "Right Ankle" ; "Left Elbow" ; "Right Elbow" ; "Neck" ; "Left Ear" ; "Right Ear" ; "Left Knee" ; "Right Knee" ; "Left Shoulder" ; "Right Shoulder" ; "Left Eye" ; "Right Eye" ; "Left Hip" ; "Right Hip" ; "Left Wrist" ; "Right Wrist". '''
        self.joint = joint
        self.dfjoint = self.dataframe[self.dataframe['joint'] == self.joint]
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.ax.plot(self.dfjoint.frame, self.dfjoint.distance, color="magenta")
        plt.title("Evolution of the distance between the gaze and the " + self.joint)
        plt.xlabel("Frame")
        plt.ylabel("Distance");
        return self

    def show(self) -> None:
        plt.show()

    def save(self, filename: str) -> Results:
        self.filename = filename
        self.fig.savefig(filename)
        return self
