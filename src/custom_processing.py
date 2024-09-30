
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import random
import re
from utils import *
from deep_translator import GoogleTranslator
                             
class Custom_Processing:
    """
    A custom transformer class designed to apply a series of transformations on a given string dataset.
    """
   
    def __init__(self, 
                 drop_cols : list[str] =None, 
                 dataframe : DataFrame =None,
                 path : str = None,
                 # translate : bool = True, 
                 remove_city : bool = True, 
                 remove_country :bool = True, 
                 drop_numbers : bool = True,
                 duplicate_pattern : bool = True, 
                 drop_duplicates : bool = True,
                 concate_colum : bool = True
                )-> None:
        
        """
        Initializes the class with a DataFrame or a path to a CSV file, and a set of options for data processing.

        Args:
            drop_cols (list[str]): The list of column names to be dropped from the DataFrame. Default is None.
            dataframe (DataFrame): The DataFrame to be processed. Default is None.
            path (str): The path to a CSV file to be read into a DataFrame. Default is None.
            remove_city (bool): Whether to remove city information from the DataFrame. Default is True.
            remove_country (bool): Whether to remove country information from the DataFrame. Default is True.
            drop_numbers (bool): Whether to drop numbers from the DataFrame. Default is True.
            duplicate_pattern (bool): Whether to remove duplicate patterns from the DataFrame. Default is True.
            drop_duplicates (bool): Whether to drop duplicate rows from the DataFrame. Default is True.
            concate_colum (bool): Whether to concatenate columns in the DataFrame. Default is True.

        Returns:
            None

        Raises:
            ValueError: If both 'dataframe' and 'path' are not None.

        Note: This function initializes several attributes of the class, including various DataFrames for special character processing, figure content processing, and others.
        """
        if dataframe is not None and path is not None:
            raise ValueError("One and only one between dataframe and path must be not None")
        
        self.dataframe = dataframe.copy(deep=True) if dataframe is not None else pd.read_csv(path)
        self.special_char_dataframe = None # Comment to add
        # self.special_char_trans_dataframe = None # Comment to add
        self.cont_figure_dataframe = None # comment to add
        self.others_dataframe = None # comment to ad
        self.duplicates_dataframe = None
        
        self.drop_cols = drop_cols
        # self.translate = translate
        self.remove_city = remove_city
        self.remove_country = remove_country
        self.drop_numbers = drop_numbers
        self.duplicate_pattern = duplicate_pattern
        self.drop_duplicates = drop_duplicates
        self.concat_col = concate_colum
        
    
    def special_char_processing(self, special_char_cols : list[str] ) -> None :
        
        """
        Processes the DataFrame of the class by filtering rows that contain special characters in specified columns.

        Args:
            special_char_cols (list[str]): The list of column names in the DataFrame that need to be checked for special characters.

        Returns:
            None

        Note: This function updates the class's DataFrame and special_char_dataframe attributes in place.
        """
        patterns = REX
        
        
        for col in tqdm(special_char_cols) : #Filter row that contain some given patterns 
            special = self.dataframe[self.dataframe[col].astype(str).str.contains(patterns)]
            self.special_char_dataframe= pd.concat((self.special_char_dataframe,special))
            
        self.dataframe =self.dataframe.drop(index=self.special_char_dataframe.index)
            
 
    def drop_cols_fig(self, cols_cont_fig : list[str]) -> None :
        """
        If self.dro_nmbers = True then all tiern name containing some figure will be dropped from the original dataframe
        The flitered dataframe containing only tiern name with with figures will be an attribute of this class
        
        Args :
            col_cont_fig(str) : The colunm on which one base to filter the tiern ame containing some figure.
            
        Returns :
            None
        """
        
        if self.drop_numbers == True :
            for col in tqdm(cols_cont_fig) : 
                cont_figures = self.dataframe[self.dataframe[col].astype(str).str.contains(r'\d+')]
                self.dataframe = self.dataframe.drop(index=cont_figures.index)
                self.cont_figure_dataframe = pd.concat((self.cont_figure_dataframe,cont_figures))
        
    def fit_transform(self,country="Country", 
                      city="tiern_location_state_city",
                      concat_col ="tiern_name_preprocessed", 
                      cols_cont_fig=["tiern_name_preprocessed"],
                      # special_char_cols=["tiern_name_preprocessed", "tiern_plant","tiern_name"],
                      special_char_cols=["tiern_name_preprocessed"],
                      cols_to_concat=["tiern_plant","tiern_name"],
                      
                     ) -> None :
        """
        Performs a series of transformations on the class's DataFrame. These transformations include dropping specified columns,
        processing strings, concatenating specified columns into a new column, dropping figures from specified columns, 
        processing special characters, and applying the functions 'concat_sub_str' and 'remove_blank_space' to all elements of the DataFrame.

        Args:
            country (str): The name of the country column in the DataFrame. Default is "Country".
            city (str): The name of the city column in the DataFrame. Default is "tiern_location_state_city".
            concat_col (str): The name of the new column to be created by concatenating the columns specified in 'cols_to_concat'. Default is "tiern_name_preprocessed".
            cols_cont_fig (list[str]): The list of column names in the DataFrame from which figures should be dropped. Default is ["tiern_name_preprocessed"].
            special_char_cols (list[str]): The list of column names in the DataFrame that need to be checked for special characters. Default is ["tiern_name_preprocessed"].
            cols_to_concat (list[str]): The list of column names in the DataFrame that should be concatenated to form a new column. Default is ["tiern_plant","tiern_name"].

        Returns:
            None

        Note: This function updates the class's DataFrame attribute in place.
        """
        
        if self.drop_cols is not None: # We want to drop some column from the original dataframe. If len(self.drop_cols), it means that
                                   # There no provided colu
            self.dataframe =self.dataframe.drop(self.drop_cols, axis=1)
        
        self.dataframe = process_dataframe_str(self.dataframe) # Basic processing (i.e string formating, NAN processing ,...)
        
        if self.concat_col :
            self.dataframe[concat_col] = self.dataframe.apply(lambda row: " ".join(row[col] for col in cols_to_concat), axis=1)
            
            # self.dataframe[new_col] = self.dataframe.apply(lambda row: row[col1] + " " + row[col2], axis=1)
        
        self.drop_cols_fig(cols_cont_fig)
        self.special_char_processing(special_char_cols)
        
        self.dataframe=self.dataframe.map(concat_sub_str)
        # self.dataframe=self.dataframe.applymap(concat_sub_str)
        # self.dataframe=self.dataframe.applymap(remove_blank_space)
        self.dataframe=self.dataframe.map(remove_blank_space)
        
        
        
        
        if self.remove_city ==True:
            self.dataframe[concat_col] = self.dataframe.apply(lambda row: remove_pattern(row[city], row[concat_col]), axis=1)
            
        if self.remove_country ==True:
            self.dataframe[concat_col] = self.dataframe.apply(lambda row: remove_pattern(row[country], row[concat_col]), axis=1)
            
        if self.duplicate_pattern ==True:
           
            self.dataframe[concat_col]=self.dataframe[concat_col].map(remove_duplication)
        
        self.dataframe["chars"] = self.dataframe[concat_col].map(lambda x : x.replace(" ",""))
        
        self.ref_dataframe = self.dataframe.copy(deep=True)
        
        
                                        
        
        if self.drop_duplicates ==True :
            
            tmp = self.dataframe.drop_duplicates([city, country, "chars"])
            self.duplicates_dataframe = self.dataframe.drop(index=tmp.index)
            self.dataframe.drop_duplicates([city, country, "chars"], inplace=True)
        
        self.others_dataframe = pd.concat((self.others_dataframe, self.dataframe.loc[self.dataframe["chars"].map(len) <=1, :]))
        self.dataframe.drop(index=self.dataframe.loc[self.dataframe["chars"].map(len) <=1, :].index, inplace=True)
        self.others_dataframe = pd.concat((self.others_dataframe, self.dataframe.groupby([city, country]).filter(lambda x : len(x) <=1)))
        self.dataframe.drop(index=self.dataframe.groupby([city, country]).filter(lambda x : len(x) <=1).index, inplace=True)
        
        #fianl processing 
        
        # self.cont_figure_dataframe = self.cont_figure_dataframe.rename(columns = {"tiern_name_preprocessed" : "suggested_name"})
        self.cont_figure_dataframe["category"] = "contain_figure"
        
        # self.duplicates_dataframe = self.duplicates_dataframe.rename(columns = {"tiern_name_preprocessed" : "suggested_name"})
        self.duplicates_dataframe["category"] = "duplicates"
        self.duplicates_dataframe.drop(columns=["chars"], inplace = True)
        
        # self.others_dataframe = self.others_dataframe.rename(columns = {"tiern_name_preprocessed" : "suggested_name"})
        self.others_dataframe["category"] = "other"
        self.others_dataframe.drop(columns=["chars"], inplace = True)
        