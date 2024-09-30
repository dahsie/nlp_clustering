import sys

sys.path.append("/home/jupyter/detection_doublons/src") # for adding the directory "src"
from mesures import *
from custom_processing import *

import pickle

import copy
import re

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import ParameterGrid



import numpy as np
import pandas as pd


from typing import Callable, Tuple, Union
from time import time
from tqdm import tqdm


import math


class Agglomerative_Hierachical_Clustering :
    """
    This class is a custum Agglomerativ Hierachical clustering. In fact, we want to use this class for clustering à raw string 
    with a specific metric ( pairewise distance) with turning them to numerical data. 
    Here, we use use pricisely Levenshtein, Damerau-Levenshtein and Jaccard distance. 
    """
    
    # Those metrics will be used to train the model. they will be use when a string  is specify as metric.
    #For exemple Agglomerative_Hierachical_Clustering(metric = "damerau_levenshtein", *args)
    METRICS = {
        "levenshtein_distance": levenshtein_distance,
        "jaccard_similarity": jaccard_similarity,
        "damerau_levenshtein_distance": damerau_levenshtein_distance,
        # Add others metrics
        }
    
    # Dictionary contain exactly the same metrics as the previous except that hthe metric is specify as a callable 
    #For exemple Agglomerative_Hierachical_Clustering(metric = damerau_levenshtein, *args)
    INV_METRICS = {value: key for key, value in METRICS.items()}
    
    #The differents linkage whitch will be use during clustring. This is used when a linkage is specified as a string 
    # For exemple Agglomerative_Hierachical_Clustering(linkage = "single_linkage", *args)
    LINKAGES ={
        "single" :single_linkage,
        "average" : average_linkage,
        "complete" : complete_linkage
        #Other linkage strategie to add
    }
    
    #This liste of linkage is used when the linkage parameter is a callable and not a string. 
    #For exemple Agglomerative_Hierachical_Clustering(linkage = single_linkage, *args)
    INV_LINKAGES ={ value : key for key, value in LINKAGES.items()}
    
    def __init__(self, distance_threshold : Union[float,int]=None, n_cluster: int =None, metric :Union[Callable[[str, str, bool], float], str]=damerau_levenshtein_distance, normalize_metric=False, linkage  ="average") -> None:
        """
        This method is like a constructor. It allow to initialize the object attributes.
        
        Args :
            
            distance_threshold(float or int) : This is the distance threshold above which clusters will not be merge to form a new cluster
            
            metric(Callable , flaot] or str): Use to compute pairewise distance between two samples but also to compute the linkage. Then it will be use to compute pairewise distance between 
            all sample or to make the linkage in order to determine to clossest clusters. For the linkage, it will be used as parameter to
            self.linkage (self.linkage(str1, str2, self.metric)
            
            linkage(str or callable) : The type of linkage to use. In our case, we have only three linkage : 'single', 'average' or 'complete'
            
            normalize_metric(bool) : to specify if the metric is a normalized one. if so, then the distance_threshold should be beteen 0 and 1 and not above.
            The distances are then turned into a dissimlaraty or similarity. 
        
        Returns :
            
            None
            
        """
        super().__init__()
        
       
        self.distance_threshold = distance_threshold
        self.initial_n_cluster = n_cluster
        self.normalize_metric= normalize_metric
        self.set_metric(metric)
        self.set_linkage(linkage)
        self.cities = {} 
        self.doublons_dataframe =None # This dataframe contains dataframe with labeled duplictes.
                                      #That is to say, if two data points are identified by
                                      # the algorithm as duplicates, they will have the same "duplication_id"
                                      # This dataframe will be created at the end of the all training process
        self.__reset() # 
        
    
    def __reset(self) -> None:
        """
        This model (self) will be use to train difference data groupe by city without creating a new model for each city data.
        So this method will allow to reinitialize some parameters for each city.
        """
        
        self.n_cluster = self.initial_n_cluster
        self.clusters = None
        self.distance_matrix = None
        self.labels = None
        self.is_fitted = False
        self.compute_score =False
        
    
    def get_linkage(self) -> str :
        """
        This method return a string as the model linkage mothod. 
        It will return "single", "average" or "complete". In the case the model is saved, when will load it,
        one will probably need to get a linkage parameter and we should be able to return a string and not a callable
        given that the linkage parameter can be either a str or a callable.
        
        Return :
            str : return the linkage name used as the model parameter
        """
        
        if isinstance(self.linkage , str):
            return self.linkage
        
        elif callable(self.metric) :
            return self.INV_LINKAGES[self.linkage]
        
    def get_metric(self) -> str:
        
        """
        This method return a string as the model metric mothod. 
        It will return "levenshtein", "damerau_leveinshtein" or "jaccard". In the case the model is saved, when will load it,
        one will probably need to get a metric parameter and we should be able to return a string and not a callable
        given that the metric parameter can be either a str or a callable.
        
        Return :
            str : the name of the metric used as model parameter
        """
        if isinstance(self.metric , str):
            return self.metric
        
        elif callable(self.metric) :
            return self.INV_METRICS[self.metric]
            
    def get_params(self) -> dict :
        """
        The method return a dictionary containing all the model parameters in a string format
        
        Returns :
            dict : All the model parameters in a string format
        """
        
        linkage = self.get_linkage()
        metric = self.get_metric()
        
        return {"distance_threshold" : self.distance_threshold, 
                "n_cluster" : self.n_cluster, 
                "linkage" : linkage,
                "normalize_metric" : self.normalize_metric,
                "metric" :metric }
    
    
    def set_linkage(self, linkage) -> None :
        """
        This method use to change the linkage parameter without creating a new model. 
        
        Args :
            linkage( str or callable) : the new parameter to be used by the model
        
        Return :
            None
        
        Raises :
            TypeError: If the provided 'linkage' is not a callable or a string among valid options for linkage.

        Example:
            >>> model = Agglomerative_Hierachical_Clustering(*args)
            >>> model.set_metric('damerau_levenshtein')
            >>> model.get_metric()
            'damerau_levenshtein'

        Note:
            The valid options for 'metric' are the keys of the 'METRICS' dictionary in the class.
        """
        if isinstance(linkage, str) and linkage in self.LINKAGES:
            self.linkage = self.LINKAGES[linkage]
            
        elif callable(linkage) and linkage in self.INV_LINKAGES:
            self.linkage = linkage
            
        else:
            raise TypeError(f"The parameter 'linkage' must be a callable or a string among valid options linkage. The valid options are {list(self.LINKAGES.keys())}")
            
    def set_metric(self, metric) -> None:
        """
        This method use to change the likage parameter without creating a new model. 
        
        Args :
            linkage( str or callable) : the new parameter to be used by the model
        
        Return :
            None
        
        Raises :
            TypeError: If the provided 'metric' is not a callable or a string among valid options for metric.

        Example:
            >>> model = Agglomerative_Hierachical_Clustering(*args)
            >>> model.set_linkage('single')
            >>> model.get_linkage()
            'single'

        Note:
            The valid options for 'metric' are the keys of the 'METRICS' dictionary in the class.
        """
        
        if isinstance(metric, str) and metric in self.METRICS:
            self.metric = self.METRICS[metric]
            
        elif callable(metric) and metric in self.INV_METRICS:
            self.metric = metric
            
        else:
            raise TypeError(f"The parameter 'metric' must be a callable or a string among valid options metrics. The valid options are {list(self.METRICS.keys())}")
            
    def set_params(self, **params: dict) -> None:
        """This methd is use to chanage all the model parameters without instanciating a new model
        
        Args :
            **params(dict): a dictionary which conatain all the model new parameters
        
        Returns :
            None
        
        Raise :
            ValueError : If there is an invalid key in the dictionary.
        """
        
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' found.")

            if key == "metric":
                self.set_metric(value)
                
            elif key == "linkage" :
                self.set_linkage(value)
                
            else:
                setattr(self, key, value)

        
    def __clusters(self, data  : np.ndarray) ->dict:
        """
        """
        dict_={}
        
        # bh_mean=0.0
        # distance_matrix = string_pairwise_distance(data, metric= self.metric,normalize_metric=False)
        # _, bh1= intra_class_distance(distance_matrix, self.labels)
        
        for i in range(len(np.unique(self.labels))):
            dataa={}
            dataa["members"] = data[self.labels==i].squeeze(1).tolist()
            
            min_value, mean_value, max_value = 0.0, 0.0, 0.0
            # bh=0.0
            
            
            if len(dataa["members"]) > 1: # Compute pairwaise distance bettween data in in the same
                                          # cluster if only if the cluster contains morethan one data point.
                                          # Else, there is no need for computing the pairwise distance
                        
                pairewise_distance=string_pairwise_distance(data[self.labels==i], metric= self.metric,normalize_metric=self.normalize_metric)
                
            
                upper_indices=np.triu_indices(len(pairewise_distance), k=1)
                min_value=np.min(pairewise_distance[upper_indices])
                max_value=np.max(pairewise_distance[upper_indices])
                mean_value=np.mean(pairewise_distance[upper_indices])
            
            dataa["distance"]= {"min" : round(min_value, 3), "average" : round(mean_value, 3), "max" :round(max_value, 3)}
            dataa["confidence"]= round(1- mean_value, 3)
            
            dict_[i] = dataa
        
        return dict_ #, round(bh_mean/len(np.unique(self.labels)), 3), bh1
        # return {i: data[self.labels==i].squeeze(1).tolist() for i in range(len(np.unique(self.labels)))}
    
        
    def __find_closest_clusters(self) -> Tuple[float, int, int]:
        """
        Find two closest clusters, we only process two closest clusters at each iteration and not beyond.
        Notice there can be more than one minimum value regarded the metric distance we use in our case (jaccard distance, levenshtein distance or damerau-Levenshtein distance)
        
        Args : 
            None
        
        Reurns :
            float, int, int : those values are respectively the minimun values, the row index and the column index of this minimum value
        """
        
        idx_upper = np.triu_indices(len(self.distance_matrix), k=1)  # Index of upper part of self.distance_matrix (skip diagonal)
        min_value = np.min(self.distance_matrix[idx_upper])  # Value of idx_upper
        min_value_index = np.where(self.distance_matrix == min_value) # Get all minimum values index
        row, col = min_value_index[0][0], min_value_index[1][0] # The raw and column of minimum value
        
        return min_value, row, col
    
            
    
    def __update(self, row : str, col : str,data : np.ndarray) -> None:
        """
        This method use the index(row index and column index) of minimun value to update the pairwise matrix distance
        of cluster. Two merged clusters should have the same label, so the labels are update in this method.
        Merging two clusters means labelling all points in those two clusters with the same value. 
        Then the row and column corresponding to the minimum value in the pairwaise matrix distance are deleted. 
        Finally the pairwise distance between the new cluster and all the remaining clusters are computed. 
        
        Args :
            row(int) : row index of the minimum value of the pairwise matrix distance between clusters;
            col(int) : column index of the minimum value of the pairwise matrix distance between clusters;
            data(np.ndarray) : the whole training data set. It will be used the get the data whithin a cluster
                               in order to compute the distance between this new cluster and the remaining clusters.
            
        Returns :
            None
        """
        
        # Updating label
        self.labels[self.labels == col] = row
        self.labels[self.labels > col] -= 1

        # Deleting the row and column 'col'  
        self.distance_matrix = np.delete(self.distance_matrix, col, 0) # Deleting row "col"
        self.distance_matrix = np.delete(self.distance_matrix, col, 1) # Deleting column 'col'

        # Updating self.distance_matrix
        for i in range(len(self.distance_matrix)):
            if row==i: # Distance betwen the cluster and itself is null so no need to compute it.
                self.distance_matrix[row, i] = 0.
                self.distance_matrix[i, row] = 0.
            else : # Compute only distance between the new cluster and all remaining clusters (note that a single data point is condered as a cluster itself)
 
                self.distance_matrix[row, i] = self.linkage(data[self.labels == row], data[self.labels == i], metric= self.metric,normalize_metric=self.normalize_metric)
                self.distance_matrix[i, row] =  self.distance_matrix[row, i] # the distance matrix is symmetric

    def fit(self, data : np.ndarray, compute_score : bool = False) -> None:
    
        """This method are use to train the model (self). 
        
        Args :
            data(np.ndarray) : The training data set. 
            compute_score(bool) : If True then the all scores will be compute after training the model.
                                  If False then None metric will be compute after training the model. 
        Returns :
            None
        Raise :
            ValueError : If the training data type is not an array-like( numpy array )
                         If normalize_distance is set to True and then the distance threshold is not between 0.0 and 1.0
                         If the self.distance_threshold and self.n_cluster are both not None. One and only one shouls be None
                         If traininf data contain only one element. No need to train a model with a one data point

        """
        if not isinstance(data, np.ndarray):
            raise ValueError("input data should be a numpy array")
            
        if self.normalize_metric and (self.distance_threshold < 0.0 or self.distance_threshold >= 1.0) :
            raise ValueError("The distance threshold must be between 0 and 1 beacause all distances are normalized and then lie between 0 and 1")
        
        if self.distance_threshold is None  and self.n_cluster is None :
            raise ValueError("distance_threshold and n_cluster should not be both None. One of them shoud be specified")
            
        if len(data) <=1:
            raise ValueError("Insufficient data. You need more than one data point to train the model.")
            
        data.sort(axis=0)
        
        self.distance_matrix = string_pairwise_distance(data, metric=self.metric,normalize_metric=self.normalize_metric, n_job=-1) # Computing the first pairwise matrix distance between all cluster.
                                                                                             #Notice that, at the begining, single data point represent a cluster
        # self.initial_distance_matrix = copy.deepcopy(self.distance_matrix)
        self.labels = np.arange(len(data)) # Labels initialisation
    
        # print("Training ...")  
        if self.distance_threshold :
            while True : # Loop until the minimum distance between two clusters is equal or above the threshold distance
                
                if len(self.distance_matrix) <=2 : # if we make a clustering, it is because we want
                    break                            # to have at least two clusters
                      
                min_value, row, col = self.__find_closest_clusters()
                
                if min_value >= self.distance_threshold :
                        break
                self.__update(row, col, data)
                
        elif self.n_cluster :
        
            while len(np.unique(self.labels)) > self.n_cluster : # Loop until the number of desired clusters is reached
                
                if len(self.distance_matrix) <=2:
                    break
                print(self.distance_matrix.size.shape)
                min_value, row, col=self.__find_closest_clusters()
                self.__update(row, col, data)
        
        self.clusters =self.__clusters(data) # Gather the all points which have the same cluster.
        self.n_cluster = len(np.unique(self.labels))
        self.is_fitted=True
        
        # print("Training ended !")
        # Computing differents scores aftrer training the model
        if compute_score :
            self.initial_distance_matrix =string_pairwise_distance(data, metric=self.metric,normalize_metric=False, n_job=-1)
            # print("computing_score ...")
            intra_class, intra_mean = intra_class_distance(self.initial_distance_matrix, self.labels)
            interclass, inter_mean = inter_class_distance(self.initial_distance_matrix , self.labels)

            self.silhouette_score, _= Silhouette_score(matrix_distance=self.initial_distance_matrix, labels=self.labels)
            self.ball_hall_score =intra_mean
            self.calisnki_harabasz_score = Calisnki_Harabasz_score(inter_mean, intra_mean,self.n_cluster)  
            self.hartigan_score = Hartigan_score(inter_mean, intra_mean)
            self.xu_score = Xu_score(intra_mean, len(data), self.n_cluster, D =1)
        
    
    def print_clusters(self, distance : str= "average") -> dict :
        """This method is used to print the clusters having more than one data point after the model is trained
        
        Args :
            distance(str) : Allow printing the distance between data points within a each cluster. It can be the average,
                            minimun or maximun distance between cluster members
        Returns :
            dict : Retrun also a dictionary containing all cluster with none single value
        
        Raise :
            ValueError : If this method is called without the model being trained
        """
        if self.is_fitted ==False:
            raise ValueError("The model is not yet trained, we can't print any cluster")
        
        none_uniq_cluster ={}
        for key in list(self.clusters.keys()):
            if len(self.clusters[key]["members"]) > 1 :
                print(f" members : {model.clusters[key]['members']} -- {distance}_distance : {self.clusters[key]['distance']['average']}")
                print()
                none_uniq_cluster[key] = model.clusters[key]['members']
        return none_uniq_cluster
    
            
    def _process_city(self, country : str, city: str, dataframe : DataFrame)-> Tuple[str,str, dict, np.array]:
        """
        This method i use to train the model given a country and a city.
        This method allows you to train the modelon each city in a parallel way given taht each city is processed independently.
        Args :
            country(str) : The country of the processing city. In fact a city acn be found in differents country.
            city(str) : The processing city.
            dataframe(pandas.DataFrame) : The whole training data set on which all data points having the same city and country will be select as the training data set of a city.
        
        Returns :
            str, str, dict, np.ndarray : country, city, all formed clusters and all labels for the selected data set base on the city and the country
        """
        
        train_data=np.unique(dataframe.loc[(dataframe["tiern_location_state_city"]==city) & (dataframe["Country"]==country),"tiern_name_preprocessed"].to_numpy()).reshape((-1,1)).astype(str)

        self.__reset()
        self.fit(train_data)
        
        return country,city, self.clusters, self.labels
        
        
    def fit_per_city(self, dataframe : DataFrame, path1 : str, path2 : str) -> None:
        """
        In this method, the self._process_city(*args) will be used to parallelize the training process. 
        All clusters with more than one data points will be saved into a pandas DataFrame with all related information
        get from the training process, for exemple the similarity scorr, city, country, tiern_name, tiern_plant of each data points
        whithin a given cluster. 
        
        Args :
        
            dataframe : The training dataframe
            path : Where to strore the training result. All training data points are save to this dataframe.
        
        Returns :
            None
            
        """
        # columns=list(dataframe.columns) + list(["duplication_id"]) +list(["similarity"])
        columns=["country", "city", "tiern_name", "tiern_plant","tiern_name_preprocessed","duplication_id",
                 # "similarity"
                ]
        sing_columns=["country", "city", "tiern_name", "tiern_plant","tiern_name_preprocessed"]
        self.doublons_dataframe =pd.DataFrame(columns=columns) # Creating the duplicates dataframe for the first
        self.singles_dataframe =pd.DataFrame(columns=sing_columns) # Creating the duplicates dataframe for the first
        
        # city_list=dataframe["tiern_location_state_city"].unique().tolist()
        df = dataframe[["Country", "tiern_location_state_city"]].drop_duplicates()
        couples = list(df.itertuples(index=False, name=None))

        results = Parallel(n_jobs=-1)(delayed(self._process_city)(country,city,dataframe) for country, city in tqdm(couples))

        # Filter out None results and update self.cities
        prev_city_mid = 1 # privious processed city maximum duplication id. This will be use as the starting pour of the duplication id of the next city.

        for result in tqdm(results):
           
            # if result is not None:
            country,city, clusters, labels= result

            key = country + " " +city
            self.cities[key] = clusters
             # self.cities[city]={"clusters":clusters,
                                 # "labels" : labels}
            for key in list(clusters.keys()):

                proc_names = clusters[key]["members"]
                n_members =len(clusters[key]["members"])
                dat=dataframe.loc[(dataframe["tiern_location_state_city"]==city) & (dataframe["Country"]==country) & (dataframe['tiern_name_preprocessed'].isin(proc_names)), ["tiern_name", "tiern_plant", "tiern_name_preprocessed"]].sort_values(by= "tiern_name_preprocessed")

                # try:
                new_dataframe=pd.DataFrame({ 'country' :[country] * n_members,
                                            'city' : [city] * n_members,
                                            'tiern_name' :dat['tiern_name'].to_list(),
                                            'tiern_plant' : dat['tiern_plant'].to_list(),
                                            'tiern_name_preprocessed' :clusters[key]["members"]                        
                })
               
                if len(clusters[key]["members"]) > 1 :
                    # new_dataframe['similarity'] = [clusters[key]["confidence"]] * n_members
                    new_dataframe['duplication_id'] = [prev_city_mid] * n_members
                    self.doublons_dataframe=pd.concat([self.doublons_dataframe,new_dataframe], axis=0)
                    prev_city_mid+=1
                else :
                    self.singles_dataframe = pd.concat([self.singles_dataframe,new_dataframe], axis=0)



        self.singles_dataframe["suggested_name"] = self.singles_dataframe['tiern_name_preprocessed'] 
        self.doublons_dataframe.to_csv(path1, index=False)
        self.singles_dataframe.to_csv(path2, index=False)