from tddm import tqdm
from sklearn.model_selection import ParameterGrid
from custom_hierarchical_clustering import  Agglomerative_Hierachical_Clustering
from typing import Callable

class GridSearch :
    """
    """
    METRICS = [
        "ball_hall_score",
        "calisnki_harabasz_score",
        "hartigan_score",
        "silhouette_score",
        "xu_score"
        # Other metric to add
    ]
    
    def __init__(self, estimator: Agglomerative_Hierachical_Clustering,
                 param_grid : dict, 
                 scoring : Callable=None, 
                 n_job=None, verbose=None) -> None :
        """
        """
        self.estimator_ : Agglomerative_Hierachical_Clustering = estimator
        self.param_grid_ : dict = param_grid # a dictionary of parameters on which the model will be finetune
        self.best_estimator_ : Agglomerative_Hierachical_Clustering = self.estimator_ # At the beginning the best model will be the initial estimator with tyhe initizl parameter
        self.best_params_ : dict = None
        self.best_score_ : float = None # We will define later witch metric to use to evaluer the modèle
        self.score_ : Callable = scoring # 
        self.__verbose_ :int = verbose
        self.metrics : dict  = self.init_metric()
        self.n_clusters = list()
        
    
    def init_metric(self) -> dict :
        """
        """
        dict_metric = dict()
        
        for metric in self.METRICS :
            dict_metric[metric] = list()
            
        return dict_metric
    
    def fit(self, train_data : np.ndarray, compute_score=False) ->None :
        """
        """
        params = ParameterGrid(self.param_grid_)
        
        
        for item in tqdm(params):
            self.estimator_.set_params(**item)
            if compute_score == True :
                self.estimator_.fit(train_data, compute_score = compute_score)
                self.metrics["ball_hall_score"].append(self.estimator_.ball_hall_score)
                self.metrics["silhouette_score"].append(self.estimator_.silhouette_score)
                self.metrics["calisnki_harabasz_score"].append(self.estimator_.calisnki_harabasz_score)
                self.metrics["hartigan_score"].append(self.estimator_.hartigan_score)
                self.metrics["xu_score"].append(self.estimator_.xu_score)
                self.n_clusters.append(self.estimator_.n_cluster)