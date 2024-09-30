
from typing import Callable, Tuple, Union
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

"""
This file implement all metric that will be use for hour custom hierachical clustering
"""



def single_linkage(ci :np.ndarray,
                   cj : np.ndarray,  
                   metric :Callable[[str, str, bool], float],
                   normalize_metric : bool =False
                  ) -> float:
        """
        Compute distance between each data point in cluster ci(first cluster) and all data points in cluster cj (second cluster)
        and then return the minimum value among all those computed distance. 

        Args :
            ci : The first cluster 
            cj : The second cluster 

        Returns :
            float : Minimum distance among all distances between each point in ci and each point in cj

        """
        return min([metric(ci[i,0], cj[j,0],normalize_metric) for i in range(ci.shape[0]) for j in range(cj.shape[0])])

def complete_linkage(ci :np.ndarray, 
                     cj : np.ndarray, 
                     metric :Callable[[str, str,bool], float],
                     normalize_metric : bool =False
                    ) -> float:
    """
    Compute distance between each data point in cluster ci(first cluster) and all data points in cluster cj (second cluster)
    and then return the maximum value among all those computed distance. 

    Args :
        ci : The first cluster 
        cj : The second cluster 

    Returns :
        float : maximum distance among all distances between each point in ci and each point in cj

    """
    return max([metric(ci[i,0], cj[j,0],normalize_metric) for i in range(ci.shape[0]) for j in range(cj.shape[0])]) 

def average_linkage(ci :np.ndarray, 
                    cj : np.ndarray, 
                    metric : Callable[[str, str, bool], float],
                    normalize_metric : bool = False
                   ) -> float:
        """
        Compute distance between each data point in cluster ci(first cluster) and all data points in cluster cj (second cluster)
        and then return the average distance of all those computed distance. 

        Args :
            ci : The first cluster 
            cj : The second cluster 

        Returns :
            flaot : average distance distance between each point in ci and each point in cj

        """

        distances = [metric(ci[i,0], cj[j,0],normalize_metric) for i in range(ci.shape[0]) for j in range(cj.shape[0])]

        return sum(distances) / len(distances)
def jaccard_similarity(str1 : str, str2 : str) -> float:
    """La fonction suivante permet de calucler la distance de jaccard entre deux strings
    La distance de jaccard entre deux string est : 1 - (nombre de caractères en commun des deux chaînes)/(le nombre
    de caractère utilisé pour former les deux chaînes
    
    Args :
        str1(str) : La premier chaîne de caractères
        str2( str) : La deuxième chaîne de caractères
    
    Returns :
        float : la distance de jaccard entre deux chaînes de caractères en paramètres
        
    Exemples :
        >> str1 = pair
        >> str2 = sapin
        Les lettres de str1 sont [a,i,p,r], celles str2 sont [a,i,n,p,s]. Les lettres communes
        sont donc [a,i,p], il y en a donc 3. Toutes les lettres sont [a,i,n,p,r,s], il y en a donc 6.
        La distance de Jaccard entre str1 et str2 est donc : d= 1 - 3/6 =1  -0.5 = 0.5
    """
    
    set1 = set(str1)
    set2 = set(str2)
    return 1.0 - len(set1 & set2) / len(set1 | set2)


def hamming_distance(str1 : str, str2 : str) -> float:
    """La fonction suivante permet de calucler la distance de Hamming entre deux strings.
    La distance est calculée entre deux mots de même longueur. C’est le nombre d’endroits où les lettres sont différentes.

    Args :
        str1(str) : La premier chaîne de caractères
        str2( str) : La deuxième chaîne de caractères

    Returns :
        float : la distance de Hamming entre deux chaînes de caractères en paramètres

    Exemples :
        >> str1 = pair
        >> str2 = sapin
        Les lettres de str1 sont [a,i,p,r], celles str2 sont [a,i,n,p,s]. Les lettres communes
        sont donc [a,i,p], il y en a donc 3. Toutes les lettres sont [a,i,n,p,r,s], il y en a donc 6.
        La distance de Jaccard entre str1 et str2 est donc : d= 1 - 3/6 =1  -0.5 = 0.5
    """
    if len(str1) != len(str2):
        raise ValueError("Les chaînes doivent avoir la même longueur")
    
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def levenshtein_distance(str1 : str, str2 : str, normalize_metric : bool=False) -> float:
    """On définit trois opérations qui permettent de passer d’un mot à un autre :
    1. suppression d’une lettre,
    2. ajout d’une lettre,
    3. remplacement d’une lettre.
    Voici un exemple de chaque type :
    1. PLANTE vers PLATE (la lettre N est supprimée),
    2. RAPE vers RAMPE (la lettre M est ajoutée),
    3. RAMER vers RALER (la lettre M est remplacée par la lettre L).
    La distance de Levenshtein entre deux mots est le nombre minimum d’opérations à effectuer afin
    de passer du premier mot au second.
    
    Args :
        str1(str) : La premier chaîne de caractères
        str2( str) : La deuxième chaîne de caractères

    Returns :
        int : nombre minimale d'opérations à effecturer pour passer de str1 à str2

    
    """
    n = len(str1) + 1
    m = len(str2) + 1
    matrix = np.zeros((n, m))
    
    for i in range(n):
        matrix[i,0] = i
        
    for j in range(m):
        matrix[0,j] = j

    for i in range(1, n):
        for j in range(1, m):
            if str1[i-1] == str2[j-1]:
                matrix[i,j] = min(matrix[i-1,j] + 1, matrix[i-1,j-1], matrix[i,j-1] + 1)
            else:
                matrix[i,j] = min(matrix[i-1,j] + 1, matrix[i-1,j-1] + 1, matrix[i,j-1] + 1)
    
    # if normalize ==True :
    if normalize_metric :
        max_len = max(len(set(str1)),len(set(str2)))
        return matrix[n-1, m-1] /max_len
    
    return matrix[n - 1, m - 1]


def damerau_levenshtein_distance(str1 : str, str2 : str, normalize_metric : bool =False) -> float:
    """
    On définit quatre opérations qui permettent de passer d’un mot à un autre :
    1. suppression d’une lettre,
    2. ajout d’une lettre,
    3. remplacement d’une lettre,
    4. transposition de deux lettres adjacentes.
    
    Args :
        str1(str) : La premier chaîne de caractères
        str2( str) : La deuxième chaîne de caractères

    Returns :
        int : nombre minimale d'opérations à effecturer pour passer de str1 à str2

    """
    n = len(str1) + 1
    m = len(str2) + 1
    matrix = np.zeros((n, m))

    for i in range(n):
        matrix[i,0] = i
        
    for j in range(m):
        matrix[0,j] = j

    for i in range(1, n):
        for j in range(1, m):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i,j] = min(
                matrix[i - 1, j] + 1,  # deletion
                matrix[i, j - 1] + 1,  # insertion
                matrix[i - 1, j - 1] + cost,  # substitution
            )
            if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
                matrix[i, j] = min(matrix[i, j], matrix[i - 2, j - 2] + cost)  # transposition
    if normalize_metric :
        set1=set(str1)
        set2=set(str2)
        max_len = max(len(set1),len(set2))
        return matrix[n-1, m-1] /max_len 
    return matrix[n-1, m-1]


def string_pairwise_distance(strings : np.ndarray, 
                             metric : Callable[[str,str,bool], float],
                             normalize_metric :bool = False,
                             n_job=None
                            ) -> np.ndarray:
    """Calcul de distance deux à deux des éléments du tableau de strings
    
    Args :
        strings (list[str]) : Une liste de string dont on veut calculer leurs distance deux à deux
        metric : Métrique pour calculer la distance entre des points de données (jaccard_distance,distance_Levenshtein, ...)
    
    Returns :
        numpy.array : la matrice de distance entre les points de données dans strings
    """
    assert isinstance(normalize_metric, bool), "normalize_metric should be a boolean"
    n = len(strings)
    result = np.zeros((n, n))
    
#     if isinstance(strings, list) :
#         strings=np.ndarray(strings).reshape((-1,1)).astype(str)
        
    if n_job is None: # A sequential programm 
        for i in range(n):
            for j in range(i+1, n):
                result[i, j] = metric(strings[i,0], strings[j,0],normalize_metric)
                result[j, i] = result[i, j]
                
    else : # Pallelize the distance computing
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        distances = Parallel(n_jobs=n_job)(delayed(metric)(strings[i,0], strings[j,0],normalize_metric) for i, j in pairs)
        for (i, j), distance in zip(pairs, distances):
            result[i, j] = distance
            result[j, i] = distance
            
    if not np.array_equal(result, np.transpose(result)) :
        raise ValueError("The distance matrix is not symetric.")
        
    if not np.all(result[np.eye(n)==0] != 0) : 
        raise ValueError("The distance matrix contains null values outside the diagonal.")
        
    return result


def Silhouette_score(data : np.ndarray=None, labels : np.ndarray=None, matrix_distance = None,metric : str =None, normalize_metric=False):
    """
    Calculate the silhouette score for clustering results.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        Input data.
    labels : array-like, shape (n_samples,)
        Predicted cluster labels.

    Returns:
        float: silhouette_avg 
        Mean silhouette coefficient for all samples.
    silhouette_values : array-like, shape (n_samples,)
        Silhouette score for each sample.
    """
    n_samples = data.shape[0] if data else matrix_distance.shape[0]
    silhouette_values = np.zeros(n_samples)
    
    if labels is None :
        raise ValueError("label should not be None")
    
    
    # Calculating pairwise matrix distances for all samples
    
    if matrix_distance is None:
        
        if metric =='jaccard' :
            matrix_distance = string_pairwise_distance(data, metric = jaccard_distance, normalize_metric=normalize_metric)
        elif metric =='levenshtein' :
            matrix_distance = string_pairwise_distance(data, metric = levenshtein_distance, normalize_metric=normalize_metric)
        elif metric =='damerau_levenshtein' :
            matrix_distance = string_pairwise_distance(data, metric = damerau_levenshtein_distance, normalize_metric=normalize_metric)
        else : 
            raise Exception("matrix distance and metric are boath None, One of them sould be provided")
            
    
    # Compute silhouette score for each sample
    for i in range(n_samples):
        cluster_label = labels[i]
        cluster_points = np.where(labels == cluster_label)[0]
        a=0.0
        # if len(cluster_points) <= 1 :
        #     # silhouette_values[i] =0.0
        #     # continue
        #     a=0.0
        if  len(cluster_points) <= 1: 
            silhouette_values[i]=0
            continue
            
        a = np.mean([matrix_distance[i, j] for j in cluster_points if j != i])
       
        b_values = []
        for other_label in np.unique(labels):
            if other_label != cluster_label:
                other_cluster_points = np.where(labels == other_label)[0]            
                b_values.append(np.mean([matrix_distance[i, j] for j in other_cluster_points]))

        b = min(b_values)

        silhouette_values[i] = (b - a) / max(a, b)

    silhouette_avg = np.mean(silhouette_values)
    return silhouette_avg, silhouette_values

def intra_class_distance(distance_matrix : np.ndarray, labels : np.ndarray) -> Tuple[np.ndarray, float]:
    """
    This function compute intra cluster distance base on the matrix distance and the labels
    For exemple, after training a given clustering model, one can compute the average of the pairwise distance of data points 
    whithin the same cluster given the distance matrix and the the labels.
    
    Args :
        distance_matrix (np.ndarray) : Pairwise matrix distance of all data points
        labels(np.ndarray) : labels of all data points. 
        
    Returns : 
        np.array : Average distance of data points within each cluster. 
                   The shape will be the number of unique cluster we have given the labels
        float :Avearage over all cluster. It is global intra cluster distance
    
    Note : This method allows computing all metric at once when we have already computed the intra clusters average distance
    """
    intra_class = np.zeros(labels.shape[0])
   
    for i in np.unique(labels):
        cluster_label = labels[i]
        cluster_points = np.where(labels == cluster_label)[0]
            
        intra_class[i] = np.mean([distance_matrix[i, j]**2 for j in cluster_points if j != i]) if  len(cluster_points) > 1 else 0
        
    return intra_class, np.mean(intra_class)


def inter_class_distance (distance_matrix : np.ndarray, labels : np.ndarray) -> Tuple[np.ndarray, float]:
    """
    This function compute intrer cluster distance base on the matrix distance and the labels
    For exemple, after training a given clustering model, one can compute the average of the pairwise distance between data point 
    within the same cluster and all other data points ouside this cluster. 
    For exemple, if we have tree clusters and two data points whithin each cluster, for the first cluster, 
    the pairwise distance between each point of the cluster and all other points in the other cluster will be compute
    and we will return the average of those distance. This is only for the first cluster. The same process will be repeat for the other cluster
    
    
    Args :
        distance_matrix (np.ndarray) : Pairwise matrix distance of all data points 
        labels(np.ndarray) : labels of all data points. 
        
    Returns : 
        np.array : Average distance of each clusters with the others ones. 
                   The shape will be the number of unique cluster we have given the labels( the exact number of cluster)
        float : Avearage over all cluster. It is global inter cluster distance
    
    Note : This method allows computing all metric at once when we have already computed the inter clusters average distance
        
    """
    inter_class = np.zeros(labels.shape[0])
    for i in np.unique(labels):
        cluster_label = labels[i]
        cluster_points = np.where(labels == cluster_label)[0]

        inter_class_ij = []
        for other_label in np.unique(labels):
            if other_label != cluster_label:
                other_cluster_points = np.where(labels == other_label)[0]            
                inter_class_ij.append(np.mean([distance_matrix[i, j]**2 for j in other_cluster_points]))

        inter_class[i] = np.mean(inter_class_ij)

    return inter_class, np.mean(inter_class)


def Calisnki_Harabasz_score(inter_mean : float, intra_mean :float, n_cluster :int) -> float :
    """This method compute the Calisnki score
    
    Args :
        inter_mean(float) : The inter cluster average distance
        intra_mean(float) : The intra cluster average distance
        n_cluster(int) : the number of clusters
    
    Returns :
        flaot : The Calisnki score
    """
    if inter_mean < 0 or intra_mean < 0 :
        raise ValueError(f"One of the average distance {inter_mean} or {intra_mean} are negative")
    if intra_mean==0 :
        return 0.0
    intra_global_mean = intra_mean/n_cluster
    inter_global_mean = inter_mean/(n_cluster -1)
    
    return inter_global_mean/intra_global_mean
  
def Hartigan_score(inter_mean: float, intra_mean: float) -> float :
    """This method compute the Hartigan score
    
    Args :
        inter_mean(float) : The inter cluster average distance
        intra_mean(float) : The intra cluster average distance
    
    Returns :
        flaot : The Hartigan score
    
    Raise :
        ValueError : If one of the argument have a negative value
    """
    
    if inter_mean < 0.0 or intra_mean < 0.0:
        raise ValueError(f"One of the average distance inter_mean = {inter_mean} or intra_mean = {intra_mean} are negative")
        
    if intra_mean ==0.0 :
        return 0.0
    return np.log(inter_mean/intra_mean)

def Xu_score(intra_mean : float, n_sample : int, n_cluster : int, D : int=1) -> float :
    """This method compute the Xu score
    
    Args :
        intra_mean(float) : The intra cluster average distance
        n_sample(int) : The number of data points from which the intra_mean is computed
        n_cluster(int) : The number of cluster from with the intra_mean is computed
        D : the number of feature in the training data set. In our case, we fix it to 1
    
    Returns :
        flaot : The Xu score
    
    Raise :
        ValueError : If intra_mean is negative.
    """
    if intra_mean < 0.0:
        raise ValueError(f"  intra_mean = {intra_mean} is negative")
   
    if intra_mean !=0:
        return D*np.log2(np.sqrt(intra_mean/(D*n_sample**2))) + np.log(n_cluster) 
    return 0.0