import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

import plotly.express as px

def pie_chart_ploty(dataframe : DataFrame, col : list[str]) -> None :
    """ Fonction pour la visualisation de la proportion de données dupliquées et uniques avec Plotly
    
    Args :
        dataframe (pandas.DataFrame) : Le jeu de données dont on veut visualiser la proportion de données uniques et dupliquées
        col (list[str])              : Liste des noms de colonnes sur lesquelles l'on veut déterminer les doublons
        
    Return :
        None
        
    Raise :
        None
    """

    total = dataframe.shape[0] # Nombre de lignes de la datafram
    dupl = dataframe.loc[dataframe[col].duplicated(), :].shape[0] # Extraction de la proportion de données dupliquées
    unique = dataframe.loc[~dataframe[col].duplicated(), :].shape[0] # Extraction de proportion de données non dupliqué
    
    assert dupl + unique == total, "La somme de lignes des données uniques et dupliquées doit être égale à la totalité lignes dans le dataframe original"
    
    
    x = {
        'Uniques': unique,
        'Dupliquées': dupl
    }

    dataa = pd.Series(x).reset_index(name='Proportion').rename(columns={'index': 'Type'})

    fig = px.pie(dataa, values='Proportion', names='Type', width=800, height=400, title="Données dupliquées vs données uniques", color_discrete_sequence=['green', 'red'])
    fig.update_layout(autosize=False,width=500,height=500)
    fig.show()

