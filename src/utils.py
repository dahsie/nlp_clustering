
import pandas as pd
from pandas import DataFrame
from deep_translator import GoogleTranslator
import re
from tqdm import tqdm
#Global variables 
PATTERNS=[r'[\u4E00-\u9FFF]+', #Chinois
          r'[\u0400-\u04FF]+',# Russe
          r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+',# Coreen
          r'[\u00C0-\u024F]+',# latin etendu
          # r'[\u3040-\u309F]+', # Japonais (Hiragana)
          # r'[\u30A0-\u30FF]+', # Japonais (Katakana)
       #other unicode to add
     ]

REX = "|".join(PATTERNS)

def process_column_str(col : DataFrame.columns) -> DataFrame.columns:
    """La fonction permet de traiter une colonne, dont le type est objet(string DataFrame), d'un DataFrame de manière uniforme.
    En effet, la lonnes dont le type de données est "objet", une tranformation doit être appliqué de sorte 
    à convertir les données en str et ensuite remplacer les données manquantes par des chaînes vide. Ce traitement permettra
    de pouvoir concaténer deux colonnes dont les types sont des str.
    
    Agrs : 
        col(DataFrame.columns) : colonne sur laquelle ont applique le traitement
    
    Returns :
        col(DataFrame.columns) : On retourne la colonne après traitement
    
    Exemple :
    
    """
    if col.dtype == 'object':

        return col.fillna('').astype(str).str.lower()
    return col


def process_dataframe_str(dataframe : DataFrame) -> DataFrame:
    """Fonction Permettant de traiter toutes les colonnes, dont les type de données sont des str, d'un dataframe
    
    Args : 
        dataframe(DataFrame) : Données sur lesquelles sont appliquées les transformations
    
    Returns : 
        dataframe(DataFrame) : Données renvoyer après transformations
    """
    return dataframe.apply(process_column_str)

def concat_sub_str(str1 : str) -> str:
    """
    This function allow concatenate sub string which have at most 1 length.
    For exemple : concat_sub_str("s v k marketing s v vv") will return "svk marketing sv vv"
    
    Args :
        str1 (str) : The string to process
    
    Return :
        str : The processed string
    """
    if isinstance(str1, str) :
        words = re.findall(r'\b\w+\b', str1)
        new_str = ''.join(word if len(word) == 1 else ' ' + word + ' ' for word in words)
        return new_str.strip()
    return str1


def remove_pattern(string1: str, string2: str) -> str:
    """Fonction permettant la suppression de la sous chaîne string1 dans la chaîne string2

    Args:
        string1 (str): est un motif
        string2 (str): une chaîne dans laquelle nous vérifierions si le motif string1 est présent.

    Returns:
        string2(str) après avoir supprimé string1 si string1 est présent dedans.
        
    Exemples :
        >> string1 = "iralco"
        >> string2 = "iralco arak iralco iran"
        >> string3 =remove_pattern(string1,string2)
        >>print(string3)
        [out] : "arak"
    """

    return string2.replace(string1,"").strip()

def remove_duplication(string1: str) -> str:
    """Cette fonction permet de supprimer des doublons dans une liste de chaîne de caractère

    Args:
        string1 (str):La chaîne de laquelle on veut supprimer les sous chaînes repétitifs

    Returns:
        str: La chaine de caractère returnée sans doublons. On retourne une chaîne
        contenant des sous chaînes 
           
    Exemples :
        >> string1 = "iralco arak iralco iran"
        >> string2 = remove_duplication(string1)
        >>print(string2)
        [out] : "iralco arak"
    """
    
    string1=string1.split()
    string2 :list[str]=list()

    for i in range(len(string1)):
        if len(string1[i])==1 or string1[i] not in string2: #Ne pas supprimer les douybons sur les caractère
                                                            # pas exemple "d o o" qui est est réalisté "d.o.o"
            string2.append(string1[i])
    
    return " ".join(string2)


def detect_pattern(dataframe : DataFrame, col : str, pattern: str) -> DataFrame :
    """La fonction permet de tester si la colonne "col" du DataFrame "dataframe" de contient l'expression regulière "pattern"

    Args:
        dataframe(DataFrame) : Un DataFrame Pandas
        col(str) : La colonne suivant laquelle nous allons tester la présence de l'expression régulière "pattern";
        pattern(str) : Une expression regulière pour rechercher les chaînes de caractères verifiant certaines conditions.

    Returns:
        DataFrame : Un DataFrame dont les lignes vérifier bien la recherche de l'expression regulière "pattern"
    """

    return dataframe[dataframe[col].astype(str).str.contains(pattern)]


# def chaine_to_chars(strings: str) -> str:
#     """La fonction permet d'obtenir tous les caratère de la chaîne passée en parametre sans les spaces
#     Args :
#         strings : la chaîne dont on veut retoruner les caratères sans espace
#     Returns :
#         str : Les caratères de la chaîne
#     """
#     return strings.lower().replace(" ","")

def remove_blank_space( str1 : str) -> str:
    """Removing all blanket space everywhere in the string.
    
    Args :
        str1(str) :
    
    Return :
        str : string without blanket space
        
    Exemples :
        >>remove_blank_space("dah ")
        >> "dah"
        >>remove_blank_space(" dah")
        >>"dah"
        remove_blank_space(" dah ")
        >>"dah"
        >>remove_blank_space(" dah   sie ")
        >> "dah sie"
    """
    if isinstance(str1, str) :
        return " ".join(item.strip('.!,? \n\t') for item in str1.split() if len(item) !=0).lower()
    return str1

def translation(dataframe : DataFrame, special_char_cols : list[str]) -> DataFrame:
   
    """
    Translates the text in specified columns of a DataFrame from their original language to English.

    Argrs:
        dataframe (DataFrame): The DataFrame containing the text to be translated.
        special_char_cols (list[str]): The list of column names in the DataFrame that contain the text to be translated.

    Returns:
        DataFrame: The DataFrame with the text in the specified columns translated to English.

    Note: This function uses the GoogleTranslator for the translation process.
    """
    mask = dataframe[special_char_cols].apply(lambda col: col.str.contains(r'[a-z]+'))
    filtered_data = dataframe[mask.any(axis=1)]
    rest_data = dataframe.drop(index=filtered_data.index)
    
    #Model 
    translator = GoogleTranslator(source= "auto",target="english")
    
    for col in tqdm(special_char_cols):
        rest_data[col] = translator.translate_batch(list(rest_data[col]))
    
    def translate_text(chaine):
        chinese_chars = re.findall(REX, chaine)
        for chars in chinese_chars:
            translation = translator.translate(chars)
            if translation is not None:  # cheick if the translation is not None
                chaine = chaine.replace(chars, translation)
        return chaine
    
    filtered_data = process_dataframe_str(filtered_data)
    # filtered_data[special_char_cols] = filtered_data[special_char_cols].applymap(translate_text)
    filtered_data[special_char_cols] = filtered_data[special_char_cols].map(translate_text)
    
    return pd.concat((rest_data, filtered_data))

def occurrences(ref_dataframe : DataFrame, 
                processing_dataframe : DataFrame, 
                col : str) -> DataFrame:
    """
    Computes the number of occurrences of a specific value in a given column of a processing_dataframe within a reference DataFrame,

    Args:
        ref_dataframe (DataFrame): The reference DataFrame to count occurrences in.
        processing_dataframe (DataFrame): The DataFrame to merge the occurrence information with.
        col (str): The name of the column to count occurrences in.

    Returns:
        DataFrame: The processing DataFrame with an additional column showing the number of occurrences of each value in the specified column.
    """
    
    # occurrence = ref_dataframe[col].value_counts().reset_index()
    occurrence = ref_dataframe[col].value_counts().reset_index()
    occurrence.columns = [col, 'occurrence']

    # Merging of occurrences with data frame processing
    processing_dataframe = processing_dataframe.merge(occurrence, on=col, how='inner')

    # Fill NaN values with 0 if necessary
    processing_dataframe['occurrence'] = processing_dataframe['occurrence'].fillna(0)
    
    return processing_dataframe


    
def suggest_group_name(group : pd.DataFrame, tiern_name):
    """
    Suggests a group name based on the occurrence of items within the group.

    Args:
        group (pd.DataFrame): The DataFrame representing the group.
        tiern_name (str): The column name in the DataFrame that contains the items for which to suggest a group name.

    Returns:
        DataFrame: The DataFrame with an additional column named 'suggest' that contains the suggested group name for each item.

    Note: 
    - If all items within the group have the same occurrence, a random item is chosen as the suggested group name.
    - If the occurrences vary, the item that appears the most is suggested as the group name.
    """
    if group["occurrence"].max() == group["occurrence"].min(): # all item whithin the group have the same occurence, so we can't make a suggestion
        group["suggested_name"] = random.choice(group[tiern_name].tolist())

    else: #We suggest the item in the group which apprear at most to be the correcte name
        group["suggested_name"] = group.loc[group["occurrence"].idxmax(), tiern_name]
    return group
    
