import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, file_path, columns=None):
        self.file_path = file_path
        self.columns = columns

    def load_data(self):
        """
        Carrega o conjunto de dados de um arquivo CSV
        """
        if self.columns is None:
            data = pd.read_csv(self.file_path)
        else:
            data = pd.read_csv(self.file_path, names=self.columns)
        return data

    def split_data(self, data, target_column):
        """Divide o conjunto de dados em features (X) e labels (y)

        Args:
            data (pandas DataFrame): Os dados a serem divididos
            target_column (str): O nome da coluna que representa as labels

        Returns:
            X (pandas DataFrame): As features do dado
            y (pandas Series): As labels
        """
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        return X, y
    
    def train_test_split(self, X, y, test_size=0.2, random_state=351):
        """Divide os dados em conjuntos de treino e teste

        Args:
            X (pandas DataFrame): Os dados
            y (pandas Series): as labels do dado
            test_size (float): A proporção dos dados a serem incluídos no conjunto de teste
            random_state (int): O valor de semente para geração de números aleatórios

        Returns:
             X_train (pandas DataFrame): Os dados de treino
             X_test (pandas DataFrame): Os dados de teste
             y_train (pandas Series): As labels de treino
             y_test (pandas Series): As labels de teste
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    

class Preprocessing:
    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()
    
    def label_encode(self, data):
        """Aplica codificação de rótulo aos dados de entrada

        Args:
            data: Os dados a serem codificados

        Return:
            encoded_data: Os dados codificados
        """
        for col in data.columns:
            data[col] = self.label_encoder.fit_transform(data[col])
        return data
    
    def apply_map(self, data):
        """Aplica uma função lambda para transformar os valores nos dados de entrada

        Args:
            data: Os dados a serem transformados

        Return:
            transformed_data: Os dados transformados
        """
        transformed_data = data.applymap(lambda x: 1.0 if x == 0 else 0.0)
        return transformed_data