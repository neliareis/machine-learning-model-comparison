from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.hyperparameters import HyperparameterSearch


class ClassificationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self._initialize_model()
        self.parameters = self.get_model_hyperparameters()
    
    def _initialize_model(self):
        """
        Inicializa o modelo de classificação específico com base no nome do modelo
        """
        if self.model_name == "LogisticRegression":
            return LogisticRegression()
        elif self.model_name == "RandomForest":
            return RandomForestClassifier()
        elif self.model_name == "SVM":
            return SVC()
        elif self.model_name == "DecisionTree":
            return DecisionTreeClassifier()
        else:
            raise ValueError("Nome do modelo inválido. Modelos suportados: LogisticRegression, RandomForest, SVM, DecisionTree")
    
    def get_model_hyperparameters(self):
        """
        Obtem os hiperparâmetros específicos do modelo
        """
        if self.model_name == "LogisticRegression":
            return {'C': [0.1, 1, 10]}
        elif self.model_name == "RandomForest":
            return {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 25, 10, 50]}
        elif self.model_name == "SVM":
            return {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}
        elif self.model_name == "DecisionTree":
            return {'max_depth': [None, 5, 25 , 50], 'min_samples_leaf': [1, 2, 3]}
 

    def train(self, X_train, y_train, search_type=None):
        """Treina o modelo de classificação com ou sem ajuste de hiperparâmetros

        Args:
            X_train (pandas DataFrame): Os dados de treino
            y_train (pandas Series): As labels de treino
            search_type (str): Nome do otimizador de hiperparâmetros

        Return:
            search_scv (dict | object): Dicionário com os parametros ou otimizador
        """
        search_scv = None
        if search_type == "grid":
            parameters = self.get_model_hyperparameters()
            search = HyperparameterSearch(parameters, self.model, "GridSearch")
            self.model, search_scv = search.fit(X_train, y_train)

        elif search_type == "randomized":
            parameters = self.get_model_hyperparameters()
            search = HyperparameterSearch(parameters, self.model, "RandomizedSearch")
            self.model, search_scv = search.fit(X_train, y_train)
            
        else:
            self.model.fit(X_train, y_train)
            search_scv = self.model.get_params(deep=False)

        return search_scv

    def predict(self, X_test):
        """Realiza as previsões usando o modelo de classificação treinado.
        
        Args:
            X_test (pandas DataFrame): Os dados de teste para fazer previsões

        Return:
            predictions (array-like): Os valores preditos
        """
        predictions = self.model.predict(X_test)
        return predictions
    
    def evaluate(self, y_test, predictions):
        """Avalie o desempenho do modelo de classificação

        Args:
            y_test (array-like): Os verdadeiros rótulos dos dados de teste
            predictions (array-like): os rótulos previstos

        Return:
            accuracy (float): a pontuação de precisão do modelo
        """
        accuracy = accuracy_score(y_test, predictions)
        return accuracy