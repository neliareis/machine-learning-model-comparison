from src.hyperparameters import HyperparameterSearch as hp

class ClassificationRunner:
    def __init__(self, models):
        self.models = models

    def run_classification(self, X_train, y_train, X_test, y_test, search_type=None):
        """Executa os modelos de classificação e avalia seu desempenho

        Args:
            X_train (array-like): Dados de entrada de treinamento
            y_train (array-like): Labels referente ao dado de treinamento
            X_test (array-like): Dados de entrada de teste
            y_test (array-like): Labels referente ao dado de teste
            search_type (str): O tipo de método de busca usado para otimização de hiperparâmetros

        Return:
            results (dict): Um dicionário contendo as pontuações de acurácia dos modelos de classificação
        """
        results = {}
        best_parameters = {}

        for model_name, model in self.models.items():
            # Treina o modelo
            search_scv = model.train(X_train, y_train, search_type=search_type)
            parameters = model.get_model_hyperparameters()

            if type(search_scv) is dict:
                param = search_scv
            else:
                param = hp.best_parameters_set(search_scv, parameters)

            # Realiza as predições
            predictions = model.predict(X_test)

            # Calcula a acurácia
            accuracy = model.evaluate(y_test, predictions)

            # Guarda os resultados
            results[model_name] = accuracy
            best_parameters[model_name] = param

        return results