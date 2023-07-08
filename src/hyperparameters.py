from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class HyperparameterSearch:
    def __init__(self, parameters, model, search_algorithm):
        self.parameters = parameters
        self.model = model
        self.search_algorithm = search_algorithm
    
    def fit(self, X, y, cv=5, n_iter=3):
        """Executa a busca de hiperparâmetros para ajuste

        Args:
            X (DataFrame): As features do dado
            y (Series | array-like): As labels
            cv (int): O número de folds da validação cruzada
            n_iter (int): O número de iterações para RandomizedSearchCV

        Returns:
            best_model: Melhor modelo obtido na busca
            search: O objeto que contém os resultados da busca
        """
        if self.search_algorithm == "GridSearch":
            search = GridSearchCV(self.model, self.parameters, cv=cv)
        elif self.search_algorithm == "RandomizedSearch":
            search = RandomizedSearchCV(self.model, self.parameters, cv=cv, n_iter=n_iter)

        search.fit(X, y)
        best_model = search.best_estimator_

        return best_model, search
    
    @staticmethod
    def best_parameters_set(search, parameters):
        """Imprime os melhores valores de parâmetro dos resultados da busca

        Args:
            search: O objeto que contém os resultados da busca
            parameters (dict): O dicionário de hiperparâmetros
         """
        parameters = {}
        best_parameters = search.best_estimator_.get_params()

        for param_name in sorted(parameters.keys()):
            parameters[param_name] = best_parameters[param_name]

        return parameters