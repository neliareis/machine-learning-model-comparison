import csv

class DictCSVWriter:
    def __init__(self, file_path):
        self.file_path = file_path

    def save_dict_to_csv(self, data_dict):
        """
        Salva o dicionário em um arquivo CSV.

        Args:
            data_dict (dict): O dicionário a ser salvo no arquivo CSV.
        """

        # Extrai as chaves do dicionário
        keys = data_dict.keys()
        # Extrai os valores do dicionário
        values = data_dict.values()
        # Prepara os dados que serão gravados no arquivo CSV
        data = zip(keys, values)

        # Abre o arquivo CSV no modo de gravação
        with open(self.file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Escreve a linha do cabeçalho com as chaves do dicionário
            writer.writerow(keys)
            # Escreve as linhas dos dados
            writer.writerows(data)