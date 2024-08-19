import pickle


class TemporaryDataStorage:
    def __init__(self, filename):
        self.filename = filename

    def save_data(self, data):
        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)

    def load_data(self):
        with open(self.filename, 'rb') as file:
            return pickle.load(file)