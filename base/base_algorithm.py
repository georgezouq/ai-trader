from keras import Sequential


"""
Basic Algorithm Class
"""


class BaseAlgorithm(object):
    def __init__(self, params):
        self.params = params
        self.model = Sequential()

        self.init_data()
        self.init_model()

    def init_data(self):
        pass

    def init_model(self):
        pass

    def run(self):
        pass

    def predict(self):
        pass

    def evolution(self):
        pass