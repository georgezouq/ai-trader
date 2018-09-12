from base.base_algorithm import BaseAlgorithm
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from base.model.document import Stock
from sklearn.preprocessing import MinMaxScaler


class LSTMAlgorithm(BaseAlgorithm):
    def __init__(self, params):
        super(LSTMAlgorithm, self).__init__(params)

    def init_model(self):
        self.model.add(LSTM(10, input_dim=self.params.get('input_dim'), return_sequences=True))
        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(loss="mse", optimizer="adam")
        self.model.summary()
        return self.model

    def run(self, x, y):
        return self.model.fit(x, y, epochs=1000, batch_size=128)


def to_scale(data):
    scale = MinMaxScaler()
    scale.fit(data)
    return scale.transform(data)


if __name__ == '__main__':
    doc_class = Stock()
    data_docs = doc_class.get_k_data('600036', '2017-01-01', '2018-01-01')
    data_dicts = [row.to_dict() for row in data_docs]
    data = [row[2:] for row in data_dicts]
    x = to_scale(data)
    y = x[4][2:]
    algorithm = LSTMAlgorithm({'input_dim': 5})
