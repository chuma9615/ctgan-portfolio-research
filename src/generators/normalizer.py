import numpy as np
import pandas as pd
from sklearn import preprocessing


class Normalizer:
    def __init__(self, method='quantile'):
        self.method = method
        self.factor_columns = None
        self.total_columns = None
        self.params = {}

    def normalize(self, data):
        """ Normalization proces, returns a dataframe with each column normalized """

        self.total_columns = data.columns
        self.factor_columns = [x for x in data.columns if x.startswith('f_')]

        if self.method == 'quantile':
            self.params['quantile'] = preprocessing.QuantileTransformer(random_state=0)
            data[self.factor_columns] = self.params['quantile'].fit_transform(data[self.factor_columns])

        return data

    def denormalize(self, data):
        """ De-Normalization proces, given a normalized dataframe, returns an inverse-transformed dataframe """

        if self.method == 'quantile':
            if type(data) == pd.DataFrame:
                data[self.factor_columns] = self.params['quantile'].inverse_transform(data[self.factor_columns])

            elif type(data) == np.ndarray:
                data_df = pd.DataFrame(data, columns=self.total_columns)
                data_df[self.factor_columns] = self.params['quantile'].inverse_transform(data_df[self.factor_columns])

                data = data_df.values

        return data
