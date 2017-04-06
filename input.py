
import pandas as pd


class Input(object):
    """
    class for handling input. having two data. train and test
    """

    def __init__(self,  train_file="train_data.csv",  test_file="test_data.csv"):
        # read data and nomalization to [0, 1]
        self.train_data = pd.read_csv(train_file) / 255
        self.test_data = pd.read_csv(test_file) / 255


    def train_batch(self, n=100):
        """
        return feature and label vector

        :param n:
        :return: xs,  ys
            array. xs is feature and ys is one hot vector of label
        """

        sample = self.train_data.sample(n=n,  axis=0)
        print(sample.shape)
        return sample.iloc[:, :-1], sample.iloc[:, -1]

    def test_batch(self, n=100):
        """
        return feature and label vector

        :param n:
        :return: xs,  ys
            array. xs is feature and ys is one hot vector of label
        """

        sample = self.test_data.sample(n=n, axis=0)
        print(sample.shape)
        return sample.iloc[:, :-1], sample.iloc[:, -1]


if __name__  == "__main__":
    Data = Input()
    print(Data.train_batch())

