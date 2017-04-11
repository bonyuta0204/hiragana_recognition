
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


class Input(object):
    """
    class for handling input. having two data. train and test
    """

    def __init__(self,  train_file="train_data.csv",  test_file="test_data.csv"):
        # read data and nomalization to [0, 1]
        self.train_data = pd.read_csv(train_file)
        print("Train Data Shape: %s" % str(self.train_data.shape))
        self.test_data = pd.read_csv(test_file)
        print("Test Data Shape: %s" % str(self.test_data.shape))
        self.labels = list(set(self.train_data["label"]))

        self.lb = LabelBinarizer()
        self.lb.fit(self.labels)

    def train_batch(self, n=100):
        """
        return feature and label vector

        :param n:
        :return: xs,  ys
            array. xs is feature and ys is one hot vector of label
        """

        sample = self.train_data.sample(n=n,  axis=0)

        return (sample.iloc[:, :-1].as_matrix() / 255,
                self.lb.transform(sample.iloc[:, -1]))

    def test_batch(self, n=100):
        """
        return feature and label vector

        :param n:
        :return: xs,  ys
            array. xs is feature and ys is one hot vector of label
        """

        sample = self.test_data.sample(n=n, axis=0)

        return sample.iloc[:, :-1].as_matrix() / 255, self.lb.transform(sample.iloc[:, -1])


if __name__  == "__main__":
    Data = Input()
    xs, ys = Data.test_batch(n=100)
    print(xs)

