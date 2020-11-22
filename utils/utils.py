import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data():
    """
    Load the training data and the testing date.
    :return: [[training data, training label], testing data]
    """
    trdata_file = pandas.read_csv("training_data.csv").to_numpy()

    data = trdata_file[:, :-1]
    label = trdata_file[:, -1]

    tedata_file = pandas.read_csv("songs_to_classify.csv").to_numpy()

    return [[data, label], tedata_file]


def cm_plot(CM, save_name):
    """
    This function will draw a confusion matrix and then save the figure as
    a pdf file.
    :param CM: Confusion matrix with ground truth in the 0 dimension.
    :param save_name: the prefix of the pdf file, e.g. SAVE_NAME.pdf

    usage example:
    CM = np.array([[2, 3], [1, 4]])
    cm_plot(CM, "test")
    """
    sns.set()
    f, ax = plt.subplots(figsize=[5.5, 4])

    label_font = {'weight': 'bold', 'size': 16}
    sns.heatmap(CM, annot=True, ax=ax, fmt="d", cmap=plt.get_cmap("BuGn"),
                annot_kws={"fontdict": {'weight': 'bold', 'size': 18}},
                xticklabels=['LIKE', 'DISLIKE'],
                yticklabels=['LIKE', 'DISLIKE'])
    ax.set_xlabel('Predict', fontdict=label_font)  # x轴
    ax.set_ylabel('Ground Truth', fontdict=label_font)  # y轴

    f.subplots_adjust(bottom=0.15, right=1, left=0.1, top=0.99)
    # plt.show()
    plt.savefig("{}.pdf".format(save_name))
    plt.close()


def nor_data(data):
    # Normalised the data
    nor_factor = np.ones(13)
    # duration
    nor_factor[2] = 700000
    # key
    nor_factor[5] = 11
    # loudness
    nor_factor[7] = -60
    # tempo
    nor_factor[10] = 250
    # time signature
    nor_factor[11] = 5
    data = data / nor_factor

    return data


if __name__ == "__main__":
    CM = np.array([[2, 3], [1, 4]])
    cm_plot(CM, "test")
    pass
