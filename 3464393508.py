import pickle
import sys

import numpy as np
from sklearn.decomposition import PCA

# From https://www.cs.toronto.edu/~kriz/cifar.html


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def extract(data, N):
    return {
        'testing': {
            'data': data[b'data'][:N],
            'labels': data[b'labels'][:N]
        },
        'training': {
            'data': data[b'data'][N:1000],
            'labels': data[b'labels'][N:1000]
        }
    }


def greyscale(data):
    red = np.array(data[:1024])
    green = np.array(data[1024:2048])
    blue = np.array(data[2048:])
    return (0.299 * red) + (0.587 * green) + (0.114 * blue)


def knn_predict(train_data, train_label, test, K):
    # neighbors = [[distance, index]]
    neighbors = []

    for t in range(len(train_data)):
        eucl_distance = np.sqrt(np.sum(np.square(train_data[t] - test)))
        neighbors.append([eucl_distance, train_label[t]])

    neighbors = sorted(neighbors)[:K]
    weights = {}

    for x in neighbors:
        if x[0] == 0:
            return x[1]
        if x[1] in weights:
            weights[x[1]] += 1 / x[0]
        else:
            weights[x[1]] = 1 / x[0]

    return sorted(weights.items(), key=lambda x: x[1])[-1][0]


def main():
    K = int(sys.argv[1])
    D = int(sys.argv[2])
    N = int(sys.argv[3])
    PATH_TO_DATA = sys.argv[4]

    data_dict = unpickle(PATH_TO_DATA)

    extracted_data = extract(data_dict, N)

    grey_data = {
        'testing': {
            'data': np.array([greyscale(x) for x in extracted_data['testing']['data']]),
            'labels': extracted_data['testing']['labels']
        },
        'training': {
            'data': np.array([greyscale(x) for x in extracted_data['training']['data']]),
            'labels': extracted_data['training']['labels']
        }
    }

    # Dimensionality reduction
    pca = PCA(n_components=D, svd_solver='full')

    # pca.fit(grey_data['training']['data'])
    grey_data['training']['data'] = pca.fit_transform(
        grey_data['training']['data'])
    # print(grey_data['training']['data'].shape)
    # pca.fit(grey_data['testing']['data'])
    grey_data['testing']['data'] = pca.transform(grey_data['testing']['data'])
    # print(grey_data['testing']['data'].shape)

    f = open('3464393508.txt', 'w')

    for i in range(len(grey_data['testing']['data'])):
        new_label = knn_predict(grey_data['training']['data'], grey_data['training'][
                                'labels'], grey_data['testing']['data'][i], K)
        f.write('{new_label} {old_label}\n'.format(
            new_label=new_label, old_label=grey_data['testing']['labels'][i]))
        #print(new_label, grey_data['testing']['labels'][i])

    f.close()
main()
