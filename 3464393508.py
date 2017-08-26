import pickle
import sys

import numpy as np


# From https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def extract(data, N):
	return {
		'testing':{
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

def main():
	K = sys.argv[1]
	D = sys.argv[2]
	N = int(sys.argv[3])
	PATH_TO_DATA = sys.argv[4]

	data_dict = unpickle(PATH_TO_DATA)

	extracted_data = extract(data_dict, N)

	grey_data = {
		'testing':{
			'data': np.array([greyscale(x) for x in extracted_data['testing']['data']]),
			'labels': extracted_data['testing']['labels']
		},
		'training':{
			'data': np.array([greyscale(x) for x in extracted_data['training']['data']]),
			'labels': extracted_data['training']['labels']
		}
	}

main()