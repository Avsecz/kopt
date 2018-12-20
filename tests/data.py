from keras.preprocessing import sequence
from keras.datasets import imdb
import h5py

def data(max_features=5000, maxlen=400):
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    # subset the data
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:100]
    y_test = y_test[:100]

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    return (x_train, y_train, [1, 2, 3, "dummy_data"]), (x_test, y_test)

def data_hdf5():
    '''
    This function returns training and testing data which is simply the data stored in keras.datasets.cifar10

    Returns
    -------
    Tuple of tuples (x_train, y_train), (x_test, y_test)
    '''
    print('Loading data...')
    # open and allow other to open and read in other processes
    f = h5py.File('data/data.h5', 'r', libver='latest', swmr=True)
    return (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])