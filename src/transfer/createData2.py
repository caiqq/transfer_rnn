import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd

TIME_STEPS = 30


def generate(seq):
    X = []
    y = []
    for i in range(len(seq) - TIME_STEPS):
        X.append([seq[i:i + TIME_STEPS]])
        y.append([seq[i + TIME_STEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_data(start=0, stop=100, num=1000, w=1):
    print('create data')
    seq_train = w * np.sin(np.linspace(start=start, stop=stop, num=num, dtype=np.float32))

    # create normal noise
    sample_no = num
    mu = 0
    sigma = 0.03
    np.random.seed(0)
    s = np.random.normal(mu, sigma, sample_no)

    seq_train = seq_train + s

    X_train, y_train = generate(seq_train)
    return X_train, y_train


def draw_data(draw_datas, data_len):
    plt.plot(range(data_len), draw_datas[:data_len, 0], "r*")
    plt.show()


def save_data(file_name, result_data):
    data_dir = '../../data/'
    print('Save result begin...')
    new_shape_2 = result_data.shape[2]
    result_data = result_data.reshape((-1, new_shape_2))
    print(result_data.shape)
    file_full_path = os.path.join(data_dir, file_name)
    shape_0 = result_data.shape[1]
    titles = []
    for i in range(shape_0):
        titles.append('label%d' % i)

    data_frame = pd.DataFrame(columns=titles, data=result_data)
    data_frame.to_csv(file_full_path, encoding='utf-8')
    print('Save successfully...')


def read_data(file_name):
    data_dir = '../../data/'
    data = pd.read_csv(os.path.join(data_dir, file_name))

    train_data = data.values[0:, 1:]
    train_label = data.values[0:, 0]

    return train_data, train_label


if __name__ == '__main__':
    train_data, train_label = create_data(start=0, stop=100, num=2110, w=1)
    print('train data shape {0}' .format(train_data.shape))
    print('train label shape {0}'.format(train_label.shape))

    # train_label = train_label.reshape((-1, 1, 1))
    # result = np.concatenate([train_data, train_label], axis=2)
    # save_data('testData.csv', result)
    # train_data, train_label = read_data('testData.csv')
