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


def create_data(start=0, stop=100, num=1000, w=1, mu=0, sigma=0.03):
    seq_train = create_original_data(start, stop, num, w)
    X_train, y_train = generate(seq_train)
    return X_train, y_train


def create_original_data(start=0, stop=100, num=1000, w=1, mu=0, sigma=0.03):
    print('create data')
    seq_train = w * np.sin(np.linspace(start=start, stop=stop, num=num, dtype=np.float32))

    # create normal noise
    sample_no = num
    np.random.seed(0)
    s = np.random.normal(mu, sigma, sample_no)

    seq_train = seq_train + s

    return seq_train

    # plt.plot(range(len(seq_train)), seq_train, "r*")
    # plt.show()



def draw_data(draw_datas, data_len, file_name):
    data_dir = '../../data/'
    data_dir = os.path.join(data_dir, file_name+'.pdf')
    f = plt.figure()
    plt.plot(range(data_len), draw_datas, "r*")
    plt.show()
    f.savefig(data_dir)


def normalization_data(datas):
    min = datas.min(axis=0)
    max = datas.max(axis=0)
    datas = (datas - min) / (max - min)
    return datas


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


def save_original_data(file_name, result_data):
    data_dir = '../../data/'
    print('Save original result begin...')
    file_full_path = os.path.join(data_dir, file_name)
    data_frame = pd.DataFrame(data=result_data)
    data_frame.to_csv(file_full_path, encoding='utf-8')
    print('Save successfully...')


def read_original_data(file_name):
    data_dir = '../../data/'
    data = pd.read_csv(os.path.join(data_dir, file_name))
    train_data = data.values[0:, 1]
    return train_data


def read_data(file_name):
    data_dir = '../../data/'
    data = pd.read_csv(os.path.join(data_dir, file_name))

    train_data = data.values[0:, 1:]
    train_label = data.values[0:, 0]

    return train_data, train_label


if __name__ == '__main__':

    train_data = create_original_data(start=0, stop=12.56, num=2512, w=1, mu=0, sigma=0.01)
    file_name = 'data_period200_w1_mu0_sigma001'
    save_original_data(file_name, train_data)
    read_data = read_original_data(file_name)
    print(read_data)
    # print('train data shape {0}' .format(train_data.shape))
    draw_data(read_data, len(train_data), file_name)

    # transfer_dir = '../../model/'
    # model_path = os.path.normpath(os.path.join(transfer_dir, '1'))
    # print('model path: {0}'.format(model_path))

    # train_label = train_label.reshape((-1, 1, 1))
    # result = np.concatenate([train_data, train_label], axis=2)
    # save_data('testData.csv', result)
    # train_data, train_label = read_data('testData.csv')
