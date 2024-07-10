import os
import copy
import numpy as np
import torch

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()



def load_dataset_v0(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    # Load data
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']   # (, timestep, num_node, feature_dim)
        data['y_' + category] = cat_data['y']   # (, timestep, num_node, feature_dim)
    # Data format
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        # 注意：这里同时对 x_train、x_val、x_test 进行了归一化，对于标签 y 并没有做归一化
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    print("train:", data['x_train'].shape, " val:", data['x_val'].shape, " test:", data['x_test'].shape)
    # Iterator to initialize the dataset
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler
    return data



def load_dataset_v1(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    # Load data
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']   # (, timestep, num_node, feature_dim)
        data['y_' + category] = cat_data['y']   # (, timestep, num_node, feature_dim)
    # Data format
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        # 注意：这里同时对 x_train、x_val、x_test 进行了归一化，对于标签 y 并没有做归一化
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    # =================================curriculum_learning======================================
    # 需要对训练样本的标签进行标准化处理，适用于标签指导训练过程的课程学习
    data['y_train'][..., 0] = scaler.transform(data['y_train'][..., 0])
    # ==========================================================================================

    print("train:", data['x_train'].shape, " val:", data['x_val'].shape, " test:", data['x_test'].shape)
    # Iterator to initialize the dataset
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler
    return data



def load_dataset_v2(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    # Load data
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']   # (, timestep, num_node, feature_dim)
        data['y_' + category] = cat_data['y']   # (, timestep, num_node, feature_dim)
    # Data format
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        # 注意：这里同时对 x_train、x_val、x_test 进行了归一化，对于标签 y 并没有做归一化
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    print("train:", data['x_train'].shape, " val:", data['x_val'].shape, " test:", data['x_test'].shape)
    
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_train']), torch.FloatTensor(data['y_train']))
    valset = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_val']), torch.FloatTensor(data['y_val']))
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_test']), torch.FloatTensor(data['y_test']))

    data['train_loader'] = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    data['val_loader'] = torch.utils.data.DataLoader(valset, batch_size=valid_batch_size, shuffle=False)
    data['test_loader'] = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)
    data['scaler'] = scaler
    return data

# if __name__ == "__main__":
#     load_dataset_v2(dataset_dir="../data/METR-LA/processed", batch_size=16, valid_batch_size=16, test_batch_size=16)