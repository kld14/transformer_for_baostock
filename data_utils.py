import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np


data_paths_file = '/data/kld/baostock/A_Code_noBJ.txt'
data_paths_prefix = '/data/kld/baostock/A_Data/dailyEst/'


def get_datasets(num_data, d_data):
    f = open(data_paths_file)
    t = f.readline().rstrip()
    file_nums = 0
    # load datasets name
    datasets = []
    while t:
        datasets.append(t)
        file_nums += 1
        if file_nums == num_data:
            break
        t = f.readline().rstrip()
    hist_data=[]
    trg_data=[]
    for dsname in datasets:
        dataset = pd.read_csv(data_paths_prefix+dsname+'.csv',usecols=[2],nrows=d_data)
        #close = dataset['close'].astype(str)
        #series = pd.Series(dataset['close'].values, index=dataset['date']) # convert series to ndarray by .values
        hist_data.append(dataset['close'].values[:-1]) # [num_data, d_data]
        trg_data.append([dataset['close'].values[-1]]) # [num_data]
    return torch.tensor(np.array(hist_data)).float(), torch.tensor(np.array(trg_data)).squeeze(-1).float()


class TransformerDataset(Dataset):
    def __init__(self, hist_data, trg_data):
        super(TransformerDataset, self).__init__()
        self.hist_data = hist_data
        self.trg_data = trg_data
    def __len__(self):
        assert self.hist_data.shape[0] == self.trg_data.shape[0]
        return self.hist_data.shape[0]
    def __getitem__(self, index):
        return self.hist_data[index], self.trg_data[index] # [a,b,c], a
        