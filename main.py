import torch
import torch.nn as nn
import data_utils
import transformer_model as trans
from torch.utils.data import DataLoader

d_data = 5  # number of historical value of each stock
num_data = 10  # total number of stocks
trainfrac = 0.7
train_batch_size = 128

# preparing data
hist_data, trg_data = data_utils.get_datasets(num_data, d_data)
print(f"hist_data:{hist_data}, trg_data:{trg_data}")
train_split = int(hist_data.shape[0] * trainfrac)
train_hist = hist_data[:train_split]
train_trg = trg_data[:train_split]
test_hist = hist_data[train_split:]
test_trg = hist_data[train_split:]

print(f"train_hist:{train_hist.shape}\ntrain_trg:{train_trg.shape}")


train_datasets = data_utils.TransformerDataset(train_hist, train_trg)
train_loader = DataLoader(dataset=train_datasets, batch_size=2, shuffle=True)
for i, batch in enumerate(train_loader):
    hist, trg = batch
    hist = hist.unsqueeze(0)
    print(hist.shape)
