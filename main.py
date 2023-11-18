import torch
import torch.nn as nn
import data_utils
import transformer_model as trans
from torch.utils.data import DataLoader

d_data = 100  # number of historical value of each stock
n_data = 1000  # total number of stocks
trainfrac = 1
train_batch_size = 128
n_epoch = 10


def main():
    device = torch.device("cuda:0")
    model = trans.TransformerEncoderOnly(d_data=d_data).to(device)
    print(f'model.predict_ln: {model.predict_ln.weight.data}')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # preparing data
    hist_data, trg_data = data_utils.get_datasets(n_data, d_data)
    #print(f"hist_data:{hist_data.shape}, trg_data:{trg_data.shape}")
    train_split = int(hist_data.shape[0] * trainfrac)
    #print(train_split)
    train_hist = hist_data[:train_split]
    train_trg = trg_data[:train_split]
    test_hist = hist_data[train_split:]
    test_trg = hist_data[train_split:]

    #print(f"train_hist:{train_hist.shape}\ntrain_trg:{train_trg.shape}")

    train_datasets = data_utils.TransformerDataset(train_hist, train_trg)
    train_loader = DataLoader(dataset=train_datasets, batch_size=train_batch_size, shuffle=True)

    for epoch in range(n_epoch):
         #print(f'epoch: {epoch}')
         train(model, device, train_loader, criterion, optimizer)

    print(f'model.predict_ln: {model.predict_ln.weight.data}')


def train(model, device, train_loader, criterion, optimizer):
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        hist, trg = batch
        hist = hist.unsqueeze(0)  # [1, batch_size, d_data]
        hist, trg = hist.to(device), trg.to(device)
        # print(hist.shape)
        outputs, attns = model(hist)
        outputs = outputs.squeeze(0)
        #print(f'outputs: {outputs.shape}, trg: {trg.shape}')
        #print(f'outputs: {outputs}, trg: {trg}')
        #print(outputs[0])
        loss=criterion(outputs,trg)
        loss.backward()
        optimizer.step()
        if i%10==5:
            print(f'batch idx: {i}, loss: {loss}')
            print(outputs[0])

main()