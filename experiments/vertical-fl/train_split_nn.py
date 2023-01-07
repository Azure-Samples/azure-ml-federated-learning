import os
import torch
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import torch.multiprocessing as mp

class BottomDataset(Dataset):
    """Image dataset."""

    def __init__(self, csv_file, transform=None, workers=1, index=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(BottomDataset, self).__init__()
        self.images = pd.read_csv(csv_file, index_col=0)
        self.images = torch.tensor(self.images.values, dtype=float)
        self.images = torch.reshape(self.images, (-1, 28, 28))
        self.images /= 255.0
        self.transform = transform
        self.workers = workers
        self.index = index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        part = image.shape[0]//self.workers
        image = image[part*self.index:part*(self.index+1)]

        if self.transform:
            image = self.transform(image)

        return image

class TopDataset(Dataset):
    """Image dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(TopDataset, self).__init__()
        self.labels = pd.read_csv(csv_file, index_col=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.labels["label"][idx]

class BottomModel(nn.Module):
    def __init__(self) -> None:
        super(BottomModel, self).__init__()
        self._model = models.resnet18(pretrained=True)
        self._model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self._model = torch.nn.Sequential(*list(self._model.children())[:-1])

    def forward(self, x) -> torch.tensor:
        return self._model(x)

class TopModel(nn.Module):
    def __init__(self) -> None:
        super(TopModel, self).__init__()
        self._model = nn.Linear(512, 10)

    def forward(self, x) -> torch.tensor:
        return self._model(x)

def main(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # initialize the process group
    dist.init_process_group(dist.Backend.GLOO, rank=rank, world_size=world_size)

    criterion_ = nn.CrossEntropyLoss()
    if rank != 0:
        net1 = BottomModel()
        optimizer_1 = Adam(net1.parameters(), lr=1e-3, weight_decay=1e-5)
        net1.train()

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=30),
                transforms.RandomPerspective(),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.1307,), std = (0.3081,)),
            ]
        )
        ds = BottomDataset("./.cache/bottom_train.csv", transform, workers=world_size-1, index=rank-1)
        dl = DataLoader(ds, batch_size=16, shuffle=False)

        for i in range(10):
            for batch in dl:
                optimizer_1.zero_grad()
                predictions1 = net1(batch)

                dist.send(predictions1, 0)
                gradient = torch.zeros_like(predictions1)
                dist.recv(gradient, 0)

                predictions1.backward(gradient)
                optimizer_1.step()

    else:
        net2 = TopModel()
        optimizer_2 = Adam(net2.parameters(), lr=1e-3, weight_decay=1e-5)
        net2.train()

        ds = TopDataset("./.cache/top_train.csv")
        dl = DataLoader(ds, batch_size=16, shuffle=False)

        acc_sum = 0
        loss_sum = 0
        for _ in range(10):
            for i, batch in enumerate(dl):
                optimizer_2.zero_grad()

                outputs = []
                for j in range(1, world_size):
                    data_inter = torch.zeros((len(batch),512))
                    dist.recv(data_inter, j)
                    outputs.append(data_inter)
                
                data_inter = torch.autograd.Variable(torch.stack(outputs), requires_grad=True)
                # data_inter = data_inter.mean(dim=0)
                predictions2 = net2(data_inter.to(torch.float32).mean(dim=0))

                loss = criterion_(predictions2, batch)
                loss.backward()
                optimizer_2.step()

                loss_sum += loss.item()
                acc_sum += (predictions2.argmax(dim=1) == batch).to(float).mean().item()
                if (i+1)%10 == 0:
                    print(f"Loss: {loss_sum/10.0}, Accuracy: {acc_sum/10.0}")
                    acc_sum = 0.0
                    loss_sum = 0.0

                for j in range(1, world_size):
                    dist.send(data_inter.grad[j-1], j)

def prepare_data():
    if not os.path.exists("./.cache/train.csv"):
        df = pd.read_csv("https://azureopendatastorage.blob.core.windows.net/mnist/processed/train.csv")
        os.makedirs("./.cache", exist_ok=True)
        df.to_csv("./.cache/train.csv")

    if not os.path.exists("./.cache/bottom_train.csv") or not os.path.exists("./.cache/top_train.csv"):
        df = pd.read_csv("./.cache/train.csv", index_col=0)

        x_df = df.iloc[:, df.columns != "label"]
        x_df.to_csv("./.cache/bottom_train.csv")

        y_df = df[["label"]]
        y_df.to_csv("./.cache/top_train.csv")

if __name__ == "__main__":
    prepare_data()
    mp.spawn(main, args=(5,), nprocs=5, join=True)