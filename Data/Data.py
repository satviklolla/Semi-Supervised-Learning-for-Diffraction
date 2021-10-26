from torch.utils.data import Dataset
import torch
class DiffractionDataset(Dataset):
    def __init__(self, data, labels=None, unsupervised=True):
        self.data = data
        self.unsupervised= unsupervised
        if not self.unsupervised:
            self.labels = labels

    def __getitem__(self, index):# Returns tensor
        if self.unsupervised:
            return self.data[index]
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def batch_u(self, batch_size):
        idx = torch.randint(0,len(self.data),(batch_size,))
        return self.data[idx]