import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DatasetMNIST(Dataset):
    def __init__(self, cale_catre_date, cale_catre_etichete):
        
        f = open(cale_catre_date,'r',encoding = 'latin-1')
        g = open(cale_catre_etichete,'r',encoding = 'latin-1')

        byte = f.read(16) #4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
        byte_label = g.read(8) #4 bytes magic number, 4 bytes nr labels
                
        data = np.fromfile(f,dtype=np.uint8).reshape(-1, 1, 28, 28)
        labels = np.fromfile(g,dtype=np.uint8)
            
        # Conversii pentru a se potrivi cu procesul de antrenare    
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        batch_data = self.data[idx]
        label = self.labels[idx]

        batch = {'data': batch_data, 'labels': label}

        return batch


trainDataset = DatasetMNIST(r'train-images.idx3-ubyte', r'train-labels.idx1-ubyte')

trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True, num_workers=0)
