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

    def len(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        batch_data = self.data[idx]
        label = self.labels[idx]

        batch = {'data': batch_data, 'labels': label}

        return batch

class Retea_CNN(nn.Module):
    
    def __init__(self, nr_clase):
        # mostenirea clasei de baza nn.Module
        super(Retea_CNN,self).__init__()
        
        # CONV => RELU => POOL
        # INPUT: 1, 28, 28  (Grey Image 28x28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=[3, 3], stride = [1, 1], padding = [1, 1])
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2])

        # CONV => RELU => POOL
        # INPUT: 3, 14, 14
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=[3, 3], stride = [1, 1], padding = [1, 1])
        self.relu2 = nn.ReLU()
        # 9, 14, 14
        self.maxpool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2])
        
        # FC Layers
        # INPUT: 9, 7, 7
        self.fc1 = nn.Linear(in_features=7*7*9, out_features=128)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(in_features=128, out_features=nr_clase)
    
    def forward(self,input_batch):

        x = self.conv1(input_batch)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1, 3)
        x = self.fc1(x)
        x = self.relu3(x)
        out = self.out(x)
        
        return out
    
# Instantiem reteaua
cnn = Retea_CNN(10)

# Specificarea functiei loss
loss_function = nn.CrossEntropyLoss(reduction='sum')

# Specificarea optimizatorului
optim = torch.optim.SGD(cnn.parameters(), lr=1e-5)

mnistTrain = DatasetMNIST(r'train-images.idx3-ubyte', r'train-labels.idx1-ubyte')
mnistTest = DatasetMNIST(r't10k-images.idx3-ubyte', r't10k-labels.idx1-ubyte')

trainLoader = DataLoader(mnistTrain, batch_size=128, shuffle=True, num_workers=0)
testLoader = DataLoader(mnistTest, batch_size=128, shuffle=False, num_workers=0)

nr_epoci = 15

for epoca in range(nr_epoci):
    predictii = []
    etichete = []

    # Luam urmatoarele <batch_size> esantioane si etichete
    for batch in trainLoader:
        batch_data = batch["data"]
        batch_labels = batch["labels"]

        # Se calculeaza predictia retelei pentru datele curente (forward pass/ propagare inainte)
        current_predict = cnn.forward(batch_data)

        #Se calculeaza valoarea momentana a fct loss
        loss = loss_function(current_predict, batch_labels) 

        # Se memoreaza predictiile si etichetele aferente batch-ului actual (pentru calculul acuratetii)
        current_predict = np.argmax(current_predict.detach().numpy(), axis=1)
        predictii = np.concatenate((predictii,current_predict))
        etichete = np.concatenate((etichete,batch_labels))

        ### TRAINING THE NETWORK ###
        # 1 - Se sterg toti gradientii calculati anteriori, pentru toate variabilele antrenabile
            # deoarece, metoda <backward> acumuleaza noile valori, in loc sa le inlocuiasca.
        optim.zero_grad()
        # 2 - Calculul tuturor gradientilor = Backpropagation
        loss.backward()
        # 3 - Actualizarea tuturor ponderilor pe baza gradientilor
        optim.step()


    # Calculam acuratetea
    acc = np.sum(predictii==etichete)/len(predictii)
    print( 'Acuratetea la epoca {} este {}%'.format(epoca+1,acc*100) )



## Testare
predictii = []
test_labels = []
for batch in testLoader:
    batch_data = batch['data']
    batch_labels = batch['labels']

    current_predict = cnn.forward(batch_data)
    current_predict = np.argmax(current_predict.detach().numpy(),axis=1)
    predictii = np.concatenate((predictii,current_predict))
    test_labels = np.concatenate([test_labels, batch_labels])

acc = np.sum(predictii==test_labels)/len(predictii)
print( 'Acuratetea la test este {}%'.format(acc*100) )

