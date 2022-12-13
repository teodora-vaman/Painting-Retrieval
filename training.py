import cv2
import matplotlib.pyplot as plt
import glob
import os
import shutil
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch import Generator
from torch.utils.data import Dataset, DataLoader, Subset, random_split, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"

classes = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18"]

class_mapping = {
    
    "01_Byzantin_Iconography": 0,
    "02_Early_Renaissance":1,
    "03_Northern_Renaissance":2,
    "04_High_Renaissance":3,
    "05_Baroque":4,
    "06_Rococo":5,
    "07_Romanticism":6,
    "08_Realism":7,
    "09_Impressionism":8,
    "10_Post_Impressionism":9,
    "11_Expressionism":10,
    "12_Symbolism":11,
    "13_Fauvism":12,
    "14_Cubism":13,
    "15_Surrealism":14,
    "16_AbstractArt":15,
    "17_NaiveArt":16,
    "18_PopArt" : 17
}


class DatasetPandora(Dataset):
    def __init__(self, excel_path):
        
        df = pd.read_excel(excel_path)

        self.data = df["paths"]
        self.labels = df["labels"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([64,64]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):

        img = cv2.imread(self.data[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.transpose(img, [2,0,1])

        batch_data = img
        batch_data = self.transf(batch_data)
        # batch_data = batch_data.to(self.device)


        batch_labels = self.labels[idx]
        # batch_labels = batch_labels.to(self.device)

        batch = {'data': batch_data, 'labels': batch_labels}

        return batch_data, batch_labels

def main():

    s = 32
    dev = torch.device('cuda')
    torch.cuda.empty_cache()

    # torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

    excel_train = "E:\Lucru\ANUL II MASTER\MLAV\ProiectDificil\Pandora_18k_Merged_Augmented\\training_paths_labels.xlsx"
    excel_test = "E:\Lucru\ANUL II MASTER\MLAV\ProiectDificil\Pandora_18k_Merged_Augmented\\testing_paths_labels.xlsx"

    mobile_net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobile_net.to(device)
    torch.backends.cudnn.benchmark = True

    mobile_net.load_state_dict(torch.load('mobileNet_PandoraTrain3.pt'))


    # Training Param
    for param in mobile_net.parameters():
        param.requires_grad = False

    mobile_net.classifier[1] = nn.Linear(in_features=1280, out_features=18, bias=True)

    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(mobile_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optim, step_size = 10, gamma=0.1)

    train_data_dataset = DatasetPandora(excel_train)
    train_loader = DataLoader(train_data_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    nr_epoci = 50
    mobile_net.train()
    mobile_net.cuda()

    model_losses = []
    model_acc = []

    ## TRAIN ##

    for epoca in range(nr_epoci):
        running_loss = 0.0
        running_corrects  = 0.0

        print(" --------  Start Epoca {}".format(epoca+1))
        for data, labels in train_loader:
            batch_data = data.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))

            outputs = mobile_net.forward(batch_data)
        
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
            
            
            _, predictions = torch.max(outputs, 1)

            loss.backward()
            optim.step()
            optim.zero_grad()

            running_loss += loss.item() * 32
            running_corrects += torch.sum(predictions == labels)
        
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        scheduler.step()

        print("-- EPOCA {} ----  Acuratetea: {} - Loss: {}".format(epoca+1, epoch_acc * 100, epoch_loss))
        model_losses.append(epoch_loss)
        model_acc.append(epoch_acc)
        torch.save(mobile_net.state_dict(), ".\\mobileNet_PandoraTrain3.pt" )




    plt.plot(np.linspace(1, nr_epoci, nr_epoci).astype(int), model_losses)
    plt.plot(np.linspace(1, nr_epoci, nr_epoci).astype(int), model_acc)

    with open('./train_loss.txt') as f:
        for item in model_losses:
            f.write("%s\n" % item)

    with open('./train_acc.txt') as f:
        for item in model_acc:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()