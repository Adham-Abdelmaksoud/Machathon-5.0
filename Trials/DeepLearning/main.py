from dataloader import read_data
from train import train_cnn
from model import CNN
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    model = CNN()
    imgs, angles = read_data(10)
    trained_model = train_cnn(model, imgs, angles, 100, nn.MSELoss(), optim.Adam())