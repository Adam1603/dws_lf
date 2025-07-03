# train.py
import torch
from torch.utils.data import DataLoader
from models.cnn_stream import CNNStream
from models.vit_stream import ViTStream
from models.dws_lf_block import DWSLFBlock
from utils.dataset import load_dataset
from utils.train_utils import train_one_epoch, evaluate

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ham10000', help='Dataset name')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

# Load dataset
train_loader, val_loader = load_dataset(args.dataset, batch_size=args.batch_size)

# Define models
cnn_model = CNNStream()
vit_model = ViTStream()
fusion_model = DWSLFBlock(in_channels1=512, in_channels2=512, num_classes=7)

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model.to(device)
vit_model.to(device)
fusion_model.to(device)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
params = list(cnn_model.parameters()) + list(vit_model.parameters()) + list(fusion_model.parameters())
optimizer = torch.optim.Adam(params, lr=args.lr)

# Training loop
for epoch in range(args.epochs):
    fusion_model.train()
    train_one_epoch(cnn_model, vit_model, fusion_model, train_loader, criterion, optimizer, epoch, device)
    evaluate(cnn_model, vit_model, fusion_model, val_loader, criterion, device)

# Save model
torch.save(fusion_model.state_dict(), f'{args.dataset}_dws_lf_model.pth')
