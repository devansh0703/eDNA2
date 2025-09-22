import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import numpy as np
from model import SequenceAutoencoder, one_hot_encode

class SequenceDataset(Dataset):
    def __init__(self, fastq_file):
        self.sequences = []
        for record in SeqIO.parse(fastq_file, "fastq"):
            self.sequences.append(str(record.seq))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded = one_hot_encode(seq)
        return encoded

def train_autoencoder(fastq_file, epochs=50, batch_size=32, latent_dim=64):
    dataset = SequenceDataset(fastq_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SequenceAutoencoder(latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), 'models/autoencoder.pth')
    print("Model saved to models/autoencoder.pth")

if __name__ == "__main__":
    train_autoencoder('data/filtered.fastq')