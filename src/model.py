import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceAutoencoder(nn.Module):
    def __init__(self, seq_length=202, latent_dim=64):
        super(SequenceAutoencoder, self).__init__()
        self.seq_length = seq_length
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * seq_length, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * seq_length),
            nn.ReLU(),
            nn.Unflatten(1, (64, seq_length)),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 4, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()  # For one-hot reconstruction
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def one_hot_encode(sequence):
    """
    One-hot encode a DNA sequence.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}  # N as A
    encoded = torch.zeros(4, len(sequence))
    for i, base in enumerate(sequence.upper()):
        if base in mapping:
            encoded[mapping[base], i] = 1
    return encoded

def decode_one_hot(encoded):
    """
    Decode one-hot to sequence.
    """
    bases = ['A', 'C', 'G', 'T']
    seq = ''
    for i in range(encoded.shape[1]):
        idx = torch.argmax(encoded[:, i])
        seq += bases[idx]
    return seq