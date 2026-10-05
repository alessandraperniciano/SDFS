
### Se define un autoencoder simple

import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim1=200, hidden_dim2=100, hidden_dim3=50, hidden_dim4=20):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim3),
            nn.Linear(hidden_dim3, hidden_dim4)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    

class Decoder(nn.Module):
    def __init__(self, hidden_dim4=20, hidden_dim3=50, hidden_dim2=100, hidden_dim1=200, output_dim=784):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim4, hidden_dim3),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim3),
            nn.Linear(hidden_dim3, hidden_dim2),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Linear(hidden_dim1, output_dim)
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim1=200, hidden_dim2=100, hidden_dim3=50, hidden_dim4=20):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4)
        self.decoder = Decoder(hidden_dim4, hidden_dim3, hidden_dim2, hidden_dim1, input_dim)

    def forward(self, x):
        shape = x.size()
        x = x.view(shape[0], -1) #se aplana la imagen
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(shape)

