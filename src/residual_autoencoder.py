
### Se define un autoencoder residual

import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim1=200, hidden_dim2=100, hidden_dim3=50, hidden_dim4=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.activation = nn.SiLU()

    def forward(self, x):
        oc1 = self.activation(self.bn1(self.fc1(x)))
        oc2 = self.activation(self.bn2(self.fc2(oc1)))
        oc3 = self.activation(self.bn3(self.fc3(oc2)))
        oc4 = self.fc4(oc3)
        return (oc1, oc2, oc3, oc4)

class Decoder(nn.Module):
    def __init__(self, hidden_dim4=20, hidden_dim3=50, hidden_dim2=100, hidden_dim1=200, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim4, hidden_dim3)
        self.bn1 = nn.BatchNorm1d(hidden_dim3)
        self.fc2 = nn.Linear(hidden_dim3, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim1)
        self.bn3 = nn.BatchNorm1d(hidden_dim1)
        self.fc4 = nn.Linear(hidden_dim1, output_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        oc1, oc2, oc3, oc4 = x
        y1 = self.activation(self.bn1(self.fc1(oc4)) + oc3)
        y2 = self.activation(self.bn2(self.fc2(y1)) + oc2)
        y3 = self.activation(self.bn3(self.fc3(y2)) + oc1)
        y4 = self.fc4(y3)
        return y4



class ResidualAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim1=200, hidden_dim2=100, hidden_dim3=50, hidden_dim4=20):
        super(ResidualAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4)
        self.decoder = Decoder(hidden_dim4, hidden_dim3, hidden_dim2, hidden_dim1, output_dim=input_dim)

    def forward(self, x):
        shape = x.size()
        x = x.view(shape[0], -1) #se aplana la imagen
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(shape)
