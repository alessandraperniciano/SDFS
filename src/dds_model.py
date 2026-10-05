
### Se define el modelo DDS

import torch
import torch.nn as nn


class DDS(nn.Module):
    def __init__(self, n_features_to_select, alpha = 0.9, beta = 2/3, delta = 0.5):
        super(DDS, self).__init__()
        self.n_features_to_select = n_features_to_select
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward(self, x):
        # Se pasa x a 2d
        shape = x.size()
        x = x.view(x.size(0), -1)
        
        # Se diferencian operaciones según fase de entrenamiento o no
        if self.training:

            # Aplicar función de activación sigmoide
            epsilon = 1e-8
            # p2 è un tensore di numeri casuali generati per essere distribuiti uniformemente, con la stessa grandezza di x
            # e nello stesso device di x. Poi i suoi valori vengono troncati affinché rispettino il range [e, 1-e] per evitare
            # problemi con log(0)(= imp) e log(1)(= 0)
            p2 = torch.clip(torch.rand(x.size(), device=x.device), epsilon, 1-epsilon) #para evitar problemas con log(0) o log(1)
            r = torch.log(p2) - torch.log(1-p2)

            xr = (x + self.alpha*r)/self.beta

            # Obtener los k (o M) principales elementos de cada ejemplo
            p = torch.rand(xr.size(0), device=xr.device)

            # Selezionati i k elementi più grandi del tensore xr per colonna per ogni istanza
            # e conserva gli indici
            values, idx = torch.topk(xr, self.n_features_to_select + 1, dim=1)

            # creazione della maschera dove 1 indica
            mask = torch.zeros_like(xr) #se crea la máscara y se completa con el n inicial
            mask.scatter_(1, idx[:, :-1].to(torch.int64), 1)
            xr_masked = -float('inf')*torch.ones_like(xr)
            xr_masked.scatter_(1, idx.to(torch.int64), values)
            s = torch.softmax(xr_masked, dim=1)


            # Trick per evitare l'overfitting
            # Per ogni istanza, a ogni epoca, la mskera seleziona anziché M features tutte le feature
            # per introdurre randomicità e prevenire l'overfitting
            # tensore con valori booleani che indicano dove ogni elemento di p è più piccolo di delta
            compared = torch.lt(p, self.delta) #se hace la comparación y se sustituyen las filas correspondientes de la máscara
            replacement = torch.ones_like(mask[0])
            mask = torch.where(compared.unsqueeze(1), replacement, mask)

        else:
            # Selección de las topk features
            values, idx = torch.topk(x / self.beta, self.n_features_to_select + 1, dim=1)
            # Se crea la máscara con las features seleccionadas
            mask = torch.zeros_like(x)
            mask.scatter_(1, idx[:, :-1].to(torch.int64), 1)
            xr_masked = -float('inf') * torch.ones_like(x)
            xr_masked.scatter_(1, idx.to(torch.int64), values)
            s = torch.softmax(xr_masked, dim=1)

            # Se aplica la sigmoide
            # s = torch.softmax(x/self.beta, dim=1)

        s = self.n_features_to_select * s
        s = s.clamp(max=1)
        
        return s.view(shape), mask.view(shape)
