import torch 
import torch.nn as nn
from typing import Tuple

#################
#### Encoder ####
#################

class Encoder(nn.Module):
    def __init__(self, num_pitches = 128):
        super().__init__()

        #region parameters
        self.num_pitches : int  = num_pitches
        self.num_layers  : int  = 2
        self.hidden_size : int  = 2048
        self.latent_size : int  = 512
        # Paper : two-layer bidirectional LSTM network, LSTM state size of 2048 for all layers and 512 latent dimensions
        #endregion
        
        #region encoder architecture layers
        self.encoder_lstm : nn.LSTM = nn.LSTM(input_size=self.num_pitches,
                                    hidden_size=self.hidden_size,
                                    num_layers = self.num_layers,
                                    bidirectional=True,
                                    batch_first = True
                                    )

        self.fc_mu    : nn.Linear   = nn.Linear(self.hidden_size * 2, self.latent_size)
        self.fc_sigma : nn.Linear   = nn.Linear(self.hidden_size * 2, self.latent_size)
        self.softplus : nn.Softplus = nn.Softplus()
        #endregion
        
    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
    #TODO named tensors make code more readable
    #region 1. LSTM ENCODER forward
        # Paper :
        # We process an input sequence x = {x1, x2, . . . , xT } to obtain the final state vectors
        # →hT , ←hT from the second bidirectional LSTM layer. These are then concatenated to produce hT
        
        # get LSTM state vectors
        # in  x = (batch, seq_len, num_pitches)
        _, (h_all, _) = self.encoder_lstm(x)
        # out h_all = (2*num_layers, batch, hidden_size)
        
        # get final state vectors →hT and ←hT. Concatenate them to produce hT 
        # in  h_all = (2*num_layers, batch, hidden_size)
        h_T_left  = h_all[2]                        # last layer final output 1 of 2 bidirectional
        h_T_right = h_all[3]                        # last layer final output 2 of 2 bidirectional
        h_T = torch.cat((h_T_left,h_T_right),1)     # last layer final output concatenated
        # out h_T = (batch, 2*hidden_size)
    #endregion
        
    #region 2. LATENT Z EMBEDDING forward
        # in h_T = (batch, 2*hidden_size)
        mu    : torch.Tensor = self.fc_mu(h_T)          # Paper : fed into two fullyconnected layers to produce the latent distribution parameters µ and σ -> get "µ" : Equation (6)        
        sigma : torch.Tensor = self.fc_sigma(h_T)       # Paper : fed into two fullyconnected layers to produce the latent distribution parameters µ and σ -> get "σ" : Equation (7)
        sigma = self.softplus(sigma)                    # Paper : equation (7) # softplus activation function # to ensure sigma is positive
        # out mu = (batch, latent_size) , sigma = (batch, latent_size)
        
        # in mu = (batch, latent_size) , sigma = (batch, latent_size)
        z : torch.Tensor = self.reparameterize(mu, sigma)          # Paper : As is standard in VAEs, µ and σ then parametrize the latent distribution as in Eq. (2)
        # out z = (batch, latent_size)
        
        return z, mu, sigma                         # Encoder output
    #endregion

    #region utils
    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)                 # normal distribution with mean 0 and variance 1
        return mu + torch.mul(sigma,eps)            # Paper : equation (2) # Hadamard product (element-wise product)
    #endregion