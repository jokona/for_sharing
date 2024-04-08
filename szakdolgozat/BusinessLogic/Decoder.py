import torch
import torch.nn as nn
from typing import Tuple, Union


#################
#### Decoder ####
#################

class Decoder(nn.Module):
    def __init__(self, seq_len : int = 48, ticks_per_section : int = 12, num_pitches : int = 128, batch_size : int = 64) -> None:
        super().__init__()
        
        #region parameters
        self.seq_len            : int = seq_len
        self.num_pitches        : int = num_pitches
        self.ticks_per_section  : int = ticks_per_section
        self.num_sections       : int = self.seq_len // self.ticks_per_section
        self.batch_size         : int = batch_size
        self.latent_size        : int = 512
        # latent size should equal latent size of encoder
        self.conductor_hidden      : int = 1024
        self.conductor_num_layers  : int = 2
        self.conductor_output_size : int = 512
        # Paper : we use a two-layer unidirectional LSTM for the conductor with a hidden state size of 1024 and 512 output dimensions
        self.decoder_hidden     : int = 1024
        self.decoder_num_layers : int = 2
        # Paper : we used a 2-layer LSTM with 1024 units(hidden state) per layer for the decoder RNN
        #endregion

        #region conductor architecture layers
        self.conductor_init_first_hidden = nn.Sequential(nn.Linear(self.latent_size,self.conductor_hidden),nn.Tanh())
        self.conductor_lstm = nn.LSTM(input_size = self.num_pitches,
                                      hidden_size = self.conductor_hidden,
                                      num_layers = self.conductor_num_layers,
                                      batch_first = True)
        self.conductor_output = nn.Linear(self.conductor_hidden,self.conductor_output_size)
        #endregion

        #region conductor's output decoder architecture layers
        self.decoder_init_first_hidden = nn.Sequential(nn.Linear(self.conductor_output_size, self.decoder_hidden),
                                                   nn.Tanh())
        self.decoder_lstm = nn.LSTM(input_size =self.conductor_output_size + self.num_pitches,
                                    hidden_size = self.decoder_hidden,
                                    num_layers = self.decoder_num_layers,
                                    batch_first = True)
        #decoder head
        self.fc_head = nn.Linear(self.decoder_hidden,self.num_pitches)
        self.softmax = nn.Softmax(dim=-1)
        #endregion

        #region start of sequence tokens
        self.decoder_start_token = torch.zeros(self.batch_size,1,self.num_pitches)
        self.decoder_start_token_eval = torch.zeros(1,1,self.num_pitches)
        #endregion

    def forward(self, z : torch.Tensor, x : Union[None, torch.Tensor] = None, teacher_forcing : bool = False) -> torch.Tensor: # Training : teacher forcing = True , generating : teacher forcing = False
    #region 1. CONDUCTOR forward
        # Paper :
        # the latent vector z is passed through a fully-connected layer
        # followed by a tanh activation to get the initial state of a “conductor” RNN.
        # The conductor RNN produces U embedding vectors c = {c1, c2, . . . , cU }, one for each subsequence. # U is num_bars in Decoder
        
        # in  z = (batch, latent_size) as hidden states
        if self.training:
            conductor_input = nn.Parameter(torch.zeros(self.batch_size, self.num_sections, self.num_pitches)) # out conductor_input = (batch, num_bars, num_pitches)
        else:
            conductor_input = nn.Parameter(torch.zeros(1, self.num_sections, self.num_pitches))           
        z = z.unsqueeze(0)                                                                            # out z = (1, batch, latent_size)
        z_as_hidden_layer1 = self.conductor_init_first_hidden(z)                                      # out z_as_hidden_layer1 = (1, batch, conductor_hidden)
        conductor_initial_hidden_state = self.initialize_state(z_as_hidden_layer1, num_layers = self.conductor_num_layers)
        # out conductor_input = (batch, num_bars, num_pitches) , conductor_initial_hidden_state = (conductor_num_layers, batch, coductor_hidden)
        
        # in  conductor_input = (batch, num_bars, num_pitches) , conductor_initial_hidden_state = (conductor_num_layers, batch, coductor_hidden)
        c, _ = self.conductor_lstm(conductor_input,conductor_initial_hidden_state)
        # out c = (batch, num_bars, conductor_hidden)
        
        # Paper
        # Once the conductor has produced the sequence of embedding vectors c, 
        # each one is individually passed through a shared fully-connected layer 
        # followed by a tanh activation to produce initial states for a final bottom-layer decoder RNN.
        
        # in  c = (batch, num_bars, conductor_hidden)
        c = self.conductor_output(c)
        # out c = (batch, num_bars, conductor_output_size)
    #endregion
   
    #region 2. CONDUCTOR'S OUTPUT DECODER forward
        # Paper :
        # The decoder RNN then autoregressively produces a sequence of distributions 
        # over output tokens for each subsequence yu via a softmax output layer.
        if self.training:
            probs = torch.zeros((self.batch_size,0,self.num_pitches))
        else:
            probs = torch.zeros((1,0,self.num_pitches))
        if self.training:
            decoder_output = self.decoder_start_token
        else:
            decoder_output = self.decoder_start_token_eval
        for note_idx in range(self.seq_len):

            bar_idx = note_idx // self.ticks_per_section

            if note_idx % self.ticks_per_section == 0:
                bar_state = c[:,bar_idx,:].unsqueeze(0)                     # out bar_state = (1, batch, conductor_output_size)
                hidden_layer1 = self.decoder_init_first_hidden(bar_state)   # out hidden_layer1 = (1, batch, decoder_hidden)
                decoder_state = self.initialize_state(hidden_layer1, num_layers = self.decoder_num_layers)
                bar_state = bar_state.permute(1,0,2)                        # out bar_state = (batch, 1, conductor_output_size)
                # out bar_state = (batch,1, conductor_output_size), decoder_state = (decoder_num_layers, batch, decoder_hidden)
                 
            # Paper : the current conductor embedding cu is concatenated with the previous output token to be used as the input.
            if teacher_forcing and note_idx != 0:
                prev_out = x[:,note_idx-1,:].unsqueeze(1)                   # out (batch, 1, num_pitches)
                decoder_input = torch.cat((bar_state, prev_out),dim=-1)
                # out decoder_input = (batch, 1, conductor_output_size + num_pitches) 
            else:
                decoder_input = torch.cat((bar_state, decoder_output), dim=-1)
                # out decoder_input = (batch, 1, conductor_output_size + num_pitches)
            
            # in decoder_input = (batch, 1, conductor_output_size + num_pitches), decoder_state = (decoder_num_layers, batch, decoder_hidden)
            decoder_output, decoder_state = self.decoder_lstm(decoder_input, decoder_state) # out decoder_output = (batch, 1, decoder_hidden)
            decoder_output = self.softmax(self.fc_head(decoder_output))
            # out decoder_output = (batch, 1, num_pitches), decoder_state = (decoder_num_layers, batch, decoder_hidden)
            
            
            prob  = decoder_output
            probs = torch.cat((probs,prob),dim = 1)
            # out probs = (batch, seq_len, num_pitches)

        return probs
    #endregion
    
    #region utils
    def initialize_state(self, hidden_state_layer1 : torch.Tensor, num_layers : int) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_states = [hidden_state_layer1]
        for i in range(num_layers - 1):
            layer_states.append(torch.zeros_like(hidden_state_layer1))
        
        hidden_state = torch.cat(layer_states, dim = 0)
        cell_state = torch.zeros_like(hidden_state)
        
        state = (hidden_state, cell_state)
        return state
    #endregion