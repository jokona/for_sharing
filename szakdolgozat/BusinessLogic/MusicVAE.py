import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
from torch.utils.data import Dataset, DataLoader, random_split
from BusinessLogic.Decoder import Decoder
from BusinessLogic.Encoder import Encoder
from BusinessLogic.MidiDataset import MidiDataset
from DataBase.DB import DB

from typing import Tuple

##################
#### MusicVAE ####
##################

class MusicVAE(nn.Module):

    def __init__(self, 
                 seq_len     : int   = 4 * 48, num_pitches : int   = 128 , ticks_per_section : int  = 12  ,
                 batch_size  : int   = 2     , num_epochs  : int   = 10  , test_ratio    : float = 0.3,
                 teacher_forcing : bool = False, shuffle   : bool  = True, loss_average_display_frequency : int  = 10) -> None:
        super().__init__()
        
        self.num_pitches  : int = num_pitches
        self.seq_length   : int = seq_len
        self.num_sections : int = self.seq_length // ticks_per_section
        self.ticks_per_section : int = ticks_per_section

        self.batch_size : int   = batch_size
        self.num_epochs : int   = num_epochs
        self.test_ratio : float = test_ratio

        self.teacher_forcing : bool = teacher_forcing
        self.shuffle : bool = shuffle
        self.loss_average_display_frequency: int = loss_average_display_frequency

        self.Encoder: Encoder = Encoder(num_pitches=num_pitches)
        self.Decoder: Decoder = Decoder(seq_len=seq_len, ticks_per_section=ticks_per_section, num_pitches=num_pitches, batch_size=batch_size)

    def set_batch_size(self, batch_size : int) -> None:
        self.batch_size = batch_size  

    def set_num_epochs(self, num_epochs : int) -> None:
        self.num_epochs = num_epochs
        
    def set_seq_length(self, seq_length : int) -> None:
        self.seq_length = seq_length
    
    def set_teacher_forcing(self, teacher_forcing : bool) -> None:
        self.teacher_forcing = teacher_forcing
        

    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, sigma = self.Encoder(x)

        probs = self.Decoder(z, x, teacher_forcing = self.teacher_forcing)

        return probs, z, mu, sigma
        # probs sum to 1 |-> only argmax can be used
        
    def fit(self, database : DB, num_epochs = None) -> None:
        
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.train()
        self.Encoder.train()
        self.Decoder.train()
        #TODO: fix this if
        if num_epochs is None:
            num_epochs = self.num_epochs

        #region initializing data loaders
        midi_dataset = MidiDataset(database=database, past_beats=self.seq_length // self.ticks_per_section, predicted_beats=1)
        dataset_size = len(midi_dataset)            #number of musics on dataset
        
        test_size  : int = int( self.test_ratio* dataset_size)  #test size length
        train_size : int = dataset_size - test_size       #train data length
        train_dataset, test_dataset = random_split(midi_dataset, [train_size, test_size])
        
        train_loader : DataLoader = DataLoader(train_dataset, shuffle=self.shuffle, batch_size=self.batch_size, num_workers=2)
        test_loader  : DataLoader = DataLoader(test_dataset, shuffle=self.shuffle, batch_size=self.batch_size, num_workers=2)
        #endregion
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss = nn.MSELoss() #nn.CrossEntropyLoss()

        # define sceduler function
        def scheduler_func(epoch) -> float:
            if epoch < num_epochs // 2:
                return 1.0
            else:
                return 0.1

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = scheduler_func)

        loss_list = []
        MSE_list = []
        KLD_list = []    # cross entropy-val a thresholdos megoldást lecsekkolni
        loss_record = [] #loss record will saved as loss_record.pickle

        print("traning start")
        print("batch_size : {}, epochs : {}".format(self.batch_size,num_epochs))
        for i in range(num_epochs):
            print(i)
            x, y = next(iter(train_loader))
            x = x.to(torch.float32)
            x = x.to(self.device)

            #probs, mert ez valószínűségi eloszlás
            prob, z, mu, sigma  = self(x)
            
            MSE = loss(prob,x)
            KLD = (0.5 * torch.mean(mu.pow(2) + sigma.pow(2) - 1 - sigma.pow(2).log()))
            total_loss =  KLD + MSE


            if (i+1) % self.loss_average_display_frequency == 0:
                print(torch.argmax(prob[0][0],dim=-1))
                print(x[0,0])

                avg_loss = np.sum(loss_list)/self.loss_average_display_frequency
                avg_MSE = np.sum(MSE_list)/self.loss_average_display_frequency
                avg_KLD = np.sum(KLD_list)/self.loss_average_display_frequency

                loss_record.append((i,avg_loss))

                torch.save(self.state_dict(),'./checkpoints/checkpoint_epoch_' + str(i+1) + '.pt')
                print("iter : ", i, "avg_loss : ", avg_loss, "avg_MSE : ", avg_MSE, "avg_KLD : ", avg_KLD)
                loss_list = []
                MSE_list = []
                KLD_list = []

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(total_loss.item())
            KLD_list.append(KLD.item())
            MSE_list.append(MSE.item())
        
        print("training done")
   
    def generate_sequence(self, seq_len = 48, x = None) -> np.ndarray:
        self.eval()
        self.Encoder.eval()
        self.Decoder.eval()
        z_random = torch.tensor(np.random.normal(size=(1, self.Encoder.latent_size)))
        z_random = z_random.to(torch.float32)
        probs = self.Decoder(z_random)
        melody_pitches = np.array(torch.argmax(probs[0],dim=-1))
        melody_array = np.zeros((melody_pitches.size, 128), dtype=int)
        melody_array[np.arange(melody_pitches.size), melody_pitches] = 64
        return melody_array
