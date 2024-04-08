
import os
import numpy as np
from typing import Tuple, Dict
from torch.utils.data import Dataset

from DataBase.DB import DB

class MidiDataset(Dataset):
    def __init__(self, database : DB, past_beats: int = 3, predicted_beats: int = 1) -> None:
        self.npy_dir    : str = database.database_folder
        self.resolution : int = database.database_resolution
        self.past_beats : int = past_beats
        self.predicted_beats: int  = predicted_beats
        #TODO : understand self.beats
        self.beats: Dict[str, int] = {melody_path: (melody_length // self.resolution) for (midi_path, (melody_path, melody_length)) in database.saved_melodies.items()}

    def __len__(self) -> int:
        sum = 0
        for _ , value in self.beats.items():
            sum = sum + 1 + (value - (self.past_beats + self.predicted_beats))
        return sum

    def __getitem__(self, idx : int) -> Tuple[np.ndarray, np.ndarray]:
        if idx > len(self) - 1:
            raise IndexError
        sum = 0
        past_idx_start = 0
        pred_idx_start = past_idx_start + self.past_beats
        filename = list(self.beats.keys())[0]
        for key, value in self.beats.items():
            if sum + 1 + (value - (self.past_beats + self.predicted_beats)) < idx + 1:
                sum = sum + 1 + (value - (self.past_beats + self.predicted_beats))
            else:
                filename = key
                past_idx_start = (idx + 1) - sum - 1
                pred_idx_start = past_idx_start + self.past_beats
                break
        notes = np.load(os.path.join(self.npy_dir,filename))
        beatlength = self.resolution
        past = notes[beatlength * past_idx_start : beatlength * pred_idx_start]
        pred = notes[beatlength * pred_idx_start : beatlength * (pred_idx_start + self.predicted_beats)]
                                      
        # out past = (past_beats * beatlength, 128), pred = (predicted_beats * beatlength, 128)
        return past, pred