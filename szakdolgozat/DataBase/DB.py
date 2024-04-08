import os
import pickle
import muspy
import mido
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, OrderedDict
from operator import attrgetter

class DB:
    def __init__(self, database_folder      : str = os.path.abspath("./database_files"),
                       database_info_file   : str = os.path.join(os.path.abspath("./database_files"),"_data_info.pkl"),
                       database_resolution  : int = 12)  -> None:
        
        self.midi_paths      : List[str] = []
        self.database_folder : str  = database_folder
        self.database_info_file  : str  = database_info_file
        self.database_resolution : int  = database_resolution
        self.saved_melodies      : Dict[str, Tuple[str, int]] = {}
        if not os.path.exists(self.database_folder):
            os.makedirs(self.database_folder)
            self.update_info_file()
        else:
            if os.path.exists(self.database_info_file):
                with open(self.database_info_file, 'rb') as file:
                    self.saved_melodies = pickle.load(file)
                    for midi_path in self.saved_melodies.keys():
                        self.add_midi_path(midi_path)
            else:
                self.update_info_file()

    def set_data_folder(self, folder_path : str) -> None:
        self.database_folder = folder_path
     
    def add_midi_path(self, path : str) -> None:
        if path not in self.midi_paths:
            self.midi_paths.append(path)
    
    def remove_midis(self, midi_paths : List[str]) -> None:
        def _remove_midi(midi_path : str) -> None:
            if midi_path in self.midi_paths:
                self.midi_paths.remove(midi_path)
            if midi_path in self.saved_melodies.keys():
                try:
                    os.remove(self.saved_melodies[midi_path][0])
                    self.saved_melodies.pop(midi_path)
                except:
                    pass
        for midi_path in midi_paths:
            _remove_midi(midi_path)
        self.update_info_file()
      
    #region util functions
    def get_music_with_resolution_compatibility_check(self, path : str) -> Tuple[muspy.Music, bool]:      
        music = muspy.read_midi(path)
        if music.resolution % self.database_resolution != 0:
            return music, False
        gap = music.resolution // self.database_resolution
        music.infer_barlines_and_beats()
        notes = []
        for track in music.tracks:
            notes.extend(track.notes)
        if not notes:
            return music, False
        notes.sort(key=attrgetter("time", "pitch", "duration", "velocity"))

        i = 0       # index for notes
        j = 0       # index for beats
        while(i < len(notes)):
            if(j < len(music.beats) - 1 and notes[i].time > music.beats[j + 1].time):
                j = j + 1
            if((notes[i].time - music.beats[j].time) % gap != 0):
                return music, False                
            i = i + 1
        return music, True
    
    def check_in4(self, music : muspy.Music) -> bool:
        in4 = True
        for ts in music.time_signatures:
            if ts.numerator != ts.denominator:
                in4 = False
                break
        return in4    
    
    def extract_melody(self, music : muspy.Music) -> np.ndarray:
        muspy.adjust_resolution(music=music, target=self.database_resolution)        
        pianoroll = muspy.to_pianoroll_representation(music)
        pianoroll = np.where(pianoroll > 0, 1, 0)
        melody_idxs = 127 - np.argmax(np.flip(pianoroll, axis = 1),axis = 1)
        melody_mask = np.zeros(shape=pianoroll.shape, dtype = int)
        melody_mask[np.arange(melody_idxs.size), melody_idxs] = 1
        melodyroll = np.where(melody_mask == 1, pianoroll, 0)
        return melodyroll
    
    def update_info_file(self) -> None:
        with open(self.database_info_file, 'wb') as file:
            pickle.dump(self.saved_melodies, file)

    #endregion
    
    def save_melody_upon_compatibility(self, midi_in_path : str, melody_out_path : str) -> Tuple[bool, int]:
        save_success = False
        melody_length = 0
        music, compatible = self.get_music_with_resolution_compatibility_check(midi_in_path)
        in4 = self.check_in4(music)
        if compatible and in4:
            melody = self.extract_melody(music)
            np.save(file = melody_out_path, arr = melody)
            melody_length = len(melody)
            save_success = True
        return save_success, melody_length
    
    def render_database(self) -> None:
        for _, midi_in_path in enumerate(self.midi_paths):
            try:
                folders, filename = os.path.split(midi_in_path)
                _, last_folder    = os.path.split(folders)
                melody_path = os.path.abspath(os.path.join(self.database_folder, last_folder + '_' + filename + '.npy'))
                save_success, melody_length = self.save_melody_upon_compatibility(midi_in_path, melody_path)
                if save_success:
                    self.saved_melodies[midi_in_path] = (melody_path, melody_length)
                    self.update_info_file()
            except:
                #TODO: log faulty midi and remove from database
                pass
    
    def save_midi(self, mido_music: mido.MidiFile, path : str) -> None:
        try:
            mido_music.save(path)
        except:
            pass

    def load_model(self, load_path : str = "./checkpoints/checkpoint.pt") -> Any:
        try:
            return torch.load(load_path)
        except:
            raise Exception("Model could not be loaded from path " + load_path)
        
    def save_model(self, state_dict : OrderedDict[str, torch.Tensor], save_path : str = "./checkpoint.pt") -> None:
        torch.save(state_dict, save_path)