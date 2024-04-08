import torch
import time
import muspy
import mido
from BusinessLogic.MusicVAE import MusicVAE
from BusinessLogic.MidiHandler import MidiHandler
from DataBase.DB import DB


class BL():
    
    # region: Constructor
    def __init__(self, database : DB, bpm : int = 120, beats_per_bar : int = 4) -> None:
        
        self.database       : DB  = database
        self.bpm            : int = bpm
        self.beats_per_bar  : int = beats_per_bar
        self.ticks_per_bar  : int = self.beats_per_bar * self.database.database_resolution

        self.MidiHandler   : MidiHandler = MidiHandler()
        self.MusicVAE      : MusicVAE    = MusicVAE()
    #endregion

    # region: Setters  
    def set_bpm(self, bpm : int) -> None:
        self.bpm = bpm
        
    def set_beats_per_bar(self, beats_per_bar : int) -> None:
        self.beats_per_bar = beats_per_bar
        self.ticks_per_bar = self.beats_per_bar * self.database.database_resolution
    #endregion
    
    # region: Methods
    def train(self) -> None:
        self.MusicVAE.fit(self.database)
        
    def save_model(self, save_path : str = "./checkpoint.pt") -> None:
        DB.save_model(self.MusicVAE.state_dict(), save_path)

    def load_model(self, load_path : str = "./checkpoint.pt") -> None:
        model = None
        try:
            model = MusicVAE()
            model.load_state_dict(DB.load_model(load_path))
        except:
            print(f"Model could not be loaded from path {load_path}")
            return
        self.MusicVAE = model
        self.MusicVAE.eval()
        
    def generate_melody(self) -> mido.MidiFile:
        melody = self.MusicVAE.generate_sequence()
        music = muspy.from_pianoroll_representation(melody)
        mido_melody = muspy.to_mido(music, use_note_off_message=True)
        return mido_melody

    def play_melody_from_file(self, path : str = "") -> None:
        try:
            mido_data = muspy.read_midi(path)
            mido_data = muspy.to_mido(mido_data,use_note_off_message=True)
            self.MidiHandler.play_mido_object(mido_data)
        except:
            pass
    
    def generate_and_save_melody(self, savepath : str = "") -> None:
        mido_melody = self.generate_melody()
        self.database.save_midi(path = savepath, mido_music = mido_melody)
    
    def generate_and_play_melody(self) -> None:
        mido_melody = self.generate_melody()
        self.MidiHandler.play_mido_object(mido_melody)
        
    def stop_play(self) -> None:
        self.MidiHandler.stop_play()
    #endregion
    