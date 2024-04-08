from events import Events
import threading
import mido
from typing import List

class MidiHandler:
    def __init__(self):
        self.current_inport  = None
        self.current_outport = None
        self.in_devices  : List[str]  = mido.get_input_names()
        self.out_devices : List[str]  = mido.get_output_names()
        self.playing  : bool = False
        
        self.in_notes_on  : List = []
        self.out_notes_on : List = []
        self.events : Events  = Events()
    
    def update_midi_in_devices(self) -> List[str]:
        self.in_devices = mido.get_input_names()
        return self.in_devices
        
    def update_midi_out_devices(self) -> List[str]:
        self.out_devices = mido.get_output_names()
        return self.out_devices

    def open_inport(self, inport_name : str) -> None:
        if self.current_inport is not None:
            self.current_inport.close()
        self.current_inport = mido.open_input(inport_name, callback = self.note_in_handler)
        
    def open_outport(self, outport_name : str) -> None:
        if self.current_outport is not None:
            self.current_outport.close()
        self.current_outport = mido.open_output(outport_name)
        
    def note_in_handler(self, note: mido.Message) -> None:
        if note.type in ["note_on", "note_off"]:
            note_id = int(note.note) if note.note is not None else -1
            if note.type == "note_on":
                self.events.note_on(note_id = note_id, is_green = True)
                if note.note not in self.in_notes_on:
                    self.in_notes_on.append(note.note)
                    
            elif note.type == "note_off":
                self.events.note_off(note_id = note_id, is_green = True)
                if note_id in self.in_notes_on:
                    self.in_notes_on.remove(note_id)
                    
    def note_out_handler(self, note: mido.Message) -> None:
        if note.type in ["note_on", "note_off"]:
            note_id = int(note.note) if note.note is not None else -1
            if note.type == "note_on":
                self.events.note_on(note_id = note_id, is_green = False)
                self.current_outport.send(mido.Message('note_on', note = note.note))
                if note.note not in self.out_notes_on:
                    self.out_notes_on.append(note.note)
            elif note.type == "note_off":
                self.events.note_off(note_id = note_id, is_green = False)
                self.current_outport.send(mido.Message('note_off', note = note.note))
                if note_id in self.out_notes_on:
                    self.out_notes_on.remove(note_id)
                    
    def play_mido_object(self, mid : mido.MidiFile) -> None:
        if self.current_outport is None:
            return
                
        def play() -> None:    
            for note in mid.play():
                if self.playing == False:
                    self.current_outport.reset()
                    break 
                self.note_out_handler(note)
            self.playing = False
            return
        
        if not self.playing:
            self.playing = True
            play_thread = threading.Thread(target=play)
            play_thread.start()
            
    def stop_play(self)->None:
        self.playing = False
        for note_id in range(128):
            self.events.note_off(note_id = note_id, is_green = False)

    #TODO: if device disconnects, send all notes off and the like