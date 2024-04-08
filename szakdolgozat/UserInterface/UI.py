import tkinter as tk
from tkinter import filedialog, messagebox, Listbox
from PIL import Image, ImageTk

from BusinessLogic.BL import BL
from typing import Union

#region constants
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 1435
IMAGE_NAMES = [
    "key_green_left",
    "key_green_mid",
    "key_green_right",
    "key_green_top",
    "key_red_left",
    "key_red_mid",
    "key_red_right",
    "key_red_top",
]
NOTES_TO_IMAGE_MAP = {
    0: "left",
    1: "top",
    2: "mid",
    3: "top",
    4: "right",
    5: "left",
    6: "top",
    7: "mid",
    8: "top",
    9: "mid",
    10: "top",
    11: "right",
    12: "left",
}

NOTES_Y = 400
NOTES_X = {
    60: 3,
    61: 52,
    62: 71,
    63: 120,
    64: 139,
    65: 207,
    66: 256,
    67: 275,
    68: 324,
    69: 343,
    70: 393,
    71: 412,
    72: 480,
    73: 529,
    74: 548,
    75: 597,
    76: 616,
    77: 685,
    78: 733,
    79: 753,
    80: 802,
    81: 821,
    82: 870,
    83: 889,
    84: 958,
    85: 1007,
    86: 1026,
    87: 1075,
    88: 1094,
    89: 1163,
    90: 1211,
    91: 1231,
    92: 1280,
    93: 1299,
    94: 1348,
    95: 1367,
    #72: 1436,
    #73: 1485,
    #74: 1504,
    #75: 1553,
    #76: 1572,
    #77: 1640,
    #78: 1689,
    #79: 1709,
    #80: 1757,
    #81: 1776,
    #82: 1825,
    #83: 1845,
}
notes_on : list = []
#endregion

class UI:

    def __init__(self, model : BL) -> None:
        self.model : BL = model
        self.model.MidiHandler.events.note_on  += self.add_note
        self.model.MidiHandler.events.note_off += self.remove_note
        
        self.window = tk.Tk()
        self.window.title("Melody AI")
        self.window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        self.frame = tk.Frame(self.window)
        self.frame.pack(side=tk.RIGHT)
        self.canvas = tk.Canvas(self.frame, bg="white", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self.canvas.pack()

        self.keys_img = ImageTk.PhotoImage(Image.open("UserInterface/images/keys.png"))
        self.canvas.create_image(0, 400, image=self.keys_img, anchor="w")

        self.images = {}
        for i in IMAGE_NAMES:
            image = ImageTk.PhotoImage(Image.open(f"UserInterface/images/{i}.png"))
            self.images[i] = image

        self.notes_widgets = {}

        #region buttons for database management
        self.select_database_button = tk.Button(self.canvas, text="ADD TO DB", command = self.add_files, height=2, width=20)
        self.print_database_button  = tk.Button(self.canvas, text="PRINT DB", command = self.print_db, height=2, width=20)
        self.view_and_remove_button = tk.Button(self.window, text="REMOVE FROM DB", command=self.view_and_remove, height=2, width=20)
        self.render_database_button = tk.Button(self.canvas, text="RENDER DB", command = self.model.database.render_database, height=2, width=20) #TODO:fix layer mistake
        #endregion
        
        #region widgets for MIDI input and output device settings
        self.start_midi_in_button = tk.Button(self.canvas, text="START INPUT", command = lambda : self.model.MidiHandler.open_inport(self.midi_in_device_var.get()), height=2, width=18)
        self.midi_in_device_var   = tk.StringVar(value="Choose MIDI IN")
        self.midi_in_device_menu  = tk.OptionMenu(self.window, self.midi_in_device_var, "")
        self.midi_in_device_menu.bind('<ButtonPress-1>', lambda e: self.update_midi_in_devices())

        self.start_midi_out_button = tk.Button(self.canvas, text="START OUTPUT", command= lambda : self.model.MidiHandler.open_outport(self.midi_out_device_var.get()), height=2, width=18)
        self.midi_out_device_var   = tk.StringVar(value="Choose MIDI OUT")
        self.midi_out_device_menu  = tk.OptionMenu(self.window, self.midi_out_device_var, "")
        self.midi_out_device_menu.bind('<ButtonPress-1>', lambda e: self.update_midi_out_devices())
        #endregion
        
        #region buttons for MIDI playback including new melody generation
        self.generate_melody_button = tk.Button(self.window, text="GENERATE MELODY", command=self.generate_and_play_melody, height=2, width=20)
        self.play_midi_button = tk.Button(self.window, text="PLAY MIDI FILE", command=self.play_midi, height=2, width=20)
        self.stop_play_button = tk.Button(self.window, text="STOP CPU OUTPUT", command=self.stop_play, height=2, width=20)
        #endregion
        
        #region buttons for model management and training
        self.save_model_button = tk.Button(self.window, text= "SAVE MODEL", command=self.save_model, height=2, width=20)
        self.load_model_button = tk.Button(self.window, text= "LOAD MODEL", command=self.load_model, height=2, width=20)
        self.train_button = tk.Button(self.window, text= "TRAIN MODEL, epochs=", command=self.validate_int_input_and_train, height=2, width=20)
        self.epochs_entry = tk.Entry(self.window)
        #endregion
        
        #region setting widget positions
        self.canvas.create_window(200,  60,  window=self.midi_in_device_menu)
        self.canvas.create_window(200,  100, window=self.start_midi_in_button)        
        self.canvas.create_window(800,  60,  window=self.midi_out_device_menu)
        self.canvas.create_window(800,  100, window=self.start_midi_out_button)
        self.canvas.create_window(400, 60, window=self.select_database_button)
        self.canvas.create_window(600, 60, window=self.view_and_remove_button)
        self.canvas.create_window(400, 100, window=self.print_database_button)
        self.canvas.create_window(600, 100, window=self.render_database_button)
        self.canvas.create_window(400, 140, window=self.play_midi_button)
        self.canvas.create_window(600, 140, window=self.stop_play_button)
        self.canvas.create_window(500, 180, window=self.generate_melody_button)
        
        
        self.canvas.create_window(1000, 60, window=self.save_model_button)
        self.canvas.create_window(1000, 100, window=self.load_model_button)
        self.canvas.create_window(1000, 140, window=self.train_button)
        self.canvas.create_window(1140, 140, window=self.epochs_entry)
        #endregion
        
    def get_note_image(self, note_id: int) -> Union[None,str]:
        real_note = None
        for n in [96,84,72,60]:
            if note_id >= n:
                real_note = note_id - n
                break
        return NOTES_TO_IMAGE_MAP[real_note] if real_note is not None else None

    def add_note(self, note_id: int, is_green : bool = True) -> None:
        #print("add_note_called", note_id)
        if note_id not in NOTES_X.keys():
            note_id = (note_id - 60) % 36 + 60
        if note_id not in NOTES_X.keys():
            return
        note_name = self.get_note_image(note_id)
        note_image_name = f"key_green_{note_name}" if is_green else f"key_red_{note_name}"
        #print("note_image_name:", note_image_name)
        new_img = self.canvas.create_image(
            NOTES_X[note_id], NOTES_Y, image=self.images[note_image_name], anchor="w"
        )
        self.notes_widgets[note_id, is_green] = new_img

    def remove_note(self, note_id: int, is_green : bool = True) -> None:
        """
        Remove a note widget from the UI
        """
        if note_id not in NOTES_X.keys():
            note_id = (note_id - 60) % 36 + 60
        if note_id not in NOTES_X.keys():
            return
        if (note_id, is_green) in self.notes_widgets:
            self.canvas.delete(self.notes_widgets[note_id, is_green])
            del self.notes_widgets[note_id, is_green]

    def update_midi_in_devices(self) -> None:
        midi_devices = self.model.MidiHandler.update_midi_in_devices()
        self.midi_in_device_var.set(midi_devices[0] if midi_devices else "No MIDI devices available")
        self.midi_in_device_menu['menu'].delete(0, 'end')
        for device in midi_devices:
            self.midi_in_device_menu['menu'].add_command(label=device, command=tk._setit(self.midi_in_device_var, device))
    
    def update_midi_out_devices(self) -> None:
        midi_devices = self.model.MidiHandler.update_midi_out_devices()
        self.midi_out_device_var.set(midi_devices[0] if midi_devices else "No MIDI devices available")
        self.midi_out_device_menu['menu'].delete(0, 'end')
        for device in midi_devices:
            self.midi_out_device_menu['menu'].add_command(label=device, command=tk._setit(self.midi_out_device_var, device))
            
    def add_files(self) -> None:
        file_paths = filedialog.askopenfilenames(filetypes=[("MIDI Files", ["*.mid", "*.Mid","*.midi","*.Midi","*.MID","*.MIDI"])], multiple = True) 
        for path in file_paths:
            
            self.model.database.add_midi_path(path) #TODO:fix layer mistake
       
    def generate_and_play_melody(self):
        self.model.generate_and_play_melody()
    
    def load_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Checkpoint file", ["*.pt"])])
        self.model.load_model(path)
        
    def save_model(self) -> None:
        try:
            path : str = filedialog.asksaveasfilename(filetypes=[("Checkpoint file", ["*.pt"])], defaultextension = ".pt")
            self.model.save_model(path)
            messagebox.showinfo("Success", f"Model saved to {path}")
        except:
            messagebox.showerror("Error", f"Error saving model. Please try again.")
       
    def play_midi(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("MIDI Files", ["*.mid", "*.Mid","*.midi","*.Midi","*.MID","*.MIDI"])])
        self.model.play_melody_from_file(path)
    
    def stop_play(self) -> None:
        self.model.stop_play()   
      
    def print_db(self) -> None:
        #for path in self.model.database.midi_paths: #TODO:fix layer mistake
        #    print(path)
        
        popup_window = tk.Toplevel(self.window)

        scrollbar = tk.Scrollbar(popup_window, orient="vertical")
        listbox = tk.Listbox(popup_window, width=50, height=20, yscrollcommand=scrollbar.set, selectmode=tk.EXTENDED)
        scrollbar.config(command=listbox.yview)
        scrollbar.pack(side="right", fill="y")
        listbox.pack(side="left",fill="both", expand=True)
        
        # Add items to the listbox
        items = self.model.database.midi_paths #TODO:fix layer mistake
        for item in items:
            listbox.insert(tk.END, item)
            
        # Create a button to get the selection
        button = tk.Button(popup_window, text="OK", command=popup_window.destroy)
        button.pack(side=tk.BOTTOM, pady=10)
            
    def print_selection(self) -> None:
        param_list = self.model.database.midi_paths #TODO:fix layer mistake
        selected_params = messagebox.askquestion("Select Parameters", "Select the parameters you want to use", 
                                                  icon="question", type="yesno", 
                                                  detail="\n".join(param_list))
        
        # Convert the user selection to a list and return it
        print(*[param for param in param_list if selected_params == "yes"])
        
    def view_and_remove(self) -> None:
        def remove_chosen() -> None:
            selected_paths = [listbox.get(idx) for idx in listbox.curselection()]
            self.model.database.remove_midis(midi_paths = selected_paths) #TODO:fix layer mistake
            popup_window.destroy()

        popup_window = tk.Toplevel(self.window)

        scrollbar = tk.Scrollbar(popup_window, orient="vertical")
        listbox = tk.Listbox(popup_window, width=50, height=20, yscrollcommand=scrollbar.set, selectmode=tk.EXTENDED)
        scrollbar.config(command=listbox.yview)
        scrollbar.pack(side="right", fill="y")
        listbox.pack(side="left",fill="both", expand=True)
        
        # Add items to the listbox
        items = self.model.database.midi_paths #TODO:fix layer mistake
        for item in items:
            listbox.insert(tk.END, item)
            
        # Create a button to get the selection
        button = tk.Button(popup_window, text="Remove chosen files", command=remove_chosen)
        button.pack(side=tk.BOTTOM, pady=10)
        
    def validate_int_input_and_train(self) -> None:
        try:
            value = int(self.epochs_entry.get())
            self.model.set_num_epochs(value)
            self.model.train()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter an integer.")