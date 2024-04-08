# szakdolgozat
ELTE IK BSc szakdolgozat 2022/23

https://drive.google.com/drive/folders/1dOceOtj6BXIXMlhLFsE7pNILy_Bnt7sE?usp=sharing

# info

- 3 layer architecture: DataBase, BusinessLogic, UserInterface
- unit testing is required for DB & BL, with functions as units, each function should have a test (exactly one per function would be best, no more needed)
- example Midi files in example_midi
- TODO for Jona: data augmentation for proper sota training (transpose, stretch, etc.)
- no good training results expected yet (see augmentation todo), it should be possible though without error. model checkpoint saves to checkpoints/checkpoint.pt by default
- the database stores path strings until it is rendered (unrendered db is lost upon closing the program), then it saves .npy files to database_files. the current default representation is piano roll matrix, with a few 1/4 beats with default resolution = 12 (ticks, that is lines in the matrix, per quarter note. note is on = 1, note is off = 0)
