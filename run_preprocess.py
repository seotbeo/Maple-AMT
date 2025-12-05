import mamt

if __name__ == "__main__":
    print("Processing midis:")
    mamt.make_midi_dataset(
        dir = "./dataset_raw/target",
        output_dir = "./dataset/target",
        audio_dir = "./dataset_raw/input",
        output_log = True)
    
    print()
    print("Processing audios:")
    mamt.make_audio_dataset(
        dir = "./dataset_raw/input",
        output_dir = "./dataset/input",
        output_log = True)