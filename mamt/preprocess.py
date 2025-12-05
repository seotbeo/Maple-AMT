import numpy as np
import os
import math
import librosa
import pretty_midi
from typing import List, Tuple, Dict
from pathlib import Path
from .audio_config import MAMT_AudioConfig
from .token_config import MAMT_TokenConfig

def collect_piano_notes(pm: pretty_midi.PrettyMIDI) -> List[pretty_midi.Note]:
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append(n)
            
    notes.sort(key=lambda n: n.start)
    return notes

def find_offset(mp3_path, midi_path):
    y1, sr = librosa.load(mp3_path, sr=None)
    pm = pretty_midi.PrettyMIDI(midi_path)
    y2 = pm.synthesize(fs=sr)

    max_shift_sec = 0.3
    ignore_intro_sec = 10.0
    ignore_samples = int(ignore_intro_sec * sr)
    
    if len(y1) > ignore_samples:
        y1_cut = y1[ignore_samples:]
    else:
        y1_cut = y1
    if len(y2) > ignore_samples:
        y2_cut = y2[ignore_samples:]
    else:
        y2_cut = y2
        
    # onset strength
    onset1 = librosa.onset.onset_strength(y = y1_cut, sr = sr, hop_length = MAMT_AudioConfig.HOP)
    onset2 = librosa.onset.onset_strength(y = y2_cut, sr = sr, hop_length = MAMT_AudioConfig.HOP)

    # cross-correlation
    corr = np.correlate(onset1, onset2, mode="full")
    lags = np.arange(-len(onset2) + 1, len(onset1))

    max_shift_frames = int(max_shift_sec * sr / MAMT_AudioConfig.HOP)
    corr[(lags < -max_shift_frames) | (lags > max_shift_frames)] = -np.inf

    best_shift_frames = lags[corr.argmax()]
    time_shift_cut = best_shift_frames * MAMT_AudioConfig.HOP / sr

    return time_shift_cut

def midi_to_token_segments(
    pm: pretty_midi.PrettyMIDI,
    offset: float = 0.0,
) -> List[Tuple[List[int], np.ndarray, np.ndarray]]:
    notes = collect_piano_notes(pm)
    if not notes:
        return []

    if offset != 0.0:
        for inst in pm.instruments:
            for note in inst.notes:
                note.start += offset
                note.end   += offset
                
                note.start = max(0.0, note.start)
                note.end = max(0.0, note.end)
                
    total_len = max(n.end for n in notes)
    num_segments = math.ceil((total_len - MAMT_TokenConfig.SEGMENT_SECONDS) / MAMT_TokenConfig.STRIDE_SECONDS) + 1

    notes_by_pitch: Dict[int, List[pretty_midi.Note]] = {p: [] for p in range(128)}
    for n in notes:
        notes_by_pitch[n.pitch].append(n)

    segments: List[Tuple[List[int], np.ndarray, np.ndarray]] = []

    for seg_idx in range(num_segments):
        seg_start = seg_idx * MAMT_TokenConfig.STRIDE_SECONDS
        seg_end = seg_start + MAMT_TokenConfig.SEGMENT_SECONDS

        tokens: List[int] = []

        n_frames = MAMT_AudioConfig.N_FRAMES
        onset = np.zeros(n_frames, dtype=np.float32)
        offset = np.zeros(n_frames, dtype=np.float32)

        # tie section
        tie_notes = set()
        for pitch in range(128):
            for n in notes_by_pitch[pitch]:
                if n.start < seg_start and n.end > seg_start:
                    tie_notes.add(pitch)
                    break

        if tie_notes:
            for p in sorted(tie_notes):
                tokens.append(MAMT_TokenConfig.note_to_token(p))
        tokens.append(MAMT_TokenConfig.END_TIE_ID)

        # note on/off
        events: List[Tuple[float, str, int]] = [] # time, "on"/"off", pitch

        for n in notes:
            if seg_start <= n.start < seg_end:
                events.append((n.start, "on", n.pitch))
            if seg_start < n.end <= seg_end:
                events.append((n.end, "off", n.pitch))

        events.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1, x[2]))

        for t, typ, pitch in events:
            rel_t = t - seg_start
            frame_idx = MAMT_AudioConfig.PAD_FRAMES + int(round(rel_t * MAMT_AudioConfig.BASE_SR / MAMT_AudioConfig.HOP))
            if 0 <= frame_idx < n_frames:
                if typ == "on":
                    onset[frame_idx] = 1.0
                else:
                    offset[frame_idx] = 1.0

        current_time_token = None
        current_type_token = None

        for t, typ, pitch in events:
            rel_t = t - seg_start
            time_tok = MAMT_TokenConfig.time_to_token(rel_t)
            if time_tok != current_time_token:
                tokens.append(time_tok)
                current_time_token = time_tok
                current_type_token = None

            if typ == "on":
                if current_type_token != MAMT_TokenConfig.ON_ID:
                    current_type_token = MAMT_TokenConfig.ON_ID
                    tokens.append(MAMT_TokenConfig.ON_ID)
                tokens.append(MAMT_TokenConfig.note_to_token(pitch))
            else:
                if current_type_token != MAMT_TokenConfig.OFF_ID:
                    current_type_token = MAMT_TokenConfig.OFF_ID
                    tokens.append(MAMT_TokenConfig.OFF_ID)
                tokens.append(MAMT_TokenConfig.note_to_token(pitch))

        tokens.append(MAMT_TokenConfig.EOS_ID)

        segments.append((tokens, onset, offset))

    return segments

def make_midi_dataset(dir, output_dir, audio_dir, audio_ext = ".mp3", output_log: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(dir):
        for file in files:
            if not file.lower().endswith((".mid", ".midi")):
                continue

            path = os.path.join(root, file)
            audio_path = os.path.join(audio_dir, Path(file).stem + audio_ext)
            try:
                shift = find_offset(audio_path, path)
            except:
                shift = 0.0
            file_name = Path(file).stem
            
            if output_log: print(f"  Processing: {file}, offset: {shift}")
            
            pm = pretty_midi.PrettyMIDI(path)
            token_segments = midi_to_token_segments(pm, shift)

            for idx, (tokens, onset, offset) in enumerate(token_segments):                    
                out_path = f"{output_dir}/{file_name}_{idx}_target.npz"
                np.savez(
                    out_path,
                    tokens = np.array(tokens, dtype=np.int64),
                    onset = onset.astype(np.float32),
                    offset = offset.astype(np.float32),
                )
                
def wav_to_mel(y, sr = MAMT_AudioConfig.BASE_SR, nfft = MAMT_AudioConfig.NFFT, hop = MAMT_AudioConfig.HOP, nmel = MAMT_AudioConfig.NMEL):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=nfft,
        hop_length=hop,
        n_mels=nmel
    )
    mel = librosa.power_to_db(mel)
    return mel

def get_mels(path: str) -> np.ndarray:
    mels = []

    y, sr = librosa.load(path, sr = MAMT_AudioConfig.BASE_SR)
    y = np.pad(y, (MAMT_AudioConfig.PAD_SAMPLES, MAMT_AudioConfig.PAD_SAMPLES), mode='constant')
    
    total_len = len(y)
    num_segments = math.ceil((total_len - MAMT_AudioConfig.SEG_SAMPLES) / MAMT_AudioConfig.STRIDE_SAMPLES) + 1
    
    for idx in range(num_segments):
        start = idx * MAMT_AudioConfig.STRIDE_SAMPLES
        end = start + MAMT_AudioConfig.SEG_SAMPLES
        segment = y[start:end]

        if len(segment) < MAMT_AudioConfig.SEG_SAMPLES:
            pad_len = MAMT_AudioConfig.SEG_SAMPLES - len(segment)
            segment = np.pad(segment, (0, pad_len))
            
        mel = wav_to_mel(segment)
        mel = mel.T
            
        mels.append(mel)
    
    return mels

def save_mel(output_dir, file, path):
    file_name = Path(file).stem

    y, sr = librosa.load(path, sr = MAMT_AudioConfig.BASE_SR)
    y = np.pad(y, (MAMT_AudioConfig.PAD_SAMPLES, MAMT_AudioConfig.PAD_SAMPLES), mode='constant')
    
    total_len = len(y)
    num_segments = math.ceil((total_len - MAMT_AudioConfig.SEG_SAMPLES) / MAMT_AudioConfig.STRIDE_SAMPLES) + 1

    for idx in range(num_segments):
        start = idx * MAMT_AudioConfig.STRIDE_SAMPLES
        end = start + MAMT_AudioConfig.SEG_SAMPLES
        segment = y[start:end]

        if len(segment) < MAMT_AudioConfig.SEG_SAMPLES:
            pad_len = MAMT_AudioConfig.SEG_SAMPLES - len(segment)
            segment = np.pad(segment, (0, pad_len))
            
        mel = wav_to_mel(segment)
        np.savez(f"{output_dir}/{file_name}_{idx}_mel.npz", mel = mel)
    
def make_audio_dataset(dir, output_dir, audio_ext = ".mp3", output_log: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(dir):
        for file in files:
            if not file.lower().endswith(audio_ext):
                continue

            path = os.path.join(root, file)
            if output_log: print(f"  Processing: {file}",)
            save_mel(output_dir, file, path)