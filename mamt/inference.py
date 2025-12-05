import numpy as np
import os
import pretty_midi
import torch
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from .audio_config import MAMT_AudioConfig
from .token_config import MAMT_TokenConfig
from .transformer import MAMT_Transformer
from .preprocess import get_mels

def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    return device

def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    model = MAMT_Transformer(
        d_model=512,
        nhead=8,
        num_enc_layers=4,
        num_dec_layers=4,
        dim_ff=2048,
        dropout=0.2,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def token_segment_to_midi(
    token_segment: List[int],
    pm: pretty_midi.PrettyMIDI,
    active_notes: Dict[int, pretty_midi.Note],
    seg_idx: int,
    velocity: int = 80,
) -> Tuple[pretty_midi.PrettyMIDI, Dict[int, pretty_midi.Note]]:
    MIN_NOTE_DUR = 0.05
    
    if len(pm.instruments) == 0:
        piano_inst = pretty_midi.Instrument(program=0, is_drum=False, name="piano")
        pm.instruments.append(piano_inst)
    else:
        piano_inst = pm.instruments[0]
        
    seg_start = seg_idx * MAMT_TokenConfig.SEGMENT_SECONDS
    seg_end = seg_start + MAMT_TokenConfig.SEGMENT_SECONDS

    i = 0
    tokens = token_segment
    current_time = seg_start
    kind = None
    while i < len(tokens):
        tok = tokens[i]
        if tok == MAMT_TokenConfig.EOS_ID:
            break

        # TIME
        if MAMT_TokenConfig.is_time(tok):
            rel_t = MAMT_TokenConfig.token_to_time(tok)
            current_time = seg_start + rel_t
            i += 1
            kind = None
            continue

        # ON/OFF
        if tok == MAMT_TokenConfig.ON_ID or tok == MAMT_TokenConfig.OFF_ID:
            kind = tok
            i += 1
            continue
            
        if MAMT_TokenConfig.is_note(tok):
            pitch = MAMT_TokenConfig.token_to_note(tok)

            if kind == MAMT_TokenConfig.ON_ID:
                n = pretty_midi.Note(
                    velocity = velocity,
                    pitch = pitch,
                    start = current_time,
                    end = seg_end
                )
                if pitch in active_notes:
                    old = active_notes[pitch]
                    old.end = current_time
                    if old.end - old.start >= MIN_NOTE_DUR:
                        piano_inst.notes.append(old)
                    else:
                        n.start = old.start
                active_notes[pitch] = n

            else:
                if pitch in active_notes:
                    n = active_notes[pitch]
                    n.end = current_time
                    if n.end - n.start >= MIN_NOTE_DUR:
                        piano_inst.notes.append(n)
                    del active_notes[pitch]

            i += 1
            continue

        i += 1
        
    return pm, active_notes

@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    device: str,
    mel: np.ndarray,
    tie: Optional[List[int]],
    max_len: int = 1024,
    bos_id: int = MAMT_TokenConfig.BOS_ID,
    eos_id: int = MAMT_TokenConfig.EOS_ID,
) -> np.ndarray:

    mel_t = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0).to(device)
    B, T, _ = mel_t.shape

    mel_mask = torch.ones((B, T), dtype=torch.bool, device=device)

    dec_list = [bos_id]
    if tie and len(tie) > 0:
        dec_list.extend(tie)
    dec_list.append(MAMT_TokenConfig.END_TIE_ID)
    
    dec_inp = torch.tensor([dec_list], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_len):
        logits, _, _ = model(mel_t, mel_mask, dec_inp)
        next_logits = logits[:, -1, :]
        next_id = next_logits.argmax(dim=-1).item()

        if next_id == eos_id:
            break

        generated.append(next_id)

        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        dec_inp = torch.cat([dec_inp, next_token], dim=1)

    return np.array(generated, dtype=np.int64)

def do_inference(dir: str, output_dir: str, checkpoint_path: str, audio_ext: str = "mp3"):
    os.makedirs(output_dir, exist_ok=True)
    
    device = check_device()
    model = load_model(checkpoint_path, device)
    
    for root, _, files in os.walk(dir):
        for file in files:
            if not file.lower().endswith(audio_ext):
                continue

            path = os.path.join(root, file)
            print(f"{file}",)
            
            mels = get_mels(path)
            pred_tokens = []
            cnt = 0
            mels = mels[0::2]

            pm = pretty_midi.PrettyMIDI()
            active_notes: Dict[int, pretty_midi.Note] = {} # pitch -> note
            tie = []

            for mel in tqdm(mels):
                pred_token = greedy_decode(model, device, mel, tie, max_len=1024)
                pred_tokens.append(pred_token.tolist())
                pm, active_notes = token_segment_to_midi(pred_token.tolist(), pm, active_notes, cnt)
                cnt += 1
                
                tie = [MAMT_TokenConfig.note_to_token(pitch) for pitch in sorted(active_notes.keys())]

            piano_inst = pm.instruments[0]
            for pitch, n in active_notes.items():
                if n.end <= n.start:
                    n.end = n.start + MAMT_TokenConfig.segment_seconds
                piano_inst.notes.append(n)
            active_notes.clear()
                
            pm.write(os.path.join(output_dir, Path(path).stem + "_output.mid"))
        