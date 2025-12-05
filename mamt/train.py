import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .token_config import MAMT_TokenConfig
from .dataset import MAMT_Dataset, split_by_audio
from .transformer import MAMT_Transformer

def collate_batch(batch):
    mels, toks, onsets, offsets = zip(*batch)

    B = len(batch)
    T_list = [m.shape[0] for m in mels]
    L_list = [t.shape[0] for t in toks]

    T_max = max(T_list)
    L_max = max(L_list)

    n_mels = mels[0].shape[1]

    mel_batch = torch.zeros(B, T_max, n_mels, dtype=torch.float32)
    mel_mask  = torch.zeros(B, T_max, dtype=torch.bool)  # True=valid
    
    onset_batch  = torch.zeros(B, T_max, dtype=torch.float32)
    offset_batch = torch.zeros(B, T_max, dtype=torch.float32)
    
    dec_inp = torch.full((B, L_max + 1), MAMT_TokenConfig.PAD_ID, dtype=torch.long)
    dec_tgt = torch.full((B, L_max + 1), MAMT_TokenConfig.PAD_ID, dtype=torch.long)

    for i, (mel, tok, onset, offset) in enumerate(zip(mels, toks, onsets, offsets)):
        T = mel.shape[0]
        L = tok.shape[0]

        mel_batch[i, :T] = torch.from_numpy(mel)
        mel_mask[i, :T] = True

        onset_batch[i, :T]  = torch.from_numpy(onset)
        offset_batch[i, :T] = torch.from_numpy(offset)
        
        # decoder input: [BOS, y0, y1, ...]
        dec_inp[i, 0] = MAMT_TokenConfig.BOS_ID
        dec_inp[i, 1:L+1] = torch.from_numpy(tok)

        # target: [y0, y1, ..., y_{L-1}, PAD]
        dec_tgt[i, :L] = torch.from_numpy(tok)

    return {
        "mel": mel_batch,
        "mel_mask": mel_mask,
        "dec_inp": dec_inp,
        "dec_tgt": dec_tgt,
        "onset_label": onset_batch,
        "offset_label": offset_batch,
    }
    
def do_train(
    mel_dir: str = "./dataset/input",
    tgt_dir: str = "./dataset/target",
    epochs = 10,
    batch_size = 8,
    lr = 1e-4,
    save_dir = "./checkpoints",
    save_name_best = "model_best.pt",
    train_ratio = 0.85,
    val_ratio = 0.1,
    test_ratio = 0.05,
    pretrained_path = None,
    freeze_encoder = False,
    strict_load = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    full_dataset = MAMT_Dataset(mel_dir, tgt_dir)

    train_ds, val_ds, test_ds = split_by_audio(
        full_dataset,
        train_ratio = train_ratio,
        val_ratio = val_ratio,
        test_ratio = test_ratio,
        seed = 42,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collate_batch,
    )
    
    model = MAMT_Transformer(
        d_model=512,
        nhead=8,
        num_enc_layers=4,
        num_dec_layers=4,
        dim_ff=2048,
        dropout=0.2,
    ).to(device)
    
    if pretrained_path is not None and os.path.exists(pretrained_path):
        print(f"[Finetune] loading from: {pretrained_path}")
        state = torch.load(pretrained_path, map_location=device)

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        incompatible = model.load_state_dict(state, strict=strict_load)
        if not strict_load:
            print(f"[Finetune] missing keys: {len(incompatible.missing_keys)}, "
                    f"unexpected keys: {len(incompatible.unexpected_keys)}")
        if freeze_encoder:
            for p in model.cnn2d.parameters():
                p.requires_grad = False
            for p in model.encoder.parameters():
                p.requires_grad = False
            for p in model.enc_pos.parameters():
                p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, save_name_best)

    best_val_loss = float("inf")
    
    loss_fn = nn.CrossEntropyLoss(
        ignore_index = MAMT_TokenConfig.PAD_ID,
        label_smoothing = 0.1
    )
    onoff_bce = nn.BCEWithLogitsLoss(reduction="none")
    lambda_on  = 1.0
    lambda_off = 1.0

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_train_loss = 0.0
        train_tok_list = []
        train_on_list = []
        train_off_list = []

        train_pbar = tqdm(train_loader, desc=f"[Train] epoch {epoch}/{epochs}")
        for batch in train_pbar:
            mel = batch["mel"].to(device)
            mel_mask = batch["mel_mask"].to(device)
            dec_inp = batch["dec_inp"].to(device)
            dec_tgt = batch["dec_tgt"].to(device)
            onset_label  = batch["onset_label"].to(device)
            offset_label = batch["offset_label"].to(device)

            logits, onset_logits, offset_logits = model(mel, mel_mask, dec_inp)
            B, L, V = logits.shape

            loss_tok = loss_fn(
                logits.view(B * L, V),
                dec_tgt.view(B * L),
            )
            onset_loss_raw  = onoff_bce(onset_logits,  onset_label)
            offset_loss_raw = onoff_bce(offset_logits, offset_label)
            
            valid = mel_mask.float()
            loss_on  = (onset_loss_raw  * valid).sum() / valid.sum()
            loss_off = (offset_loss_raw * valid).sum() / valid.sum()

            loss = loss_tok + lambda_on * loss_on + lambda_off * loss_off

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_tok_list.append(loss_tok.item())
            train_on_list.append(loss_on.item())
            train_off_list.append(loss_off.item())
            total_train_loss += loss.item()
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "tok": f"{loss_tok.item():.4f}",
                "on": f"{loss_on.item():.4f}",
                "off": f"{loss_off.item():.4f}",
            })

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_tok_list = []
        val_on_list = []
        val_off_list = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"[Val] epoch {epoch}/{epochs}", leave=False)
            for batch in val_pbar:
                mel = batch["mel"].to(device)
                mel_mask = batch["mel_mask"].to(device)
                dec_inp = batch["dec_inp"].to(device)
                dec_tgt = batch["dec_tgt"].to(device)
                onset_label  = batch["onset_label"].to(device)
                offset_label = batch["offset_label"].to(device)

                logits, onset_logits, offset_logits = model(mel, mel_mask, dec_inp)
                B, L, V = logits.shape
                
                val_loss_tok = loss_fn(
                    logits.view(B * L, V),
                    dec_tgt.view(B * L),
                )
                val_onset_loss_raw  = onoff_bce(onset_logits,  onset_label)
                val_offset_loss_raw = onoff_bce(offset_logits, offset_label)
                
                val_valid = mel_mask.float()
                val_loss_on  = (val_onset_loss_raw  * val_valid).sum() / val_valid.sum()
                val_loss_off = (val_offset_loss_raw * val_valid).sum() / val_valid.sum()
                
                val_loss = val_loss_tok + lambda_on * val_loss_on + lambda_off * val_loss_off
                
                val_tok_list.append(val_loss_tok.item())
                val_on_list.append(val_loss_on.item())
                val_off_list.append(val_loss_off.item())
                total_val_loss += val_loss.item()
                val_pbar.set_postfix({
                    "loss": f"{val_loss.item():.4f}",
                    "tok": f"{val_loss_tok.item():.4f}",
                    "on": f"{val_loss_on.item():.4f}",
                    "off": f"{val_loss_off.item():.4f}",
                })

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        print(f"[EPOCH {epoch}/{epochs}] train_loss {avg_train_loss:.4f}, val_loss {avg_val_loss:.4f}")
        train_tok_mean = np.mean(train_tok_list)
        train_tok_min  = np.min(train_tok_list)
        train_tok_max  = np.max(train_tok_list)
        
        train_on_mean = np.mean(train_on_list)
        train_on_min  = np.min(train_on_list)
        train_on_max  = np.max(train_on_list)

        train_off_mean = np.mean(train_off_list)
        train_off_min  = np.min(train_off_list)
        train_off_max  = np.max(train_off_list)

        val_tok_mean = np.mean(val_tok_list)
        val_tok_min  = np.min(val_tok_list)
        val_tok_max  = np.max(val_tok_list)
        
        val_on_mean = np.mean(val_on_list)
        val_on_min  = np.min(val_on_list)
        val_on_max  = np.max(val_on_list)

        val_off_mean = np.mean(val_off_list)
        val_off_min  = np.min(val_off_list)
        val_off_max  = np.max(val_off_list)

        print(
            f"[TOKEN] train(mean {train_tok_mean:.4f}, min {train_tok_min:.4f}, max {train_tok_max:.4f}) "
            f"val(mean {val_tok_mean:.4f}, min {val_tok_min:.4f}, max {val_tok_max:.4f})"
        )
        print(
            f"[ONSET] train(mean {train_on_mean:.4f}, min {train_on_min:.4f}, max {train_on_max:.4f}) "
            f"val(mean {val_on_mean:.4f}, min {val_on_min:.4f}, max {val_on_max:.4f})"
        )
        print(
            f"[OFFSET] train(mean {train_off_mean:.4f}, min {train_off_min:.4f}, max {train_off_max:.4f}) "
            f"val(mean {val_off_mean:.4f}, min {val_off_min:.4f}, max {val_off_max:.4f})"
        )

        model_save_path = os.path.join(save_dir, f"model_{epoch}.pt")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"best val loss ({best_val_loss:.4f})")
            best = model_save_path
            
        torch.save(model.state_dict(), model_save_path)
        print(f"model saved to {model_save_path} (val loss {avg_val_loss:.4f})")
        print()

    # Test
    model.load_state_dict(torch.load(best, map_location=device))
    model.to(device)
    model.eval()

    total_test_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="[Test]")
        for batch in test_pbar:
            mel = batch["mel"].to(device)
            mel_mask = batch["mel_mask"].to(device)
            dec_inp = batch["dec_inp"].to(device)
            dec_tgt = batch["dec_tgt"].to(device)

            logits, onset_logits, offset_logits = model(mel, mel_mask, dec_inp)
            B, L, V = logits.shape
            
            test_loss = loss_fn(
                logits.view(B * L, V),
                dec_tgt.view(B * L),
            )
            total_test_loss += test_loss.item()

            preds = logits.argmax(dim=-1)
            mask = (dec_tgt != MAMT_TokenConfig.PAD_ID)
            correct = (preds == dec_tgt) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_test_loss = total_test_loss / max(1, len(test_loader))
    test_accuracy = total_correct / max(1, total_tokens)

    print(f"Test result: loss {avg_test_loss:.4f}, token_acc {test_accuracy:.4f}")

    return {
        "best_val_loss": best_val_loss,
        "test_loss": avg_test_loss,
        "test_token_acc": test_accuracy,
        "best_model_path": best_path,
    }
