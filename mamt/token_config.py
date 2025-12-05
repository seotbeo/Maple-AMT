class MAMT_TokenConfig:
    STRIDE_SECONDS: float = 1.024
    SEGMENT_SECONDS: float = 2.048
    MAX_TIME_BINS: int = 102

    ON_ID: int = 0
    OFF_ID: int = 1
    TIME_START: int = 2
    NOTE_START: int = TIME_START + MAX_TIME_BINS
    EOS_ID: int = NOTE_START + 128
    END_TIE_ID: int = EOS_ID + 1
    PAD_ID: int = END_TIE_ID + 1
    BOS_ID: int = PAD_ID + 1
    
    VOCAB_SIZE: int = BOS_ID + 1
    
    @classmethod
    def time_resolution(cls) -> float:
        return cls.SEGMENT_SECONDS / cls.MAX_TIME_BINS
    
    @classmethod
    def time_to_token(cls, rel_time: float) -> int:
        t = min(max(rel_time, 0.0), cls.SEGMENT_SECONDS - 1e-6)
        idx = int(t / cls.time_resolution())
        return cls.TIME_START + idx

    @classmethod
    def token_to_time(cls, time_token: int) -> float:
        idx = time_token - cls.TIME_START
        return (idx + 0.5) * cls.time_resolution()

    @classmethod
    def note_to_token(cls, pitch: int) -> int:
        pitch = min(max(pitch, 0), 127)
        return cls.NOTE_START + pitch

    @classmethod
    def token_to_note(cls, token: int) -> int:
        pitch = token - cls.NOTE_START
        pitch = min(max(pitch, 0), 127)
        return pitch

    @classmethod
    def is_time(cls, token: int) -> bool:
        return cls.TIME_START <= token < cls.TIME_START + cls.MAX_TIME_BINS

    @classmethod
    def is_note(cls, token: int) -> bool:
        return cls.NOTE_START <= token < cls.NOTE_START + 128
    
    @classmethod
    def token_to_text(cls, token: int) -> str:
        if token == cls.ON_ID:
            return "<ON>"
        elif token == cls.OFF_ID:
            return "<OFF>"
        elif cls.TIME_START <= token < cls.NOTE_START:
            return f"<TIME {token - cls.TIME_START}>"
        elif cls.NOTE_START <= token < cls.EOS_ID:
            return f"<NOTE {token - cls.NOTE_START}>"
        elif token == cls.EOS_ID:
            return "<EOS>"
        elif token == cls.END_TIE_ID:
            return "<TIE END>"
        elif token == cls.PAD_ID:
            return "<PAD>"
        elif token == cls.BOS_ID:
            return "<BOS>"
        else:
            return f"<UNK {token}>"