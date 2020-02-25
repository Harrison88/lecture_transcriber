from pathlib import Path

import pydub
import numpy as np


def prepare_audio(audio_filepath, desired_framerate):
    audio_file = Path(audio_filepath)
    audio = pydub.AudioSegment.from_file(str(audio_filepath))
    
    print("Audio duration before framerate change:", audio.duration_seconds)
    audio = audio.set_frame_rate(desired_framerate)
    audio_seconds = audio.duration_seconds
    print("Audio duration after change:", audio_seconds)
    
    split_audio = pydub.silence.split_on_silence(audio, silence_thresh=-70, min_silence_len=500)
    print("Summed audio duration after split:", sum(audio.duration_seconds for audio in split_audio))
    split_audio = [np.frombuffer(a._data, np.int16) for a in split_audio]
    return split_audio, audio_seconds
