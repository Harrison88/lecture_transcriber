from pathlib import Path
import argparse

import pydub
import numpy as np
from tqdm import tqdm


def preprocess_audio(audio_filepath, desired_framerate):
    audio, audio_seconds = get_audiosegment(audio_filepath, desired_framerate)
    
    split_audio = split_on_silence(audio)
    split_audio = audiosegments_to_np(split_audio)
    return split_audio, audio_seconds

def split_on_silence(audio):
    split_audio = pydub.silence.split_on_silence(audio, silence_thresh=-75, min_silence_len=700)
    summed_duration = sum(audio.duration_seconds for audio in split_audio)
    print("Summed audio duration (seconds) after split:", summed_duration)
    print("Number of segments:", len(split_audio))
    print("Average segment duration (seconds):", summed_duration / len(split_audio))
    return split_audio


def get_audiosegment(audio_filepath, desired_framerate):
    audio_file = Path(audio_filepath)
    audio = pydub.AudioSegment.from_file(str(audio_filepath))
    
    print("Audio duration before framerate change:", audio.duration_seconds)
    audio = audio.set_frame_rate(desired_framerate)
    audio_seconds = audio.duration_seconds
    print("Audio duration after change:", audio_seconds)
    
    return audio, audio_seconds

def audiosegments_to_np(audiosegments):
    return [np.frombuffer(a._data, np.int16) for a in split_audio]

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare an audio file to be transcribed; outputs segments into directory")
    parser.add_argument("audio", help="Path to the audio file to preprocess")
    parser.add_argument("output_dir", help="Directory to put processed segments in")
    parser.add_argument("--framerate", type=int, default=16000)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    audio, audio_seconds = get_audiosegment(args.audio, args.framerate)
    split_audio = split_on_silence(audio)
    output_dir = Path(args.output_dir)
    for index, audio_segment in tqdm(enumerate(split_audio)):
        output_path = output_dir / f"{index}.wav"
        audio_segment.export(str(output_path), format="wav")

if __name__ == "__main__":
    main()
