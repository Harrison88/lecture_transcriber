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

def split_on_silence(audio, *, silence_thresh=-75):
    split_audio = pydub.silence.split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=700)
    summed_duration = sum(audio.duration_seconds for audio in split_audio)
    print("Summed audio duration (seconds) after split:", summed_duration)
    print("Number of segments:", len(split_audio))
    print("Average segment duration (seconds):", summed_duration / len(split_audio))
    return split_audio


def determine_silence_threshold(audio):
    chunk_size = 50
    lower_bounds = range(0, len(audio), chunk_size)
    upper_bounds = range(chunk_size, len(audio)+chunk_size, chunk_size)
    
    samples = []
    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
        segment = audio[lower_bound:upper_bound]
        samples.append(segment.dBFS)
    
    silence = min(samples)
    average = audio.dBFS
    loudest = max(samples)
    
    print(f"Audio loudness stats: {silence = }; {average = }; {loudest = }")
    
    threshold = silence - ((silence - average) * .15)
    print(f"Threshold calculated to be {threshold} dBFS")
    
    return threshold


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
    threshold = determine_silence_threshold(audio)
    split_audio = split_on_silence(audio, silence_thresh=threshold)
    output_dir = Path(args.output_dir)
    for index, audio_segment in tqdm(enumerate(split_audio)):
        output_path = output_dir / f"{index}.wav"
        audio_segment.export(str(output_path), format="wav")

if __name__ == "__main__":
    main()
