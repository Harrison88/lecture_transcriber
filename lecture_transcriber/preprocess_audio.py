from pathlib import Path
import argparse

import pydub
import numpy as np
from tqdm import tqdm

import subprocess
import shlex


class RewindableChunker:
    def __init__(self, audiosegment, size=50):
        self.audiosegment = audiosegment
        self.size = size
        self.lower_bounds = range(0, len(audiosegment), size)
        self.upper_bounds = range(size, len(audiosegment) + size, size)
        self.current_index = 0

    def __iter__(self):
        while self.current_index < len(self.lower_bounds):
            lower_bound = self.lower_bounds[self.current_index]
            upper_bound = self.upper_bounds[self.current_index]
            yield self.audiosegment[lower_bound:upper_bound]
            self.current_index += 1

    def rewind(self, ms):
        iterations = (ms // self.size) + 1
        self.current_index = self.current_index - iterations

    def __len__(self):
        return len(self.lower_bounds)

    @property
    def current_time(self):
        return self.upper_bounds[self.current_index]


def preprocess_audio(audio_filepath, desired_framerate):
    audio, audio_seconds = get_audiosegment(audio_filepath, desired_framerate)
    threshold = determine_silence_threshold(audio)
    with tqdm(total=len(audio), unit="ms", miniters=1) as progress_bar:
        split_audio = split_on_silence(
            audio, silence_thresh=threshold, progress_bar=progress_bar
        )
    split_audio = audiosegments_to_np(split_audio)
    return split_audio, audio_seconds


def split_on_silence(
    audio, *, min_silence_len=1500, silence_thresh=-75, progress_bar=None
):
    if min_silence_len < 300:
        if progress_bar:
            progress_bar.update(len(audio))
        return [audio]
    nonsilent_ranges = pydub.silence.detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    range_lengths = [r[1] - r[0] for r in nonsilent_ranges]
    split_audio = []
    previous_stop = 0
    for start, stop in nonsilent_ranges:
        end = True
        skipped_silence = start - previous_stop
        previous_stop = stop

        new_segment = audio[start:stop]
        if new_segment.duration_seconds > 30:
            end = False
            new_len = min_silence_len - 200
            smaller_split = split_on_silence(
                new_segment,
                min_silence_len=new_len,
                silence_thresh=silence_thresh,
                progress_bar=progress_bar,
            )
            split_audio.extend(smaller_split)
        else:
            split_audio.append(new_segment)

        if progress_bar and end:
            progress_bar.update(skipped_silence + len(new_segment))
        elif progress_bar:
            progress_bar.update(skipped_silence)

    return split_audio


def determine_silence_threshold(audio):
    chunk_size = 50
    lower_bounds = range(0, len(audio), chunk_size)
    upper_bounds = range(chunk_size, len(audio) + chunk_size, chunk_size)

    samples = []
    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
        segment = audio[lower_bound:upper_bound]
        samples.append(segment.dBFS)

    silence = min(samples)
    average = audio.dBFS
    loudest = max(samples)

    print(f"Audio loudness stats: {silence = }; {average = }; {loudest = }")

    threshold = silence - ((silence - average) * 0.15)
    print(f"Threshold calculated to be {threshold} dBFS")

    return threshold


def convert_samplerate(audio, desired_sample_rate):
    sox_cmd = f"sox --type raw --channels {audio.channels} --rate {audio.frame_rate} --bits {audio.sample_width * 8} --encoding signed-integer - --type raw --bits 16 --channels 1 --encoding signed-integer --endian little --compression 0.0 --no-dither - rate {desired_sample_rate} "
    try:
        output = subprocess.run(
            shlex.split(sox_cmd), input=audio._data, capture_output=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError("SoX returned non-zero status: {}".format(e.stderr))
    except OSError as e:
        raise OSError(
            e.errno,
            "SoX not found, use {}hz files or install it: {}".format(
                desired_sample_rate, e.strerror
            ),
        )

    return output.stdout


def get_audiosegment(audio_filepath, desired_framerate):
    audio_file = Path(audio_filepath)
    audio = pydub.AudioSegment.from_file(str(audio_filepath))

    print("Audio duration before framerate change:", audio.duration_seconds)
    data = convert_samplerate(audio, desired_framerate)
    del audio
    audio = pydub.AudioSegment(
        data, sample_width=2, channels=1, frame_rate=desired_framerate
    )
    audio_seconds = audio.duration_seconds
    print("Audio duration after change:", audio_seconds)

    return audio, audio_seconds


def audiosegments_to_np(audiosegments):
    return [np.frombuffer(a._data, np.int16) for a in audiosegments]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an audio file to be transcribed; outputs segments into directory"
    )
    parser.add_argument("audio", help="Path to the audio file to preprocess")
    parser.add_argument("output_dir", help="Directory to put processed segments in")
    parser.add_argument("--framerate", type=int, default=16000)

    return parser.parse_args()


def main():
    args = parse_args()

    audio, audio_seconds = get_audiosegment(args.audio, args.framerate)
    threshold = determine_silence_threshold(audio)
    with tqdm(total=len(audio), unit="ms", miniters=1) as progress_bar:
        split_audio = split_on_silence(
            audio, silence_thresh=threshold, progress_bar=progress_bar
        )
    durations = [audio.duration_seconds for audio in split_audio]
    largest = max(durations)
    smallest = min(durations)
    average = sum(durations) / len(durations)
    print(f"{largest=}; {smallest=}; {average=}")
    print(
        "Duration before split:", audio_seconds, "Duration after split:", sum(durations)
    )
    output_dir = Path(args.output_dir)
    for index, audio_segment in tqdm(enumerate(split_audio)):
        output_path = output_dir / f"{index}.wav"
        audio_segment.export(str(output_path), format="wav")


if __name__ == "__main__":
    main()
