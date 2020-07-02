import logging
from pathlib import Path
import shlex
import subprocess

import pydub

logger = logging.getLogger(__name__)


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


def get_audiosegment(audio_file, desired_framerate, format="mp3"):
    audio = pydub.AudioSegment.from_file(audio_file, format=format)

    logger.debug(f"Audio duration before framerate change: {audio.duration_seconds}")
    data = convert_samplerate(audio, desired_framerate)
    del audio
    audio = pydub.AudioSegment(
        data, sample_width=2, channels=1, frame_rate=desired_framerate
    )
    audio_seconds = audio.duration_seconds
    logger.debug(f"Audio duration after change: {audio_seconds}")

    return audio, audio_seconds
