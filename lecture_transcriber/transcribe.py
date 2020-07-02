import argparse
import sys
import logging

from deepsegment import DeepSegment as sbd
import deepspeech
from tqdm import tqdm

from . import preprocess_audio
from . import utils


logger = logging.getLogger(__name__)
segmenter = sbd("en")


def transcribe(audiosegment, model):
    stream = model.createStream()
    chunks = utils.RewindableChunker(audiosegment)
    sentences = []
    len_output = 0
    reached_seconds = 0
    progress_bar = tqdm(
        total=int(audiosegment.duration_seconds), unit="a_sec", desc="Transcribing"
    )
    for chunk in chunks:
        data = utils.audiosegment_to_np(chunk)
        model.feedAudioContent(stream, data)
        output = model.intermediateDecode(stream).split()
        if len(output) > len_output and len(output) > 1:
            segments = segmenter.segment(" ".join(output[:-1]))
            if len(segments) > 1:
                logger.debug(f"Adding sentence starting with {' '.join(output[:3])}...")
                sentences.append(segments[0])
                metadata = model.finishStreamWithMetadata(stream)
                words = utils.words_from_metadata(metadata)
                next = len(" ".join(segments[1:]).split()) + 2
                word = words[-next]
                end_ms = metadata.items[-1].start_time * 1000
                ms = (
                    abs(((word["start_time"] + word["duration"]) * 1000) - end_ms) + 200
                )
                current_seconds = chunks.current_time / 1000
                if current_seconds > reached_seconds:
                    progress_bar.update(current_seconds - reached_seconds)
                    progress_bar.refresh()
                    reached_seconds = current_seconds
                chunks.rewind(int(ms))
                stream = model.createStream()
                len_output = 0
            else:
                len_output = len(output)

    progress_bar.close()
    return sentences
