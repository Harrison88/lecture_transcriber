import argparse
import sys
import json
import time
import logging

logging.disable()

import deepspeech
from tqdm import tqdm

from . import preprocess_audio

from deepsegment import DeepSegment as sbd

segmenter = sbd("en")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using DeepSpeech"
    )

    parser.add_argument("audio", help="Path to the audio file to transcribe")

    parser.add_argument(
        "--model",
        "-m",
        default="models/output_graph.pbmm",
        help="Path to the model (protocol buffer binary file)",
    )

    parser.add_argument(
        "--lm",
        "-l",
        const="models/lm.binary",
        nargs="?",
        help="Path to the language model binary file",
    )

    parser.add_argument(
        "--trie",
        "-t",
        const="models/trie",
        nargs="?",
        help="Path to the language model trie file created with native_client/generate_trie",
    )

    parser.add_argument(
        "--beam_width", type=int, default=500, help="Beam width for the CTC decoder"
    )

    parser.add_argument(
        "--lm_alpha", type=float, default=0.75, help="Language model weight (lm_alpha)"
    )

    parser.add_argument(
        "--lm_beta", type=float, default=1.85, help="Word insertion bonus (lm_beta)"
    )

    parser.add_argument(
        "--output", "-o", nargs="?", default=sys.stdout, type=argparse.FileType("w")
    )

    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Transcribe the audio file without splitting into segments",
    )

    args = parser.parse_args()

    return args


def create_deepspeech_model(args):
    deepspeech_model = deepspeech.Model(args.model, args.beam_width)
    if args.trie and args.lm:
        deepspeech_model.enableDecoderWithLM(
            args.lm, args.trie, args.lm_alpha, args.lm_beta
        )

    return deepspeech_model


def words_from_metadata(metadata):
    word = ""
    word_list = []
    word_start_time = 0

    for index, item in enumerate(metadata.items):
        if item.character != " ":
            word = word + item.character

        if item.character == " " or index == metadata.num_items - 1:
            word_duration = item.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = {
                "word": word,
                "start_time": word_start_time,
                "duration": word_duration,
            }
            word_list.append(each_word)

            word = ""
            word_start_time = 0
        elif len(word) == 1:
            word_start_time = item.start_time

    return word_list


def main():
    args = parse_args()

    print("Loading DeepSpeech Model")
    deepspeech_model = create_deepspeech_model(args)

    stream = deepspeech_model.createStream()
    audiosegment, audio_seconds = preprocess_audio.get_audiosegment(
        args.audio, deepspeech_model.sampleRate()
    )
    chunks = preprocess_audio.RewindableChunker(audiosegment)
    sentences = []
    len_output = 0
    for chunk in tqdm(chunks):
        data = preprocess_audio.audiosegments_to_np([chunk])
        deepspeech_model.feedAudioContent(stream, data[0])
        output = deepspeech_model.intermediateDecode(stream).split()
        if len(output) > len_output and len(output) > 1:
            segments = segmenter.segment(" ".join(output[:-1]))
            if len(segments) > 1:
                sentences.append(segments[0])
                metadata = deepspeech_model.finishStreamWithMetadata(stream)
                words = words_from_metadata(metadata)
                next = len(" ".join(segments[1:]).split()) + 2
                word = words[-next]
                end_ms = metadata.items[-1].start_time * 1000
                ms = (
                    abs(((word["start_time"] + word["duration"]) * 1000) - end_ms) + 200
                )

                chunks.rewind(int(ms))
                stream = deepspeech_model.createStream()
                len_output = 0
            else:
                len_output = len(output)

    print("Outputting transcription")
    json.dump(sentences, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
