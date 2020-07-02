import argparse
import json
import logging
import sys

from . import preprocess_audio
from . import transcribe
from . import utils

logging.getLogger().setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    deepspeech_model = utils.create_deepspeech_model(
        args.model, trie=args.trie, lm=args.lm
    )

    audiosegment, audio_seconds = preprocess_audio.get_audiosegment(
        args.audio, deepspeech_model.sampleRate()
    )

    sentences = transcribe.transcribe(audiosegment, deepspeech_model)

    logger.info(f"Outputting transcription as {'JSON' if args.json else 'text'}")
    if args.json:
        json.dump(sentences, args.output)
    else:
        args.output.write(" ".join(sentences))

    args.output.close()

    logger.info("Transcription completed successfully")


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
        "--json", "-j", action="store_true", help="Output sentences as a JSON array"
    )

    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Transcribe the audio file without splitting into segments",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
