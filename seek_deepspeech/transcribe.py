import argparse
import sys


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

    args = parser.parse_args()


def main():
    args = parse_args()
    
if __name__ == "__main__":
    main()
