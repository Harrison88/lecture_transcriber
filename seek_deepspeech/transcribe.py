import argparse
import sys
import json

import deepspeech
from tqdm import tqdm

import preprocess_audio


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
    
    parser.add_argument("--no-split", action="store_true", help="Transcribe the audio file without splitting into segments")

    args = parser.parse_args()
    
    return args


def create_deepspeech_model(args):
    deepspeech_model = deepspeech.Model(args.model, args.beam_width)
    if args.trie and args.lm:
        deepspeech_model.enableDecoderWithLM(args.lm, args.trie, args.lm_alpha, args.lm_beta)
        
    return deepspeech_model


def main():
    args = parse_args()
    
    print("Loading DeepSpeech Model")
    deepspeech_model = create_deepspeech_model(args)
    
    if not args.no_split:
        print("Preparing audio for transcription")
        audio_files, audio_seconds = preprocess_audio.preprocess_audio(args.audio, deepspeech_model.sampleRate())
    else:
        audio, audio_seconds = preprocess_audio.get_audiosegment(args.audio, deepspeech_model.sampleRate())
        audio_files = preprocess_audio.audiosegments_to_np([audio])
    
    print("Transcribing audio")
    transcriptions = []
    for audio in tqdm(audio_files):
        transcription = deepspeech_model.stt(audio)
        transcriptions.append(transcription)
    
    print("Outputting transcription")
    json.dump(transcriptions, args.output)
    
    print("Done!")
    
if __name__ == "__main__":
    main()
