import deepspeech
import numpy as np


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


def create_deepspeech_model(
    model, *, beam_width=500, trie=None, lm=None, lm_alpha=0.75, lm_beta=1.85
):
    deepspeech_model = deepspeech.Model(model, beam_width)
    if trie and lm:
        deepspeech_model.enableDecoderWithLM(lm, trie, lm_alpha, lm_beta)

    return deepspeech_model


def audiosegment_to_np(audiosegment):
    return np.frombuffer(audiosegment._data, np.int16)
