import random
import numpy as np
import regex as re

try:
    #pylint: disable=unused-import,bare-except
    from typing import List, Tuple
except:
    pass

class Vocabulary(object):
    def __init__(self, tokenized_text=None): # type: List[str] -> Vocabulary
        self.word_to_index = {}
        self.index_to_word = []
        self.word_count = {}

        self.add_tokenized_text(["<pad>", "<s>", "</s>", "<unk>"])

        if tokenized_text:
            self.add_tokenized_text(tokenized_text)

    def add_word(self, word): # type: str -> None
        if word not in self.word_to_index:
            self.word_to_index[word] = len(self.index_to_word)
            self.index_to_word.append(word)
            self.word_count[word] = 0
        self.word_count[word] += 1

    def add_tokenized_text(self, tokenized_text): # type: List[str] -> None
        for word in tokenized_text:
            self.add_word(word)

    def get_train_word_index(self, word): # type: str -> int
        if word not in self.word_count or self.word_count[word] <= 1:
            if random.random > 0.5:
                return self.word_to_index["<unk>"]
            else:
                return self.word_to_index[word]
        else:
            return self.word_to_index[word]

    def get_word_index(self, word): # type: str -> int
        if word not in self.word_count:
            return self.word_to_index["<unk>"]
        else:
            return self.word_to_index[word]

    def __len__(self): # type: int
        return len(self.index_to_word)

    def __contains__(self, word): # type: bool
        return word in self.word_to_index

    def trunkate(self, size): # type: int -> None
        """
        Trunkates the Vocabulary to requested size by keep only the most
        frequent tokens.
        """

        # sort by frequency
        words_by_freq = \
            sorted(self.word_count.keys(), key=lambda w: self.word_count[w])

        # keep the least frequent words which are not special symbols
        words_to_delete = \
            [w for w in words_by_freq[:-size] if not re.match(ur"^<.*>$", w)]
        # sort by index ... bigger indices needs to be removed first
        # to keep the lists propertly shaped
        delete_words_by_index = \
            sorted([(w, self.word_to_index[w]) for w in words_to_delete], key=lambda p: -p[1])

        for word, index in delete_words_by_index:
            del self.word_count[word]
            del self.index_to_word[index]

        self.word_to_index = {}
        for index, word in enumerate(self.index_to_word):
            self.word_to_index[word] = index

    def sentences_to_tensor(self, sentences, max_len, train=False):
        # type: (List[List[str]], int, bool) -> Tuple[np.Array, np.Array]
        """
        Generates the tensor representation for the provided sentences.

        Args:

            sentences: List of sentences as lists of tokens.
            max_len: Maximum lengh of a sentence toward which they will be
              padded to.
            train: Flag whether this is for training purposes.

        """

        word_indices = [np.zeros([len(sentences)], dtype=np.int) for _ in range(max_len + 2)]
        weights = [np.zeros([len(sentences)]) for _ in range(max_len + 1)]

        word_indices[0] = np.repeat(self.get_word_index("<s>"), len(sentences))

        for i in range(max_len + 1):
            for j, sent in enumerate(sentences):
                if i < len(sent):
                    word_indices[i + 1][j] = self.get_train_word_index(sent[i]) if train \
                                         else self.get_word_index(sent[i])
                    weights[i][j] = 1.0
                elif i == len(sent):
                    word_indices[i + 1][j] = self.get_word_index("</s>")
                    weights[i][j] = 1.0

        return word_indices, weights

    def vectors_to_sentences(self, vectors):
        #pylint: disable=fixme
        # TODO type
        sentences = [[] for _ in range(vectors[0].shape[0])]

        for vec in vectors:
            for sentence, word_i in zip(sentences, vec):
                if not sentence or (sentence and sentence[-1] != "</s>"):
                    sentence.append(self.index_to_word[word_i])

        return [s[:-1] for s in sentences]



