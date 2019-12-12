"""Code for part 1 using the Naive Bayes baseline"""
import logging

import numpy as np

import data_utils as du

logger = logging.getLogger(__name__)


class NaiveBayesClassifier(object):
    def __init__(self, models, smoothing=0):
        """
        Initialise the NB classifer
        :param models: list of tuples of (ngram size, cutoff)
        :param smoothing: The smoothing param for one word
        """
        self.models = models
        self.smoothing = smoothing
        logger.info("Created Naive Bayes classifer with: models={0}, smoothing={1}".format(self.models, self.smoothing))

    def count_word_occ(self, file_list):
        occurence_dict = dict()
        for model in self.models:
            tmp_occ_dict = dict()
            for file in file_list:
                words = du.get_words(file)
                ngrams = du.generate_n_grams(words, model[0])
                for ngram in ngrams:
                    tmp_occ_dict[ngram] = tmp_occ_dict.get(ngram, 0) + 1

            # Cutoff
            cutoff = model[1]
            for ngram, count in tmp_occ_dict.items():
                if count >= cutoff:
                    occurence_dict[ngram] = occurence_dict.get(ngram, 0) + count
        return occurence_dict

    def create_log_freqs(self, occurences):
        self.word_freq = dict()
        for cat, cat_occs in occurences.items():
            cat_freq = dict()
            for word, count in cat_occs.items():
                cat_freq[word] = np.log(count + self.smoothing) - np.log(
                    self.total_word_counts[cat] + self.smoothing * self.vocab_size)
            self.word_freq[cat] = cat_freq

    def train(self, training_data):
        """
        Compute the naive bayes on the training data
        :param training_data: dict of category->list of str
        """
        total_docs = sum([len(files) for files in training_data.values()])
        self.priors = {category: np.log(len(files) / total_docs) for category, files in training_data.items()}

        occurences = {cat: self.count_word_occ(file_list) for cat, file_list in training_data.items()}
        self.total_word_counts = {cat: sum([count for count in occs.values()]) for cat, occs in occurences.items()}
        self.vocab = set()
        for cat_occ in occurences.values():
            self.vocab.update(cat_occ.keys())
        self.vocab_size = len(self.vocab)\

        self.create_log_freqs(occurences)

    def predict_file(self, file):
        words = du.get_words(file)
        ngrams = list()
        for model in self.models:
            model_ngrams = du.generate_n_grams(words, model[0])
            ngrams += model_ngrams
        # Initialize score to priors
        scores = self.priors.copy()
        for ngram in ngrams:
            if self.smoothing > 0:
                for category in scores.keys():
                    scores[category] += self.word_freq[category].get(
                        ngram,
                        np.log(
                            self.smoothing) - np.log(
                            self.total_word_counts[category] + self.smoothing * self.vocab_size))
            else:
                # Skip if not in all categories
                in_all = True
                for cat_freq in self.word_freq.values():
                    if ngram not in cat_freq.keys():
                        in_all = False
                if in_all:
                    for category in scores.keys():
                        scores[category] += self.word_freq[category][ngram]

        return 1 if scores['pos'] > scores['neg'] else 0

    def predict(self, test_data):
        """
        Test the current version of NB
        :param test_data: list of files
        """
        predictions = list()
        for file in test_data:
            predictions.append(self.predict_file(file))

        return np.array(predictions)
