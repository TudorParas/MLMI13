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

    def count_occurences(self, file_list):
        """
        Count the occurences of n_grams
        :param file_list: list of str representing the content of files
        :return: Dictionary of frequencies mapping str->int (word->word count)
        """
        occurence_dict = dict()
        for model in self.models:
            logger.info("Counting occurences for model with n={0}, cutoff={1}".format(model[0], model[1]))
            model_occurence_dict = dict()
            for file in file_list:
                words = du.get_words(file)

                n_grams = du.generate_n_grams(words, model[0])
                for n_gram in n_grams:
                    model_occurence_dict[n_gram] = model_occurence_dict.get(n_gram, 0) + 1

            model_occurence_dict = self.cutoff_infrequent_ngrams(model_occurence_dict, model[1])
            for key in model_occurence_dict.keys():
                occurence_dict[key] = occurence_dict.get(key, 0) + model_occurence_dict[key]

        return occurence_dict

    def cutoff_infrequent_ngrams(self, ngrams, cutoff):
        return {ngram: count for ngram, count in ngrams.items() if cutoff <= count}

    def compute_total_word_counts(self, occurences):
        self.total_word_counts = {category: sum(word_occurences.values()) for category, word_occurences in
                                  occurences.items()}

    def compute_log_frequencies(self, occurences):
        """
        Compute frequencies from occurence counts.
        """
        self.frequency_dict = dict()
        for category, word_occurences in occurences.items():
            category_log_freq_dict = dict()
            for key in word_occurences.keys():
                value = np.log(
                    (word_occurences[key] + self.smoothing) / (
                            self.total_word_counts[category] + self.vocab_size * self.smoothing))
                category_log_freq_dict[key] = value
            self.frequency_dict[category] = category_log_freq_dict

    def train_naive_bayes(self, training_data):
        """
        Compute the naive bayes on the training data
        :param training_data: dict of category->list of str
        """
        total_docs = sum([len(files) for files in training_data.values()])
        self.priors = {category: np.log(len(files) / total_docs) for category, files in training_data.items()}

        occurences = {category: self.count_occurences(files) for category, files in training_data.items()}

        self.vocab = set()
        for category, word_occurences in occurences.items():
            self.vocab.update(word_occurences.keys())
        self.vocab_size = len(self.vocab)
        logger.info("Computed vocab size {}".format(self.vocab_size))
        self.compute_total_word_counts(occurences)
        self.compute_log_frequencies(occurences)

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
                    scores[category] += self.frequency_dict[category].get(
                        ngram,
                        np.log(
                            self.smoothing) - np.log(
                            self.total_word_counts[category] + self.smoothing * self.vocab_size))
            else:
                # Skip if not in all categories
                in_all = True
                for word_freq in self.frequency_dict.values():
                    if ngram not in word_freq.keys():
                        in_all = False
                if in_all:
                    for category in scores.keys():
                        scores[category] += self.frequency_dict[category][ngram]

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
