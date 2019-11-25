"""
Script with helpful methods for loading the data
"""
import logging
import os
from os.path import join

import nltk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

current_dir_ = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = join(current_dir_, 'dataset/data')
DATA_TAGGED_DIR = join(current_dir_, 'dataset/data-tagged')
DATA_MODELS_DIR = join(current_dir_, 'dataset/models')
DATA_RESOURCES_DIR = join(current_dir_, 'dataset/resources')
DATA_DOC2VEC_DIR = join(current_dir_, 'dataset/doc2vec')
MODELS_DOC2VEC_DIR = join(current_dir_, 'models/doc2vec')


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        # Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]


def get_files_from_dir(dir_path):
    return os.listdir(dir_path)


def find_bad_files(dir_path):
    print("Nr of files {}".format(len(get_files_from_dir(dir_path))))
    for file in get_files_from_dir(dir_path):
        if '(' in file:
            print(file)


def load_files(data_dir, subdir):
    pos_dir = join(data_dir, subdir)
    file_list = list()
    for file in get_files_from_dir(pos_dir):
        with open(join(pos_dir, file), 'r', encoding="utf8") as f:
            file_list.append(f.read())
    logger.info("Successfully read {0} {1} files".format(len(file_list), subdir))
    return file_list


def load_data(subdir):
    """
    Load the files from the data directory.
    :param subdir: 'POS' or 'NEG'
    :return: list of str of files
    """
    return load_files(DATA_TAGGED_DIR, subdir)

def load_untagged_data(subdir):
    return load_files(DATA_DIR, subdir)

def load_doc2vec_files():
    files = []
    subdirs = ["train/neg", "train/pos", "train/unsup", "test/neg", "test/pos"]
    for subdir in subdirs:
        files += load_files(DATA_DOC2VEC_DIR, subdir)
    return files


@Memoize
def get_words(file):
    """Get a list of lowercase words from file"""
    file = file.lower()
    return [line.split('\t')[0] for line in file.split('\n')]


def generate_n_grams(words, n):
    """
    Generate a list of n-grams
    :param words: list of str
    :return: list of tuple of length n
    """
    n_grams = [tuple(words[i: i + n]) for i in range(len(words) - n + 1)]
    return n_grams
