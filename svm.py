"""Code containing SVM stuff"""

import logging

from gensim.utils import simple_preprocess

from data_utils import load_untagged_data
from doc2vec import get_doc2vec_model

logger = logging.getLogger(__name__)


class SVM():
    def __init__(self, dm=1, vector_size=100, epochs=20, context_window=3, min_count=2, hierarchical_softmax=1,
                 dm_concat=1):
        self.doc2vec = get_doc2vec_model(dm, vector_size, epochs, context_window, min_count, hierarchical_softmax,
                                         dm_concat)


    def preprocess_files(self, files):
        inferred_docs = list()
        for file in files:
            inferred_docs.append(self._process_file(file))
        return inferred_docs

    def _process_file(self, file):
        tokens = simple_preprocess(file)
        return self.doc2vec.infer_vector(tokens)


svm = SVM()
