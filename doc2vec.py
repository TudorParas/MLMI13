"""Used for doing doc2vec training"""

import logging
from os.path import join, exists

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

from data_utils import load_doc2vec_files, MODELS_DOC2VEC_DIR

logger = logging.getLogger(__name__)


def get_preprocessed_files():
    file_list = load_doc2vec_files()
    tokens_list = [simple_preprocess(file) for file in file_list]
    documents = [TaggedDocument(tokens, [index]) for index, tokens in enumerate(tokens_list)]

    return documents


def get_doc2vec_model(dm=1, vector_size=100, epochs=20, context_window=3, min_count=2, hierarchical_softmax=1,
                      dm_concat=1):
    """

    Parameters
    ----------
    dm : {1,0}, optional
        Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
        Otherwise, `distributed bag of words` (PV-DBOW) is employed.
    vector_size : int, optional
        Dimensionality of the feature vectors.
    epochs : int, optional
        Number of iterations (epochs) over the corpus.
    context_window : int, optional
        The maximum distance between the current and predicted word within a sentence.
    min_count : int, optional
        Ignores all words with total frequency lower than this.
    hierarchical_softmax : {1,0}, optional
        If 1, hierarchical softmax will be used for model training.
        If set to 0, and `negative` is non-zero, negative sampling will be used.
    dm_concat : {1,0}, optional
        If 1, use concatenation of context vectors rather than sum/average;
        Note concatenation results in a much-larger model, as the input
        is no longer the size of one (sampled or arithmetically combined) word vector, but the
        size of the tag(s) and all words in the context strung together.

        The input parameters are of the following types:
            * `word` (str) - the word we are examining
            * `count` (int) - the word's frequency count in the corpus
            * `min_count` (int) - the minimum count threshold.
    """
    model = load_doc2vec_model(dm, vector_size, epochs, context_window, min_count, hierarchical_softmax, dm_concat)

    if model is None:
        logger.info("Model not found, training a new model")
        model = Doc2Vec(dm=dm, vector_size=vector_size, epochs=epochs, window=context_window,
                        hs=hierarchical_softmax, min_count=min_count, dm_concat=dm_concat)

        train_corpus = get_preprocessed_files()
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        save_doc2vec_model(model)

    # vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
    # print(vector)
    return model


def save_doc2vec_model(model):
    model_name = "model_{dm}_{vector_size}_{epochs}_{context_window}_{min_count}_{hierarchical_softmax}_{dm_concat}".format(
        dm=model.dm, vector_size=model.vector_size, epochs=model.epochs, context_window=model.window,
        min_count=model.min_count, hierarchical_softmax=model.hs, dm_concat=model.dm_concat
    )
    model.save(join(MODELS_DOC2VEC_DIR, model_name))


def load_doc2vec_model(dm, vector_size, epochs, context_window, min_count, hierarchical_softmax, dm_concat):
    model_name = "model_{dm}_{vector_size}_{epochs}_{context_window}_{min_count}_{hierarchical_softmax}_{dm_concat}".format(
        dm=bool(dm), vector_size=vector_size, epochs=epochs, context_window=context_window,
        min_count=min_count, hierarchical_softmax=hierarchical_softmax, dm_concat=dm_concat
    )
    logger.info("Attempting to load {}".format(model_name))
    model_path = join(MODELS_DOC2VEC_DIR, model_name)
    if exists(model_path):
        return Doc2Vec.load(model_path)
    else:
        return None


def pretrain_models():
    epochs = 10
    vector_size = 100
    dm_concat = 0
    for dm in [0, 1]:
        for context_window in [2, 4]:
            for hierarchical_softmax in [0, 1]:
                get_doc2vec_model(dm=dm, vector_size=vector_size, context_window=context_window,
                                  hierarchical_softmax=hierarchical_softmax, dm_concat=dm_concat, epochs=epochs)
