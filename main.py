"""Script used for testing the implementations"""
import logging

import data_utils as du
import evaluation as ev
from naive_bayes import NaiveBayesClassifier
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NEG_FILE_LIST = du.load_data('NEG')
POS_FILE_LIST = du.load_data('POS')


def get_training_test_data():
    neg_train_data = NEG_FILE_LIST[:900]
    neg_test_data = NEG_FILE_LIST[900:]

    pos_train_data = POS_FILE_LIST[:900]
    pos_test_data = POS_FILE_LIST[900:]

    training_data = {"pos": pos_train_data, "neg": neg_train_data}
    test_data = {"pos": pos_test_data, "neg": neg_test_data}

    return training_data, test_data


def eval_nb(training_data, test_data, models, smoothing=0):
    nb = NaiveBayesClassifier(models=models, smoothing=smoothing)
    nb.train_naive_bayes(training_data)
    # print(nb.frequency_dict)
    scores = nb.test_naive_bayes(test_data)
    neg_file_scores, pos_file_scores = scores["neg"], scores["pos"]

    correct_neg = [1 if score["neg"] > score["pos"] else 0 for score in neg_file_scores]
    correct_pos = [1 if score["neg"] < score["pos"] else 0 for score in pos_file_scores]

    return correct_neg, correct_pos


def print_nb_accuracy(models, smoothing=0):
    training_data, test_data = get_training_test_data()
    correct_neg, correct_pos = eval_nb(training_data, test_data, models, smoothing)
    accuracy = ev.get_nb_accuracy(correct_neg, correct_pos)
    print("NB for models={0} smoothing={1} got accuracy of {2}".format(models, smoothing, accuracy))
    return accuracy


def compare_models(compared_models):
    training_data, test_data = get_training_test_data()
    for modelA, modelB in compared_models:
        correct_negA, correct_posA = eval_nb(training_data, test_data, models=modelA[0], smoothing=modelA[1])
        correct_negB, correct_posB = eval_nb(training_data, test_data, models=modelB[0], smoothing=modelB[1])
        accuracy_A = ev.get_nb_accuracy(correct_negA, correct_posA)
        accuracy_B = ev.get_nb_accuracy(correct_negB, correct_posB)
        p_value = ev.compare_models(correct_negA, correct_posA, correct_negB, correct_posB)
        print("Got p_value {6} for comparing models:\n"
              "\t{0} smoothing {1}: accuracy {2}\n"
              "\t{3} smoothing {4}: accuracy {5}".format(
            modelA[0], modelA[1], accuracy_A, modelB[0], modelB[1], accuracy_B, p_value))


def get_rr_folds():
    neg_folds = [NEG_FILE_LIST[k::10] for k in range(10)]
    pos_folds = [POS_FILE_LIST[k::10] for k in range(10)]
    return neg_folds, pos_folds


def process_fold_k(neg_folds, pos_folds, k):
    neg_train_data = [file for fold in (neg_folds[:k] + neg_folds[k + 1:]) for file in fold]
    neg_test_data = neg_folds[k]

    pos_train_data = [file for fold in (pos_folds[:k] + pos_folds[k + 1:]) for file in fold]
    pos_test_data = pos_folds[k]

    training_data = {"pos": pos_train_data, "neg": neg_train_data}
    test_data = {"pos": pos_test_data, "neg": neg_test_data}

    return training_data, test_data


def perform_crossvalidation_accuracy(models,  neg_folds, pos_folds, smoothing=0):
    accuracies = []
    correct = []
    for k in range(len(neg_folds)):
        training_data, test_data = process_fold_k(neg_folds=neg_folds, pos_folds=pos_folds, k=k)
        correct_neg, correct_pos = eval_nb(training_data, test_data, models, smoothing)
        accuracy = ev.get_nb_accuracy(correct_neg, correct_pos)
        accuracies.append(accuracy)
        correct.append((sum(correct_neg), sum(correct_pos)))

    return accuracies, correct


if __name__ == '__main__':
    neg_folds, pos_folds = get_rr_folds()
    # print_nb_accuracy(models=[(1, 4)], smoothing=0)
    # print_nb_accuracy(models=[(2, 7)], smoothing=0)
    # # Mixed model
    # print_nb_accuracy(models=[(1, 4), (2, 7)], smoothing=0)

    # print_nb_accuracy(models=[(1, 4)], smoothing=0.005)
    # print_nb_accuracy(models=[(2, 7)], smoothing=0.005)
    # # # Mixed model
    # print_nb_accuracy(models=[(1, 4), (2, 7)], smoothing=1)

    # compared_models = [
    #     (([(1, 4)], 0), ([(1, 4)], 10)),
    #     (([(2, 7)], 0), ([(2, 7)], 10)),
    #     (([(1, 4), (2, 7)], 0), ([(1, 4), (2, 7)], 10))
    # ]
    # compare_models(compared_models)
    # #
    # #
    # accuracies, correct = perform_crossvalidation_accuracy(models=[(1, 4)], smoothing=0, neg_folds=neg_folds,
    #                                                        pos_folds=pos_folds)
    # print(accuracies)
    # print(correct)

    # smoothing_acces = []
    # log_smoothings = np.linspace(-5, 5, 20)
    # smoothings = np.power(10, log_smoothings)
    # for smoothing in smoothings:
    #     smoothing_acces.append(print_nb_accuracy(models=[(1, 4)], smoothing=smoothing))
    # print(smoothing_acces)
    # plt.plot(log_smoothings, smoothing_acces)
    # plt.show()