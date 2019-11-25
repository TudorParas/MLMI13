import unittest

import naive_bayes


class MyTestCase(unittest.TestCase):
    def test_count_occurences(self):
        file_list = ["This;is, a. test file", "This;     another $test file"]
        nb1 = naive_bayes.NaiveBayesClassifier(n=1)
        uni_occ_dict = nb1.count_occurences(file_list)
        self.assertEqual(uni_occ_dict, {('a',): 1,
                                        ('another',): 1,
                                        ('file',): 2,
                                        ('is',): 1,
                                        ('test',): 2,
                                        ('this',): 2})

        nb2 = naive_bayes.NaiveBayesClassifier(n=2)
        bi_occ_dict = nb2.count_occurences(file_list)
        self.assertEqual(bi_occ_dict, {('a', 'test'): 1,
                                       ('another', 'test'): 1,
                                       ('is', 'a'): 1,
                                       ('test', 'file'): 2,
                                       ('this', 'another'): 1,
                                       ('this', 'is'): 1})

    def test_cutoff(self):
        file_list = ["This;is, a. test file", "This;     another $test file"]
        nb1 = naive_bayes.NaiveBayesClassifier(n=1, cutoff=2)
        uni_occ_dict = nb1.count_occurences(file_list)
        uni_occ_dict = nb1.cutoff_infrequent_ngrams(uni_occ_dict)
        self.assertEqual(uni_occ_dict, {('file',): 2,
                                        ('test',): 2,
                                        ('this',): 2})


if __name__ == '__main__':
    unittest.main()
