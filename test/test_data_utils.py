import unittest
import data_utils as du

class MyTestCase(unittest.TestCase):
    def test_generate_n_grams(self):
        words = ['I', 'like', 'to', 'write', 'this']
        unigram_split = du.generate_n_grams(words, 1)

        self.assertEqual(unigram_split, [('I',), ('like',), ('to',), ('write',), ('this',)])

        n = 2
        bigram_split = du.generate_n_grams(words, 2)
        self.assertEqual(bigram_split, [('I', 'like'), ('like', 'to'), ('to', 'write'), ('write', 'this')])

    def test_get_words(self):
        file = "This isn't a random, test file!"
        words = du.get_words(file)

        self.assertEqual(words,['this', 'is', "n't", 'a', 'random', ',', 'test', 'file', '!'])



if __name__ == '__main__':
    unittest.main()
