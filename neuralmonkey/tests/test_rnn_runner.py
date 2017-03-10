#!/usr/bin/env python3.5

import unittest
import numpy as np

from neuralmonkey.runners.rnn_runner import (_n_best_indices,
                                             _score_one_seq_expansion,
                                             likelihood_beam_score)
from neuralmonkey.vocabulary import END_TOKEN_INDEX, PAD_TOKEN_INDEX


class TestRNNRunner(unittest.TestCase):
    def test_n_best_indices(self):
        input_small = np.arange(5)
        res_small = _n_best_indices(input_small, 10)
        self.assertTrue(np.all(res_small == [4, 3, 2, 1, 0]))
        self.assertEqual(res_small.shape, (5,))

        input_bigger = np.hstack((np.arange(100) / 1000, 1 + np.arange(3)))
        res_bigger = _n_best_indices(input_bigger, 3)
        self.assertTrue(np.all(res_bigger == [102, 101, 100]))
        self.assertEqual(res_bigger.shape, (3,))

        empty = _n_best_indices(np.array([]), 10)
        self.assertEqual(empty.shape, (0,))

    def test_score_one_seq_expansion(self):
        r"""Test beam expansion scoring.
                                         / 0 (-2) => score -2
                  / 0 (-2) => score -2 <
                 |                       \ 1 (-4) => score -2.7
        0 (-2) -<
                 |                       / 0 (-5) => score -3.7
                  \ 1 (-4) => score -3 <
                                         \ 1 (-7) => score -4.3

                 / 0 (-5) => score -4 - not in 2nd beam
        1 (-3) -<
                 \ 1 (-7) => score -5 - not in 2nd beam

        2 (-4) - not in the first beam
        3 (-5) - not in the first beam
        4 (-6) - not in the first beam
        """

        beam_size = 2
        scoring_function = likelihood_beam_score

        first_hyp_logprobs = np.array([[-2, -3, -4, -5, -6]])
        hypotheses, hyp_logprobs = _score_one_seq_expansion(
            first_hyp_logprobs, [None], [None], beam_size, scoring_function)

        self.assertTrue(np.all(hypotheses == np.array([[0], [1]])))
        self.assertTrue(np.all(hyp_logprobs == np.array([[-2], [-3]])))

        next_distributions = np.array([[-2, -4],
                                       [-5, -7]])

        new_hyps, new_logprobs = _score_one_seq_expansion(
            next_distributions, hypotheses, hyp_logprobs,
            beam_size, scoring_function)

        self.assertTrue(np.all(np.array([[0, 0], [0, 1]] == new_hyps)))
        self.assertTrue(np.all(np.array([[-2, -2], [-2, -4]] == new_logprobs)))

        last_hyps, last_logprobs = _score_one_seq_expansion(
            next_distributions, new_hyps, new_logprobs,
            beam_size, scoring_function)

        self.assertTrue(np.all(np.array([[0, 0, 0], [0, 0, 1]] == last_hyps)))
        self.assertTrue(
            np.all(np.array([[-2, -2, -2], [-2, -2, -4]] == last_logprobs)))

    def test_not_expand_finished(self):
        """Hypotheses containg END_TOKEN should not be expanded further."""
        beam_size = 2
        scoring_function = likelihood_beam_score

        next_distributions = np.random.uniform(-100, -1, size=(1, 10))

        ending_hypothesis = np.array([[10, END_TOKEN_INDEX]])
        ending_hyp_logprobs = np.random.uniform(-100, -1, size=(1, 2))

        expanded_hyp, _ = _score_one_seq_expansion(
            next_distributions, ending_hypothesis, ending_hyp_logprobs,
            beam_size, scoring_function)
        self.assertTrue(np.all(
            expanded_hyp == [10, END_TOKEN_INDEX, PAD_TOKEN_INDEX]))

        finished_hypothesis = np.array(
            [[11, END_TOKEN_INDEX, PAD_TOKEN_INDEX]])
        finished_hyp_logprobs = np.random.uniform(-100, -1, size=(1, 3))
        exp_finished_hyp, _ = _score_one_seq_expansion(
            next_distributions, finished_hypothesis, finished_hyp_logprobs,
            beam_size, scoring_function)
        self.assertTrue(np.all(exp_finished_hyp == [
            11, END_TOKEN_INDEX, PAD_TOKEN_INDEX, PAD_TOKEN_INDEX]))


if __name__ == "__main__":
    unittest.main()
