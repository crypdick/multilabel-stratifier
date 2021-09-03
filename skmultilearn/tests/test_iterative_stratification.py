import unittest

import numpy as np

from skmultilearn import IterativeStratification


class IterativeStratificationTest(unittest.TestCase):
    def test_if_variables_are_initialized_correctly(self):
        stratifier = IterativeStratification(n_splits=2, order=1)
        y = np.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])

        stratifier._init_vars(y)

        stratifier._prepare_stratification()

        self.assertEqual(stratifier.n_samples, 4)
        self.assertEqual(stratifier.n_labels, 2)
        self.assertEqual(len(stratifier.rows), 4)
        self.assertEqual(len(stratifier.rows_used), 4)
        self.assertEqual(len(stratifier.percentage_per_split), 2)
        self.assertEqual(len(stratifier.desired_samples_per_split), 2)
        self.assertEqual(len(stratifier.splits), 2)
        self.assertTrue(not any(stratifier.rows_used.values()))
        self.assertFalse(any(stratifier.rows_used.values()))
        self.assertEqual(stratifier.order, 1)

        for d in stratifier.percentage_per_split:
            self.assertEqual(d, 1 / 2.0)

        for d in stratifier.desired_samples_per_split:
            self.assertEqual(d, y.shape[0] / 2.0)

        self.assertEqual(len(stratifier.all_combinations), 2)
        self.assertEqual(len(stratifier.per_row_combinations[0]), 0)
        self.assertEqual(len(stratifier.per_row_combinations[1]), 1)
        self.assertEqual(len(stratifier.per_row_combinations[2]), 1)
        self.assertEqual(len(stratifier.per_row_combinations[3]), 2)

        self.assertEqual(len(stratifier.samples_with_combination), 2)
        self.assertEqual(len(stratifier.desired_samples_per_combination_per_split), 2)
        for combination, samples in stratifier.samples_with_combination.items():
            self.assertEqual(len(set(combination)), 1)
            self.assertEqual(len(samples), 2)

        for combination, desirability in stratifier.desired_samples_per_combination_per_split.items():
            self.assertEqual(len(set(combination)), 1)
            self.assertEqual(len(desirability), 2)
            for desire in desirability:
                self.assertEqual(desire, 1.0)

    def test_if_positive_evidence_does_not_include_negative_evidence(self):
        stratifier = IterativeStratification(n_splits=2, order=1)
        y = np.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])

        stratifier._init_vars(y)
        stratifier._prepare_stratification()
        stratifier._distribute_positive_evidence()

        self.assertFalse(stratifier.rows_used[0])
        self.assertTrue(stratifier.rows_used[1])
        self.assertTrue(stratifier.rows_used[2])
        self.assertTrue(stratifier.rows_used[3])

        for combination, samples in stratifier.desired_samples_per_combination_per_split.items():
            for desire in samples:
                self.assertEqual(desire, 0)

    def test_if_negative_evidence_is_distributed(self):
        stratifier = IterativeStratification(n_splits=2, order=1)
        y = np.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])

        stratifier._init_vars(y)
        stratifier._prepare_stratification()

        stratifier._distribute_positive_evidence()
        self.assertFalse(stratifier.rows_used[0])

        stratifier._distribute_negative_evidence()
        self.assertTrue(stratifier.rows_used[0])

    def test_if_stratification_works(self):
        stratifier = IterativeStratification(n_splits=2, order=1)
        X = np.matrix([[0], [1], [2], [3]])
        y = np.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.assertEqual(len(list(stratifier.split(X, y))), 2)


if __name__ == "__main__":
    unittest.main()
