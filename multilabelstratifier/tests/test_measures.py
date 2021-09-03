# import unittest
#
# from multilabelstratifier import (
#     example_distribution,
#     folds_label_combination_pairs_without_evidence,
#     folds_without_evidence_for_at_least_one_label_combination,
#     label_combination_distribution,
#     percentage_of_label_combinations_without_evidence_per_fold,
# )
#
#
# class FoldMeasuresTests(unittest.TestCase):
#     def test_example_distribution_calculates_proper_value(self, fixture_y, fixture_folds):
#         n_samples = float(fixture_y.shape[0])
#         for fold_data in fixture_folds:
#             folds, expected_ed = fold_data[0], fold_data[1]
#             equal_distribution = [n_samples / len(folds) for _ in range(len(folds))]
#             self.assertEqual(example_distribution(folds, equal_distribution), expected_ed)
#
#     def test_label_distribution_calculates_proper_value(self, fixture_y, fixture_folds):
#         for fold_data in fixture_folds:
#             folds, expected_ld, expected_lpd = fold_data[0], fold_data[2], fold_data[3]
#             self.assertEqual(label_combination_distribution(fixture_y, folds, 1), expected_ld)
#             self.assertEqual(label_combination_distribution(fixture_y, folds, 2), expected_lpd)
#
#     def test_folds_without_evidence_counts_proper_values(self, fixture_folds, fixture_y):
#         for fold_data in fixture_folds:
#             folds, expected_fold_count, expected_expected_fold_count_pairs = fold_data[0], fold_data[4], fold_data[6]
#             self.assertEqual(
#                 folds_without_evidence_for_at_least_one_label_combination(fixture_y, folds, order=1),
#                 expected_fold_count,
#             )
#             self.assertEqual(
#                 folds_without_evidence_for_at_least_one_label_combination(fixture_y, folds, order=2),
#                 expected_expected_fold_count_pairs,
#             )
#
#     def test_folds_combination_pairs_without_evidence_counts_proper_values(self, fixture_folds, fixture_y):
#         for fold_data in fixture_folds:
#             folds, expected_fold_count, expected_expected_fold_count_pairs = fold_data[0], fold_data[5], fold_data[7]
#             self.assertEqual(
#                 folds_label_combination_pairs_without_evidence(fixture_y, folds, order=1), expected_fold_count
#             )
#             self.assertEqual(
#                 folds_label_combination_pairs_without_evidence(fixture_y, folds, order=2),
#                 expected_expected_fold_count_pairs,
#             )
#
#     def test_label_evidence_percentages_calculate_proper_values(self, fixture_folds, fixture_y):
#         for fold_data in fixture_folds:
#             folds, expected_label_percentage, expected_label_pair_percentage = fold_data[0], fold_data[8], fold_data[9]
#             self.assertEqual(
#                 percentage_of_label_combinations_without_evidence_per_fold(fixture_y, folds, order=1),
#                 expected_label_percentage,
#             )
#             self.assertEqual(
#                 percentage_of_label_combinations_without_evidence_per_fold(fixture_y, folds, order=2),
#                 expected_label_pair_percentage,
#             )
