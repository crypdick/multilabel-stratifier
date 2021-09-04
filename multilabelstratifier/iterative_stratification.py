# -*- coding: utf-8 -*-
"""
Iterative stratification for multi-label data.

The classifier follows methods outlined in Sechidis11 and Szymanski17 papers related to stratyfing
multi-label data.

In general what we expect from a given stratification output is that a strata, or a split, is close to a given, demanded size,
usually equal to 1/k in k-split approach, or a x% train to test set division in 2-split splits.

The idea behind this stratification method is to assign label combinations to splits based on how much a given combination is desired
by a given split, as more and more assignments are made, some splits are filled and positive evidence is directed into other splits,
in the end negative evidence is distributed based on a splits desirability of size.

You can also watch a `video presentation <http://videolectures.net/ecmlpkdd2011_tsoumakas_stratification/?q=stratification%20multi%20label>`_ by G. Tsoumakas which explains the algorithm. In 2017 Szymanski & Kajdanowicz extended the algorithm
to handle high-order relationships in the data set, if order = 1, the algorithm falls back to the original Sechidis11 setting.

If order is larger than 1 this class constructs a list of label combinations with replacement, i.e. allowing combinations of lower
order to be take into account. For example for combinations of order 2, the stratifier will consider both
label pairs (1, 2) and single labels denoted as (1,1) in the algorithm. In higher order cases the
when two combinations of different size have similar desirablity: the larger, i.e. more specific combination
is taken into consideration first, thus if a label pair (1,2) and label 1 represented as (1,1) are of similar
desirability, evidence for (1,2) will be assigned to splits first.

You can use this class exactly the same way you would use a normal scikit KFold class:

.. code-block :: python

    from multilabelstratifier.model_selection import IterativeStratification

    k_split = IterativeStratification(n_splits=2, order=1):
    for train, test in k_split.split(X, y):
        classifier.fit(X[train], y[train])
        result = classifier.predict(X[test])
        # do something with the result, comparing it to y[test]

Most of the methods of this class are private, you will not need them unless you are extending the method.

If you use this method to stratify data please cite both:
Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data. Machine Learning and Knowledge Discovery in Databases, 145-158.
http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf

Piotr Szymański, Tomasz Kajdanowicz ; Proceedings of the First International Workshop on Learning with Imbalanced Domains: Theory and Applications, PMLR 74:22-35, 2017.
http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html

Bibtex:

.. code-block:: bibtex

    @article{sechidis2011stratification,
      title={On the stratification of multi-label data},
      author={Sechidis, Konstantinos and Tsoumakas, Grigorios and Vlahavas, Ioannis},
      journal={Machine Learning and Knowledge Discovery in Databases},
      pages={145--158},
      year={2011},
      publisher={Springer}
    }

    @InProceedings{pmlr-v74-szymański17a,
      title =    {A Network Perspective on Stratification of Multi-Label Data},
      author =   {Piotr Szymański and Tomasz Kajdanowicz},
      booktitle =    {Proceedings of the First International Workshop on Learning with Imbalanced Domains: Theory and Applications},
      pages =    {22--35},
      year =     {2017},
      editor =   {Luís Torgo and Bartosz Krawczyk and Paula Branco and Nuno Moniz},
      volume =   {74},
      series =   {Proceedings of Machine Learning Research},
      address =      {ECML-PKDD, Skopje, Macedonia},
      publisher =    {PMLR},
    }
"""

import itertools
import logging
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def iterative_train_test_split(X, y, test_size):
    """
    Iteratively stratified train/test split.

    Parameters
    ----------
    test_size : float, [0,1]
        the proportion of the dataset to include in the test split, the rest will be put in the train set

    Returns
    -------
    X_train, y_train, X_test, y_test
        stratified division into train/test split
    """

    stratifier = IterativeStratification(
        n_splits=2, order=2, sample_distribution_per_split=[test_size, 1.0 - test_size]
    )
    train_indexes, test_indexes = next(stratifier.split(X, y))

    X_train, y_train = X[train_indexes, :], y[train_indexes, :]
    X_test, y_test = X[test_indexes, :], y[test_indexes, :]

    return X_train, y_train, X_test, y_test


def _split_tie_break(desired_samples_per_split, M):
    """
    Helper function to split a tie between splits with same desirability of a given sample.

    Parameters
    ----------
    desired_samples_per_split: np.array[Float], :code:`(n_splits)`
        number of samples desired per split
    M : np.array(int)
        List of splits between which to break the tie

    Returns
    -------
    split_number : int
        The selected split index to put samples into
    """
    if len(M) == 1:
        return M[0]
    else:
        max_val = max(desired_samples_per_split[M])
        M_prim = np.where(np.array(desired_samples_per_split) == max_val)[0]
        M_prim = np.array([x for x in M_prim if x in M])
        return np.random.choice(M_prim, 1)[0]


def _get_most_desired_combination(samples_with_combination):
    """
    Select the next most desired combination whose evidence should be split among splits.

    Parameters
    ----------
    samples_with_combination : Dict[Combination, List[int]], :code:`(n_combinations)`
            map from each label combination present in y to list of sample indexes that have this combination assigned

    Returns
    -------
    combination: Combination
        the combination to split next
    """
    currently_chosen = None
    best_number_of_combinations, best_support_size = None, None

    for combination, evidence in samples_with_combination.items():
        number_of_combinations, support_size = (len(set(combination)), len(evidence))
        if support_size == 0:
            continue
        if currently_chosen is None or (
            best_number_of_combinations < number_of_combinations and best_support_size > support_size
        ):
            currently_chosen = combination
            best_number_of_combinations, best_support_size = number_of_combinations, support_size

    return currently_chosen


class IterativeStratification(_BaseKFold):
    """
    Iteratively stratify a multi-label data set into splits.

    Construct an interative stratifier that splits the data set into splits trying to maintain balanced representation
    with respect to order-th label combinations.

    Attributes
    ----------

    n_splits : number of splits, int
        the number of splits to stratify into

    order : int, >= 1
        the order of label relationship to take into account when balancing sample distribution across labels

    sample_distribution_per_split : None or List[float], :code:`(n_splits)`
        desired percentage of samples in each of the splits, if None and equal distribution of samples per split
        is assumed i.e. 1/n_splits for each split. The value is held in :code:`self.percentage_per_split`.

    random_state : int
        the random state seed (optional)
    """

    def __init__(self, n_splits=3, order=1, sample_distribution_per_split=None, random_state=None):
        self.order = order
        super(IterativeStratification, self).__init__(n_splits, shuffle=False, random_state=random_state)

        if sample_distribution_per_split:
            self.percentage_per_split = sample_distribution_per_split
        else:
            self.percentage_per_split = [1 / float(self.n_splits) for _ in range(self.n_splits)]

    def _prepare_stratification(self):
        """
        Prepares variables for performing stratification.

        For the purpose of clarity, the type Combination denotes List[int], :code:`(self.order)` and represents a
        label combination of the order we want to preserve among splits in stratification. The total number of
        combinations present in :code:`(y)` will be denoted as :code:`(n_combinations)`.

        Sets
        ----

        self.n_samples, self.n_labels : int, int
            shape of y

        self.desired_samples_per_split: np.array[Float], :code:`(n_splits)`
            number of samples desired per split

        self.desired_samples_per_combination_per_split: Dict[Combination, np.array[Float]], :code:`(n_combinations, n_splits)`
            number of samples evidencing each combination desired per each split

        Parameters
        ----------

        y : output matrix or array of arrays (n_samples, n_labels)

        Returns
        -------

        rows : List[List[int]], :code:`(n_samples, n_labels)`
            list of label indices assigned to each sample

        rows_used : Dict[int, bool], :code:`(n_samples)`
            boolean map from a given sample index to boolean value whether it has been already assigned to a split or not

        all_combinations :  List[Combination], :code:`(n_combinations)`
            list of all label combinations of order self.order present in y

        per_row_combinations : List[Combination], :code:`(n_samples)`
            list of all label combinations of order self.order present in y per row

        samples_with_combination : Dict[Combination, List[int]], :code:`(n_combinations)`
            map from each label combination present in y to list of sample indexes that have this combination assigned

        splits: List[List[int]] (n_splits)
            list of lists to be populated with samples
        """
        logger.info("Preparing stratification")
        start = time.time()
        # for every row
        for sample_index, label_assignment in enumerate(self.rows):
            # for every n-th order label combination
            # register combination in maps and lists used later
            for combination in itertools.combinations_with_replacement(label_assignment, self.order):
                if combination not in self.samples_with_combination:
                    self.samples_with_combination[combination] = []

                self.samples_with_combination[combination].append(sample_index)
                self.all_combinations.append(combination)
                self.per_row_combinations[sample_index].append(combination)

        self.all_combinations = [list(x) for x in set(self.all_combinations)]

        self.desired_samples_per_combination_per_split = {
            combination: np.array(
                [len(evidence_for_combination) * self.percentage_per_split[j] for j in range(self.n_splits)]
            )
            for combination, evidence_for_combination in self.samples_with_combination.items()
        }
        logger.info(f"Prep stratification finished in {time.time()-start}")

    def _distribute_positive_evidence(self):
        """
        Internal method to distribute evidence for labeled samples across splits.

        For params, see documentation of :code:`self._prepare_stratification`. Does not return anything, modifies
        params.
        """
        logger.info("Distributing positive evidence")
        start = time.time()

        most_desirable_combo = _get_most_desired_combination(self.samples_with_combination)
        while most_desirable_combo is not None:
            while len(self.samples_with_combination[most_desirable_combo]) > 0:
                row = self.samples_with_combination[most_desirable_combo].pop()
                if self.rows_used[row]:
                    continue

                max_val = max(self.desired_samples_per_combination_per_split[most_desirable_combo])
                M = np.where(np.array(self.desired_samples_per_combination_per_split[most_desirable_combo]) == max_val)[
                    0
                ]
                m = _split_tie_break(self.desired_samples_per_combination_per_split[most_desirable_combo], M)
                self.splits[m].append(row)
                self.rows_used[row] = True
                for i in self.per_row_combinations[row]:
                    if row in self.samples_with_combination[i]:
                        self.samples_with_combination[i].remove(row)
                    self.desired_samples_per_combination_per_split[i][m] -= 1
                self.desired_samples_per_split[m] -= 1

            most_desirable_combo = _get_most_desired_combination(self.samples_with_combination)

        logger.info(f"Distributing positive evidence finished in {time.time()-start}")

    def _distribute_negative_evidence(self):
        """
        Internal method to distribute evidence for unlabeled samples across splits.

        For params, see documentation of :code:`self._prepare_stratification`. Does not return anything, modifies
        params.
        """
        logger.info("Distributing negative evidence")
        start = time.time()

        available_samples = [i for i, v in self.rows_used.items() if not v]
        samples_left = len(available_samples)

        while samples_left > 0:
            row = available_samples.pop()
            self.rows_used[row] = True
            samples_left -= 1
            split_selected = np.random.choice(np.where(self.desired_samples_per_split > 0)[0], 1)[0]
            self.desired_samples_per_split[split_selected] -= 1
            self.splits[split_selected].append(row)

        logger.info(f"Distributing negative evidence finished in {time.time()-start}")

    def _iter_test_indices(self, X, y=None, groups=None):
        """
        Internal method for providing scikit-learn's split with splits.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        split : List[int]
            indexes of test samples for a given split, yielded for each of the splits
        """
        if self.random_state:
            check_random_state(self.random_state)  # todo this doesn't seem to have any effect

        self._init_vars(y)

        self._prepare_stratification()

        self._distribute_positive_evidence()
        self._distribute_negative_evidence()

        for split in self.splits:
            yield split

    def _init_vars(self, y):
        self.y = y
        self.n_samples, self.n_labels = self.y.shape
        self.desired_samples_per_split = np.array(
            [self.percentage_per_split[i] * self.n_samples for i in range(self.n_splits)]
        )
        self.rows = sp.lil_matrix(y).rows
        self.rows_used = {i: False for i in range(self.n_samples)}
        self.all_combinations = []
        self.per_row_combinations = [[] for i in range(self.n_samples)]
        self.samples_with_combination = {}
        self.splits = [[] for _ in range(self.n_splits)]
