# multilabel-stratifier

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Build status](https://ci.appveyor.com/api/projects/status/d04lx2q3viw090my?svg=true)](https://ci.appveyor.com/project/crypdick/multilabel-stratifier)

This is a fork of [scikit-multilearn](http://scikit.ml), which unfortunately appears abandoned. In particular, I am maintaining a version of the IterativeStratification algorithm.

## Installation

To install scikit-multilearn, simply type the following command:

```bash
$ pip install -e git+git://github.com/crypdick/multilabel-stratifier#egg=multilabel-stratifier
```

This will install the latest release from Github. Eventually, I will get this on the Python package index.

## Basic Usage

Before proceeding to classification,  this library assumes that you have
a dataset with the following matrices:

- `x_train`, `x_test`: training and test feature matrices of size `(n_samples, n_features)`
- `y_train`, `y_test`: training and test label matrices of size `(n_samples, n_labels)`


More examples and use-cases can be seen in the
[documentation](http://scikit.ml/api/classify.html). For using the MEKA
wrapper, check this [link](http://scikit.ml/api/meka.html#mekawrapper).

## Contributing

This project is open for contributions. Here are some of the ways for
you to contribute:

- Bug reports/fix
- Features requests
- Use-case demonstrations
- Documentation updates

In case you want to implement your own multi-label classifier, please
read our [Developer's Guide](http://scikit.ml/api/base.html) to help
you integrate your implementation in our API.

To make a contribution, just fork this repository, push the changes
in your fork, open up an issue, and make a Pull Request!


## Cite

If you used scikit-multilearn in your research or project, please
cite [our work](https://arxiv.org/abs/1702.01460):

```bibtex
@ARTICLE{2017arXiv170201460S,
   author = {{Szyma{\'n}ski}, P. and {Kajdanowicz}, T.},
   title = "{A scikit-based Python environment for performing multi-label classification}",
   journal = {ArXiv e-prints},
   archivePrefix = "arXiv",
   eprint = {1702.01460},
   year = 2017,
   month = feb
}
```
