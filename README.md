# Code for Paper: Is Epistemic Uncertainty Faithfilly Represented by Evidential Deep Learning Methods?

This repository provides the code to reproduce the experiments from the paper
 [Is Epistemic Uncertainty Faithfully Represented by Evidential Deep Learning Methods?](https://openreview.net/forum?id=mxjB0LIgpT),
 accepted for publication at ICML 2024.

Authors: Mira Juergens, Nis Meinert, Viktor Bengs, Eyke Huellermeier and Willem Waegeman.
## Abstract

Trustworthy ML systems should not only return accurate predictions, but also a reliable representation of their uncertainty. Bayesian methods are commonly used to quantify both aleatoric and epistemic uncertainty, but alternative approaches, such as evidential deep learning methods, have become popular in recent years. The latter group of methods in essence extends empirical risk minimization (ERM) for predicting second-order probability distributions over outcomes, from which measures of epistemic (and aleatoric) uncertainty can be extracted. This paper presents novel theoretical insights of evidential deep learning, highlighting the difficulties in optimizing second-order loss functions and interpreting the resulting epistemic uncertainty measures. With a systematic setup that covers a wide range of approaches for classification, regression and counts, it provides novel insights into issues of identifiability and convergence in second-order loss minimization, and the relative (rather than absolute) nature of epistemic uncertainty measures.

## Method

## Citation
If the code or paper was used for your research, please consider citing our paper:
````
@inproceedings{
juergens2024is,
title={Is Epistemic Uncertainty Faithfully Represented by Evidential Deep Learning Methods?},
author={Mira Juergens and Nis Meinert and Viktor Bengs and Eyke H{\"u}llermeier and Willem Waegeman},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=mxjB0LIgpT}
}
````
