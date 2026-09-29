# Code and prediction artifacts

Supporting code for:

> A. Chaudhari, K. Jani, D. Pandya and R. Sheth, "A Weighted Soft-Voting Ensemble of CNNs
> for 120-Class Plant Disease Classification on a Large-Scale Laboratory–Field Dataset"
> (under review; the journal reference will be added on acceptance).

The dataset combines 16 public sources with the authors' own field images, which are
published separately as **KhetLeaf**, doi:10.5281/zenodo.22937994 (CC BY 4.0). Supplementary
Table S1 lists every source with its licence; the images of the other sources are not
redistributed here.

## Contents

```
code_release/
├── reproduce_results.py          recomputes Tables 2, 4, 5 and 7 from the stored predictions
├── requirements.txt              versions used for the reported results
├── LICENSE                       MIT
├── notebooks/
│   ├── train_backbone.ipynb      trains one backbone; run once per model
│   └── weighted_voting.ipynb     weighted soft voting over the three probability files
└── predictions/
    ├── combined_model_predictions.csv   per-image predictions of the reported run
    ├── dataset_splits.csv               the 60:20:20 split of all 100,149 images
    ├── class_statistics.csv              per-class image counts and laboratory/field split
    ├── test_domain_tags.csv              laboratory/field tag of each of the 20,118 test images
    ├── test_ground_truth.csv            image, class id, class name for the 20,118 test images
    ├── inception_pred.csv               InceptionV3 class probabilities  (90.66%)
    ├── efficient_pred.csv               EfficientNetB0 class probabilities (88.75%)
    ├── resnet_pred_regenerated.csv      ResNet50, re-inference — see the note below (87.89%)
    └── resnet_predicted_labels.csv      ResNet50 predictions of the reported run (89.62%)
```

## The dataset metadata

`predictions/dataset_splits.csv` records which of the three subsets each of the 100,149 images
belongs to: 60,046 train, 19,985 validation, 20,118 test, over 120 classes. Together with
`predictions/class_statistics.csv` (per-class counts and the laboratory/field split, identical
to Table S2 of the supplementary material) and `predictions/test_domain_tags.csv` (the
laboratory/field tag of every test image), it documents the composition of the compiled dataset.

Note that `file` alone is not a unique key: 58 filenames occur in two different classes,
because some source datasets number their files from 1 within each class. The unique key is
`(class, file)`. No filename occurs twice within the same class, and none appears in more than
one split, so there is no overlap between train, validation and test at the file level.

The images themselves are not redistributed here. The authors' own images are published as the
KhetLeaf dataset, doi:10.5281/zenodo.22937994; the other sixteen sources are available from the
links in Supplementary Table S1, each under its own licence.

## Reproducing the reported numbers

```bash
python3 reproduce_results.py        # needs only numpy and pandas
```

It reads `predictions/combined_model_predictions.csv` — the per-image predictions of the
three backbones and of the ensemble on the test set, with the ground truth and the
laboratory/field tag — and prints Tables 2, 4, 5 and 7.

Accuracy and MCC reproduce the published tables exactly. Macro-averaged precision, recall and
F1 can differ in the second decimal, because they depend on how classes absent from a subset
are handled: this script averages over the classes present in the subset's ground truth, which
is `sklearn`'s `f1_score(..., labels=np.unique(y_true))`. On the field subset only 74 of the
120 classes exist, and averaging over all 120 instead would count the absent ones as zero and
understate the result by roughly 27 points. Every metric and statistical test is implemented
directly in the script so that the numbers do not depend on a scikit-learn or scipy version.

## Training

`notebooks/train_backbone.ipynb` trains one backbone. Set `BACKBONE` to `resnet50`,
`inceptionv3` or `efficientnetb0` and run it once per model — that is how the reported models
were produced: one notebook, re-used, with the backbone and its `preprocess_input` swapped.
Each backbone is paired with the preprocessing its ImageNet weights were trained with.

It writes `<tag>_pred.csv` (class probabilities), `<tag>_history.csv` (per-epoch accuracy and
loss, the source of the training curves), a per-class report and a confusion matrix. It does not
save model weights - see below.

Settings, as in Section 3.2: 224×224 input, rotation up to 90°, width and height shifts up to
50%, brightness in [0.1, 0.7], horizontal and vertical flips; head GAP → 512 → 512 → 120 with
no dropout; all layers trainable; Adamax at 1e-4; batch size 16; 30 epochs; seed 42;
final-epoch weights.

`notebooks/weighted_voting.ipynb` then combines the three probability files with weights
0.3 / 0.4 / 0.3 for ResNet50 / InceptionV3 / EfficientNetB0.

## The two ResNet50 prediction files

`predictions/` holds two files for ResNet50, and they are not interchangeable:

* `resnet_predicted_labels.csv` — the per-image predictions behind Table 2 (89.62%), extracted
  from `combined_model_predictions.csv`. Predicted class labels only.
* `resnet_pred_regenerated.csv` — class probabilities, 87.89%, produced by re-running the
  saved ResNet50 weights. The probability file of the run reported in the paper was
  overwritten by a later training run, so this is the closest available substitute.

`weighted_voting.ipynb` uses the regenerated probabilities and therefore reports about
92.27% rather than the 92.35% in the paper. The published figures are reproduced exactly by
`reproduce_results.py`, which works from the per-image predictions of the reported run in
`combined_model_predictions.csv`. That the ensemble differs by 0.08 points while its ResNet
member differs by 1.7 is a useful indication that the result does not hinge on one file.

Model weights are not included.

## Environment

The reported results were produced under WSL2 on Windows 11 with an NVIDIA T1000 8 GB GPU:
Python 3.9.19, TensorFlow 2.18.0, Keras 3.8.0, CUDA 12.5.1, cuDNN 9. See `requirements.txt`.

## Citing

Please cite the paper above, and the dataset record doi:10.5281/zenodo.22937994 if you use the
authors' own images. Code is released under the MIT licence (`LICENSE`); the prediction files
are released under CC BY 4.0.
