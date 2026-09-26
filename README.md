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
    ├── test_ground_truth.csv            image, class id, class name for the 20,118 test images
    ├── inception_pred.csv               InceptionV3 class probabilities  (90.66%)
    ├── efficient_pred.csv               EfficientNetB0 class probabilities (88.75%)
    ├── resnet_pred_regenerated.csv      ResNet50, re-inference — see the note below (87.89%)
    └── resnet_predicted_labels.csv      ResNet50 predictions of the reported run (89.62%)
```

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

## Why no model weights are included

The checkpoints of the reported training runs were not retained. Three `.keras` files survived
in the authors' working folder, and each was tested against the probability file it should
reproduce, on a 2,012-image stratified sample of the test set:

| Saved checkpoint | Parameters | Best per-image agreement with its prediction file |
|---|---|---|
| ResNet50 | 24,961,016 | 91.4% |
| InceptionV3 | 23,176,088 | 93.3% |
| EfficientNetB0 | 5,029,659 | 2.3% |

The parameter counts match the architectures in Table 6, but none of the checkpoints reproduces
the predictions of its reported run - the agreement of a checkpoint with its own run would be
essentially 100%, and image-resizing differences move it by only a few points. The
EfficientNetB0 file performs at chance under every candidate preprocessing and was evidently
saved before training. Publishing them would place files in a permanent record that readers
would reasonably take for the reported models, so none are included and the training notebook
does not write any.

In the same vein, the probability file of the reported ResNet50 run no longer exists: it was
overwritten on 24 July 2025 by a later run of the same notebook. What this archive contains
instead:

* `resnet_predicted_labels.csv` - the **reported** run's per-image predictions, 89.62%,
  extracted from `combined_model_predictions.csv`. Labels only; probabilities cannot be
  recovered from labels and none has been fabricated.
* `resnet_pred_regenerated.csv` - a re-inference from the surviving ResNet50 checkpoint,
  87.89%. Real model output, but not the run in the paper.

`weighted_voting.ipynb` uses the regenerated file and so reports about 92.27% rather than the
published 92.35%. That the ensemble moves by 0.08 points while its ResNet member moves by 1.7
is some evidence that the reported result does not rest on one particular file. The published
figures themselves are reproduced exactly by `reproduce_results.py` from the per-image
predictions of the reported run.

## Environment

The reported results were produced under WSL2 on Windows 11 with an NVIDIA T1000 8 GB GPU:
Python 3.9.19, TensorFlow 2.18.0, Keras 3.8.0, CUDA 12.5.1, cuDNN 9. See `requirements.txt`.

## Citing

Please cite the paper above, and the dataset record doi:10.5281/zenodo.22937994 if you use the
authors' own images. Code is released under the MIT licence (`LICENSE`); the prediction files
are released under CC BY 4.0.
