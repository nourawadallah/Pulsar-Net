# Pulsar-Net — Pulsar Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Pulsars are rapidly rotating neutron stars that emit beams of radio waves. Radio surveys produce thousands of pulsar candidates, but most are not actually pulsars and instead come from radio-frequency interference (RFI) or noise. This makes automatically filtering candidates an important step in pulsar searches.

Pulsar-Net is a machine learning pipeline that classifies radio survey candidates as **pulsars or non-pulsars**. I used the HTRU2 dataset and built additional features around the shape and signal characteristics of each candidate, then compared a Logistic Regression baseline against XGBoost. The final model also uses threshold optimisation and SHAP analysis to examine how it makes its predictions.

---

## The Data

I used the **[HTRU2 dataset](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2)** from the High Time Resolution Universe Survey.

It contains **17,898 candidates**, of which only **1,639 (9.2%) are pulsars**.

Each candidate starts with 8 statistical measurements describing its integrated pulse profile and DM-SNR curve:

* mean
* standard deviation
* kurtosis
* skewness

for each of the two signal representations.

The strong class imbalance makes accuracy a poor primary metric. A model that predicted every candidate as a non-pulsar would already achieve around 91% accuracy while finding zero pulsars, so I used **PR-AUC** as the main metric.

**Reference:** R. J. Lyon et al., *Fifty Years of Pulsar Candidate Selection*, MNRAS, 2016.

---

## What I Did

The original 8 measurements describe individual properties of the signals, but don't directly capture relationships between them. I engineered **11 additional features**, bringing the dataset to 19 features in total.

The new features capture things like:

* signal strength relative to noise
* pulse-profile sharpness and concentration
* relationships between the pulse profile and DM-SNR curve
* characteristics associated with dispersion through the interstellar medium

I also applied sign-preserving logarithmic transformations to several heavily skewed features so that extreme values would have less influence on the models.

I split the data into training, validation, and test sets using stratification. The training set was used for model fitting and 5-fold cross-validation, the validation set was reserved for selecting the classification threshold, and the test set was kept untouched until the final evaluation.

### Baseline

I started with Logistic Regression using standardisation and balanced class weights.

### XGBoost

I then trained an XGBoost classifier with class weighting and tuned tree complexity, learning rate, and subsampling. It was evaluated using the same 5-fold stratified cross-validation as the baseline.

Rather than assuming that a probability of 0.5 should separate pulsars from non-pulsars, I searched for a better threshold on the validation set. I selected the threshold that maximised **F2-score**, putting more emphasis on correctly finding pulsars.

---

## Results

The two models performed very similarly during cross-validation:

| Model               |     CV PR-AUC |
| ------------------- | ------------: |
| Logistic Regression | 0.929 ± 0.012 |
| XGBoost             | 0.928 ± 0.010 |

After threshold optimisation, XGBoost achieved the following results on the held-out test set:

| Metric    |     Score |
| --------- | --------: |
| PR-AUC    | **0.934** |
| ROC-AUC   | **0.980** |
| Precision |  **0.84** |
| Recall    |  **0.90** |
| F1        |  **0.87** |

The selected classification threshold was **0.355**.

This gives the model a 90% recall while still maintaining 84% precision, which is particularly useful for pulsar searches where missing real pulsars is costly.

---

## What the Model Learned

The XGBoost feature importance was heavily dominated by the pulse profile's kurtosis:

| Feature                  | Importance |
| ------------------------ | ---------: |
| `log_kurtosis_profile`   |      0.480 |
| `kurtosis_profile`       |      0.261 |
| `pulsar_signature_score` |      0.032 |
| `std_dmsnr`              |      0.025 |
| `log_sharpness_index`    |      0.024 |

The model therefore relies heavily on **the shape and peakedness of the integrated pulse profile** when separating pulsars from non-pulsars.

SHAP analysis is included to go beyond feature importance and show how these features influence individual predictions.

The notebook also includes confusion matrices, classification reports, and precision-recall and ROC curves to inspect the model from multiple perspectives.

---

## Try It Yourself

Download `HTRU_2.csv` from the [HTRU2 dataset](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2) and place it in the project directory.

Then:

```bash
git clone https://github.com/nourawadallah/pulsar-net.git
cd pulsar-net
pip install -r requirements.txt
jupyter notebook pulsar_net.ipynb
```

## Project Files

```text
pulsar-net/
├── pulsar_net.ipynb
├── pulsar_xgb_model.pkl
├── requirements.txt
└── README.md
```
