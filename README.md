# Pulsar-Net

![Python](https://img.shields.io/badge/Python-6B7280?style=flat-square&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-6B7280?style=flat-square)

Pulsars are rapidly rotating neutron stars that emit beams of radio waves. Radio surveys produce thousands of pulsar candidates, and automatically distinguishing real pulsars from radio-frequency interference (RFI) and noise can help make these surveys easier to analyse.

Pulsar-Net is a machine learning pipeline for classifying **pulsar candidates as real pulsars or non-pulsars** using statistical features extracted from their radio signals. I used the HTRU2 dataset, engineered additional features around signal shape and characteristics, and compared Logistic Regression with XGBoost. The final model also uses threshold optimisation and SHAP analysis to examine its predictions.

---

## The Data

I used the **[HTRU2 dataset](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2)** from the High Time Resolution Universe Survey.

It contains **17,898 candidates**, of which **1,639 (9.2%) are pulsars**.

Each candidate is described by 8 statistical measurements: mean, standard deviation, kurtosis, and skewness calculated from both the **integrated pulse profile** and the **DM-SNR curve**.

The class imbalance makes accuracy alone insufficient for evaluating a pulsar detector. A model could achieve high accuracy by favouring the majority non-pulsar class. In this case, however, the final model performs well across the different metrics, achieving **98% accuracy, 90% recall, and 84% precision**, alongside a **0.934 PR-AUC**.

**Reference:** R. J. Lyon et al., *Fifty Years of Pulsar Candidate Selection*, MNRAS, 2016.

---

## What I Did

The original 8 measurements describe individual properties of the signals, but don't directly capture relationships between them. I engineered **11 additional features**, bringing the dataset to 19 features in total.

The new features capture relationships involving signal strength, pulse-profile sharpness and concentration, and characteristics of the DM-SNR curve. I also applied sign-preserving logarithmic transformations to several heavily skewed features.

I split the data into training, validation, and test sets using stratification. The training set was used for model fitting and 5-fold cross-validation, while the validation set was used to select the classification threshold. The test set remained untouched until the final evaluation.

I started with Logistic Regression as a baseline, using standardisation and balanced class weights, then trained XGBoost with class weighting and tuned tree complexity, learning rate, and subsampling.

Rather than assuming that 0.5 was the best probability threshold, I searched for the threshold that maximised F2-score on the validation set. This resulted in a final threshold of **0.355**, putting greater emphasis on correctly identifying pulsars.

---

## Results

The two models performed similarly during cross-validation:

| Model               |     CV PR-AUC |
| ------------------- | ------------: |
| Logistic Regression | 0.929 ± 0.012 |
| XGBoost             | 0.928 ± 0.010 |

On the held-out test set, the threshold-optimised XGBoost model achieved:

| Metric    |     Score |
| --------- | --------: |
| Accuracy  |  **0.98** |
| PR-AUC    | **0.934** |
| ROC-AUC   | **0.980** |
| Precision |  **0.84** |
| Recall    |  **0.90** |
| F1        |  **0.87** |

The model correctly identifies **90% of the pulsars** while maintaining **84% precision**, showing that the high overall accuracy is not simply a result of predicting the majority non-pulsar class.

---

## What the Model Learned

The strongest signals in the final model come from the shape of the integrated pulse profile. `log_kurtosis_profile` is the most important feature, followed by the original `kurtosis_profile`, while engineered features such as `pulsar_signature_score` and `log_sharpness_index` also contribute to the predictions.

This points to **pulse-profile shape and peakedness** as particularly useful characteristics for distinguishing real pulsars from RFI and noise.

SHAP analysis provides a more detailed look at how these features influence individual predictions, while the notebook includes confusion matrices, classification reports, and PR and ROC curves to evaluate the model from different perspectives.

---

## How to Run

Download `HTRU_2.csv` from the [HTRU2 dataset](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2) and place it in the project directory.

Then:

```bash
git clone https://github.com/nourawadallah/pulsar-net.git
cd pulsar-net
pip install -r requirements.txt
jupyter notebook pulsar_net.ipynb
```

The dataset is not included in the repository.

## Files

```text
pulsar-net/
├── pulsar_net.ipynb
├── pulsar_xgb_model.pkl
├── requirements.txt
└── README.md
```
