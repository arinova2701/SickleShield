# SickleShield 🛡️

A machine learning triage system for Sickle Cell Disease in rural India.

In tribal regions of Chhattisgarh, Madhya Pradesh, and Odisha, Sickle Cell Disease is common but hard to catch early. The nearest lab is often hours away, and many families make that trip only to get a negative result. SickleShield is a first-pass filter: it estimates risk before anyone travels, so testing reaches the people who actually need it.

> **Triage** means sorting who needs care first. Here, it means flagging likely cases before a long, costly trip to a diagnostic centre.

## The Problem

Sickle Cell Disease heavily affects tribal communities in Chhattisgarh, Madhya Pradesh, Odisha, and Gujarat. Diagnosis needs a blood test available only in city labs, so families travel hours, often for nothing. ICMR's National Sickle Cell Mission (2023) trained 21,000 workers, but rural screening is still scarce.

## How It Works

```
Patient at home
      |
[1] Symptom questionnaire  ->  risk score
      |
High risk -> visit nearest PHC
      |
[2] Health worker uploads blood smear photo
      |
Model flags: Sickle Cell / Normal
      |
Refer to hospital  /  All clear
```

No costly equipment. Just a phone and a basic PHC microscope.

## Component 1: Symptom Risk Scorer · `In Progress`

Self-reported inputs, no equipment needed: age, gender, community, family history, pain episodes, jaundice, fatigue, frequent infections.

**Model:** XGBoost

No public Sickle Cell symptom dataset exists yet, so this trains on an anaemia dataset as a proxy to prove the pipeline works (the two conditions share features like fatigue and jaundice). A data request is in with ICMR to retrain on real screening data.

## Component 2: Blood Smear Classifier · `Complete`

A health worker uploads a slide photo and the model detects sickle-shaped cells instantly.

**Model:** ResNet18 with transfer learning
- Trained the final layer, then unfroze `layer4` with differential learning rates
- Handled the 5.74:1 class imbalance with weighted loss
- Augmentation: flips, rotation, colour jitter

**Results**

| Metric | Value |
|---|---|
| Accuracy | 96% |
| Sickle Recall | 98% |
| Sickle F1 | 0.98 |
| Macro F1 | 0.91 |
| Normal Recall | 86% |

![Training History](assets/training_history.png)

![Confusion Matrix](assets/confusion_matrix_seaborn.png)

**Limitations**
- Normal recall (86%) is limited by only 147 normal images
- Dataset is from Uganda, not India (cell shape generalises, but this needs validation)

## Roadmap

| Phase | Status |
|---|---|
| Component 2 image classifier | Done |
| Component 1 symptom scorer | In progress |
| ICMR data request | Submitted |
| Streamlit interface for PHCs | Planned |
| Retrain on ICMR data | Planned |
| Multi-disease triage (TB, anaemia) | Future |

## Dataset

**Component 2:** Florence Tushabe et al., Sickle Cell Microscopy Dataset (Uganda). 991 images (844 positive, 147 negative), 80/20 split.

**Component 1:** Anaemia proxy dataset, to be replaced with ICMR data.

## Setup

```bash
git clone https://github.com/arinova2701/SickleShield.git
cd SickleShield
pip install -r requirements.txt

kaggle datasets download -d florencetushabe/sickle-cell-disease-dataset \
  --unzip -p data/sickle

jupyter notebook component2/sickle_cell_eda.ipynb
```

## Acknowledgements

Dataset by Florence Tushabe, Samuel Mwesige et al., Soroti University, Uganda. Motivated by the ICMR National Sickle Cell Anaemia Mission (2023).

---

> *Stay curious. Build relentlessly.*
