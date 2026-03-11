
# ECG Signal Processing vs Representation Learning on MIMIC-IV-ECG

This project investigates how **classical ECG signal processing features** compare with **pretrained representation learning models** (HuBERT-ECG) for ECG classification tasks using the **MIMIC-IV-ECG** dataset.

Two classification tasks are studied:

- **Atrial Fibrillation Detection:** AFIB vs Normal ECG
- **Abnormal ECG Detection:** Abnormal vs Normal ECG

The goal is to evaluate whether **pretrained ECG encoders can outperform interpretable signal processing pipelines** for clinical ECG analysis.

We compare three feature representations:

| Representation | Description |
|---|---|
| Signal Features | Classical ECG signal processing features |
| HuBERT Embeddings | 512-dimensional embeddings from pretrained HuBERT-ECG |
| Signal + HuBERT | Combined feature set |

Downstream models evaluated:

- Logistic Regression
- XGBoost


## Project Structure

```
├── data/
│ ├── metadata/             # Raw metadata CSV files from MIMIC-IV-ECG
│ ├── raw_waveforms/        # Raw ECG waveform files
│ ├── processed/            # Processed metadata and feature tables
│ ├── figures/              # Figures from exploratory analysis
│ └── modeling_datasets/    # Fully processed final datasets for modeling
│
├── notebooks/
│ ├── build_ecg_metadata_table.ipynb        # Dev of cohort selection and label construction
│ ├── build_signal_feature_tables.ipynb    # Dev of classical ECG signal processing features
│ ├── build_hubert_embedding_tables.ipynb  # Dev of HuBERT ECG embedding pipeline
│ └── build_model_datasets.ipynb            # Construction of final modeling datasets
│
├── config/
│ ├── feature_columns.json    # Config file for feature selection for experiments
│ └── aria2_input.txt         # Download manifest for ECG waveform files
│
├── src/
│ ├── load_ecg.py                           # Utility for loading waveform files
│ ├── build_ecg_metadata_table.py           # Metadata and cohort construction pipeline
│ ├── build_signal_feature_tables.py        # Classical signal feature extraction pipeline
│ ├── build_hubert_embedding_tables.py      # HuBERT embedding encoding pipeline
│ ├── build_model_datasets.py               # Construction of modeling datasets
│ └── run_model_experiments.ipynb           # Performs modeling experiment and analysis results
│
├── results/
│ └── figures/                # Figures from experiment analysis
│
└── README.md
```

---

## Dataset

This project uses the **MIMIC-IV-ECG Matched Subset**, which contains:

- 12-lead ECG waveforms  
- 10-second recordings  
- 500 Hz sampling rate  
- machine generated ECG interpretation reports  

Access to the dataset requires credentialed access through **PhysioNet**.


## Label Construction

ECG labels are derived from the machine interpretation text provided in the MIMIC-IV-ECG metadata.

Three mutually exclusive cohorts are defined:

| Label | Description |
|------|-------------|
| **is_af** | ECG reports containing atrial fibrillation |
| **is_normal_strict** | ECG reports explicitly labeled normal |
| **is_clearly_abnormal** | ECG reports labeled abnormal (excluding AF) |

Ambiguous ECGs (e.g., borderline findings or nonspecific abnormalities) are excluded from classification tasks.


## Classification Tasks

Two binary classification tasks are constructed.

### Task 1: Atrial Fibrillation Detection

Atrial Fibrillation vs Normal ECG reports

### Task 2: Abnormal ECG Detection

Abnormal vs Normal ECG reports


## Subset Construction

Downloading and processing the entire MIMIC-IV-ECG dataset is computationally expensive.  
To make experiments tractable, a balanced subset of ECG recordings is constructed.

The subset contains:

| Cohort | Size |
|------|------|
| Atrial Fibrillation | 5,000 |
| Strict Normal | 5,000 |
| Clearly Abnormal | 5,000 |

The **same 5,000 strictly normal ECGs** are used in both classification tasks as the control group.

This produces:

AF vs Normal dataset

- 5,000 AF  
- 5,000 Normal  

Normal vs Abnormal dataset

- 5,000 Normal  
- 5,000 Abnormal  

## Feature Pipelines (Signal Features vs Representation Embeddings)

### Classical ECG Signal Features

Classical ECG analysis relies heavily on manually engineered features derived
from signal processing and cardiology knowledge. Designing these features
requires understanding:

- ECG waveform morphology
- cardiac electrophysiology
- rhythm variability
- spectral characteristics of biosignals

While these features provide interpretable physiological summaries of ECG
signals, they may fail to capture complex patterns present in raw waveform
data.

This motivates the use of **representation learning approaches**, such as
pretrained ECG encoders, which learn feature representations directly from
large-scale ECG datasets.

#### 1. Time-Domain Waveform Statistics

Basic statistical descriptors summarize amplitude characteristics of the ECG
waveform.

These include:

- mean
- standard deviation
- skewness
- kurtosis
- minimum and maximum amplitude
- peak-to-peak range

These statistics capture overall waveform variability and morphological
irregularities that may indicate abnormal cardiac activity.

Time-domain statistics are commonly used as baseline descriptors in ECG and
biosignal analysis pipelines.

#### 2. Spectral Features

ECG signals also contain important information in the frequency domain.
Spectral features are computed from the power spectral density of the ECG
signal.

Extracted features include:

- spectral entropy
- dominant frequency
- spectral centroid
- bandpower estimates

Spectral entropy measures signal complexity and disorder in the frequency
distribution, while the spectral centroid provides a weighted average
frequency representing the center of mass of the power spectrum.

Frequency-domain descriptors have been widely used in biomedical signal
analysis for characterizing rhythmic physiological processes.

#### 3. Hjorth Parameters

Hjorth parameters provide compact descriptors of signal dynamics and are
widely used in EEG and other electrophysiological signal analyses.

Three Hjorth parameters are computed:

- **Activity** — signal variance
- **Mobility** — mean frequency of the signal
- **Complexity** — variation in frequency over time

These parameters summarize signal complexity and oscillatory behavior and are
particularly useful for detecting irregular rhythms and morphological
variations in physiological signals.

#### 4. R-Peak and RR Interval Features

Cardiac rhythm information is captured using features derived from
R-peak detection.

R-peaks correspond to ventricular depolarization events and provide the basis
for measuring heart rate and rhythm variability.

Extracted features include:

- number of detected R-peaks
- mean RR interval
- RR interval standard deviation
- RMSSD (root mean square of successive differences)
- RR interval range
- estimated heart rate (BPM)

These features quantify beat-to-beat variability and rhythm irregularities,
which are particularly important for detecting arrhythmias such as atrial
fibrillation.

#### Feature Extraction Summary

| Feature Category | Description |
|---|---|
| Time-domain statistics | Waveform amplitude distribution and variability |
| Spectral features | Frequency-domain structure of the ECG signal |
| Hjorth parameters | Signal complexity and oscillatory dynamics |
| RR interval features | Heart rate variability and rhythm irregularity |

Total features extracted: **62 engineered ECG features**.


### Representation Learning (HuBERT ECG)

While classical signal processed features provide interpretable physiological summaries 
of ECG signals, they require extensive domain knowledge and may fail to capture complex 
patterns present in raw waveform data.

This motivates the use of **representation learning approaches**, such as
pretrained ECG encoders, which learn feature representations directly from
large-scale ECG datasets.

ECG waveforms are also encoded using a **pretrained HuBERT-ECG transformer model**.

The encoder produces:

- **512-dimensional embedding vectors**
- learned representations of ECG waveform structure

These embeddings serve as input features for downstream classifiers.


---

## Quick Start

### 1. Generate Metadata Tables

**Before running script:** download metadata files from MIMIC-IV-ECG.

```bash
wget -r -np -l 1 -nd -A "*.csv" https://physionet.org/files/mimic-iv-ecg/1.0/
```

Place downloaded metadata files into:

```bash
data/metadata/
```

Run script from the project root:

```bash
python src/build_ecg_metadata_table.py --make-plots --write-aria2
```

Optional flags:

```
--make-plots      Save cohort distribution figure
--write-aria2     Generate waveform download manifest (required for step 2)
```

This script:

- loads MIMIC-IV-ECG metadata files  
- constructs ECG interpretation reports  
- parses ECG labels  
- creates classification cohorts  
- samples the balanced subset used for experiments  
- generates an ECG waveform download manifest

Output metadata tables are written to:

```
data/processed/
```


### 2. Download ECG Waveforms

ECG Waveforms were downloaded by making 32 parralel requests to MIMIC-IV-ECG as download speed is bottlenecked by the large 
number of small files, requiring about 800k x 2 (# of ecg records * two files per record) ~ 1.6 million requests. Each file
downloads instantly but starting and stopping each request takes most of the time. We will use **aria2c** to make 32 parrel 
requests to speed up the download prcoess. 

First, install aria2c using your preferred package manager. `aria2_input.txt` is generated by the script in step 1 and contains
the request url and download filepath for each waveform. 

Run the command below in your terminal to start the download. ***Warning: takes 2-3 hours to download the 15k subset!***

```bash
aria2c -i config/aria2_input.txt \
-j 32 \
--enable-http-pipelining=true \
--min-split-size=1M \
--continue=true \
--auto-file-renaming=false
```

Downloaded waveform files should be stored in:

```
data/raw_waveforms/
```


### 3. Build Classical Signal Processing Feature Tables

```bash
python src/build_signal_feature_tables.py
```

This pipeline:

- loads waveform files  
- performs signal validation  
- computes ECG signal processing features
- builds a feature table for each classification task set

Outputs are saved to:

```
data/processed/
```


### 4. Generate HuBERT Embeddings

```bash
python src/build_hubert_embedding_tables.py
```

This pipeline:

- loads ECG waveform segments
- encodes signals using the pretrained HuBERT ECG model
- produces 512-dimensional embedding vectors
- builds a feature table for each classification task set

Outputs are written to:

```
data/processed/
```


### 5. Construct Final Modeling Datasets

```bash
python src/build_model_datasets.py
```

This script:

- merges signal features and embeddings
- constructs feature matrices for each task
- outputs modeling-ready datasets

Outputs are written to:

```
data/modeling_datasets/
```


### Pipeline Summary

The full pipeline consists of five stages:

1. **Metadata Construction**  
   `build_ecg_metadata_table.py`

2. **Waveform Download**  
   `aria2_input.txt → data/raw_waveforms/`

3. **ECG Signal Processing**  
   `build_signal_feature_tables.py`

4. **Embedding Generation**  
   `build_hubert_embedding_tables.py`

5. **Construct Model Ready Datasets**  
   `build_model_datasets.py`


---

## Signal Featues vs HuBERT Embedding Modeling Experiment

Modeling experiments are performed in:

```bash
src/run_model_experiments.ipynb
```

Three feature sets are evaluated:

| Feature Set | Description |
|---|---|
Signal | 62 engineered features |
HuBERT | 512 embedding features |
Signal + HuBERT | Combined features |

Models evaluated:

- Logistic Regression  
- XGBoost  

The experiment is analysis of a 3 by 2 (feature sets x models) model evaluation grid. 

Evaluation metrics include:

- AUROC  
- F1 Score  
- Precision  
- Recall  
- Accuracy  

Experiment notebook structure:

- load and inspect datasets
- run model training grid (train a logistic reg model and a XGBoost model for each feature set)
- compare signal features and HuBERT embeddings task performance
- performs cross-validation for robust validation
- generates result tables and figures

Outputs are written to:

```bash
results/  
results/figures/
```

## Results Summary

Across both classification tasks, **HuBERT embeddings consistently outperform classical ECG signal features**.

Combining signal features with embeddings provides only marginal improvement, suggesting that the pretrained encoder already captures much of the relevant waveform structure.

These results highlight the potential of **representation learning for automated ECG analysis**.


## Data Availability & License

This project uses the **MIMIC-IV-ECG dataset**, which is subject to PhysioNet data use agreements.

Raw ECG waveforms and derived datasets are not included in this repository and all datasets can be reproduced by running the pipeline scripts described above.
