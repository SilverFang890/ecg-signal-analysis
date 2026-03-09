
# ECG Signal Processing vs Representation Learning on MIMIC-IV-ECG

This project investigates how **classical ECG signal processing features** compare with **pretrained representation learning models** (e.g., HuBERT-ECG) for ECG classification tasks using the **MIMIC-IV-ECG** dataset.

Two classification tasks are studied:

- **Atrial Fibrillation Detection:** Afib vs Normal ECG
- **Abnormal ECG Detection:** Abnormal vs Normal ECG

The goal is to evaluate whether modern pretrained ECG encoders outperform interpretable signal processing pipelines for clinical ECG analysis.

---

## Project Structure

```
├── data/                                 
│   ├── metadata/                       # Raw metadata CSV files from MIMIC-IV-ECG
│   ├── raw_waveforms/                  # Raw ecg waveforms downloaded from MIMIC-IV-ECG
│   ├── processed/                      # Processed dataset saved as parquets
│   └── figures/                        # Figures from exploratory data analysis
│
├── notebooks/
│   ├── extract_data.ipynb              # Notebook for development of data extraction and label creation
│   └── extract_ecg_features.ipynb      # Finalized ecg processing pipeline
│
├── utils/
│   └── aria2_input.txt                 # aria2c input to download large amounts of ecg files from MIMIC-IV-ECG
│
├── src/                                       
│   ├── load_ecg.py                     # Helper module for loading ecg data                         
│   ├── build_mimic_ecg_metadata.py     # Finalized data extraction pipeline
│   └── build_ecg_feature_tables.py     # Finalized ecg processing pipeline
│
└── README.md
```


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


## Quick Start

This section describes the minimal steps required to reproduce the dataset and feature tables used in this project.


### 1. Generate Metadata Tables

From the project root:

```bash
python src/build_mimic_ecg_metadata.py --make-plots --write-aria2
```

Optional flags:

```
--make-plots      Save cohort distribution figure
--write-aria2     Generate waveform download manifest
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

Waveforms are downloaded using the generated `aria2_input.txt`.

Example command:

```bash
aria2c -i utils/aria2_input.txt \
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


### 3. Build ECG Feature Tables

Once waveform files are downloaded, run the ECG processing pipeline:

```bash
python src/build_ecg_feature_tables.py
```

This pipeline:

- loads waveform files  
- performs signal validation  
- computes ECG signal processing features  
- builds feature tables for downstream modeling

Outputs are saved to:

```
data/processed/
```


## Pipeline Summary

The full pipeline consists of three stages:

1. **Metadata Construction**  
   `build_mimic_ecg_metadata.py`

2. **Waveform Download**  
   `aria2_input.txt → data/raw_waveforms/`

3. **Feature Extraction**  
   `build_ecg_feature_tables.py`

---

### Output Artifacts

| Output | Description |
|------|-------------|
| `meta_labels.parquet` | Master ECG metadata table |
| `afib_metadata.parquet` | AF vs Normal metadata |
| `afib_subset_metadata.parquet` | AF vs Norm subset metadata |
| `afib_subset_features.parquet` | AF vs Norm processing features |
| `norm_metadata.parquet` | Abnormal vs Normal metadata |
| `norm_subset_metadata.parquet` | Abn vs Norm subset metadata |
| `norm_subset_features.parquet` | Abn vs Norm processing features |
