
# ECG Signal Processing vs Representation Learning on MIMIC-IV-ECG

This project investigates how **classical ECG signal processing features** compare with **pretrained representation learning models** (e.g., HuBERT-ECG) for ECG classification tasks using the **MIMIC-IV-ECG** dataset.

Two classification tasks are studied:

- **Atrial Fibrillation Detection:** Afib vs Normal ECG
- **Abnormal ECG Detection:** Abnormal vs Normal ECG

The goal is to evaluate whether modern pretrained ECG encoders outperform interpretable signal processing pipelines for clinical ECG analysis.

---

# Project Structure

```
├── data/                                 
│   ├── metadata/                       # Raw metadata CSV files from MIMIC-IV-ECG
│   ├── raw_waveforms/                  # Raw ecg waveforms downloaded from MIMIC
│   ├── processed/                      # Processed dataset saved as parquets
│   └── figures/                        # Figures from exploratory data analysis
│
├── notebooks/
│   └── extract_data.ipynb              # Notebook for development of data extraction and label creation
│
├── utils/
│   └── aria2_input.txt                 # aria2c input to download large amounts of ecg files from MIMIC-IV-ECG
│
├── src/
│   └── build_mimic_ecg_metadata.py     # Finalized data extraction pipeline
│
└── README.md
```

---

# Dataset

This project uses the **MIMIC-IV-ECG Matched Subset**, which contains:

- 12-lead ECG waveforms  
- 10-second recordings  
- 500 Hz sampling rate  
- machine generated ECG interpretation reports  

Access to the dataset requires credentialed access through **PhysioNet**.

---

# Label Construction

ECG labels are derived from the machine interpretation text provided in the MIMIC-IV-ECG metadata.

Three mutually exclusive cohorts are defined:

| Label | Description |
|------|-------------|
| **is_af** | ECG reports containing atrial fibrillation |
| **is_normal_strict** | ECG reports explicitly labeled normal |
| **is_clearly_abnormal** | ECG reports labeled abnormal (excluding AF) |

Ambiguous ECGs (e.g., borderline findings or nonspecific abnormalities) are excluded from classification tasks.

---

# Classification Tasks

Two binary classification tasks are constructed.

## Task 1: Atrial Fibrillation Detection

Atrial Fibrillation vs Normal ECG reports

## Task 2: Abnormal ECG Detection

Abnormal vs Normal ECG reports

---

# Subset Construction

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

---

# Metadata Generation Script

The script `src/build_mimic_ecg_metadata.py` constructs all metadata tables used in the project.

1. Load MIMIC-IV-ECG metadata files
2. Combine machine interpretation fields into a single report
3. Parse ECG labels using regex keyword matching
4. Construct the master metadata table (`META`)
5. Verify that cohorts are mutually exclusive
6. Build task specific datasets
7. Sample the balanced subset
8. Generate a waveform download manifest for aria2

---

# Running the Metadata Pipeline

From the project root directory:

```
python src/build_mimic_ecg_metadata.py
```

Optional flags:

```
--make-plots      Save cohort distribution figure
--write-aria2     Generate waveform download manifest
```

Example:

```
python src/build_mimic_ecg_metadata.py --make-plots --write-aria2
```

---

# Output Files

The script generates the following metadata files:

```
data/processed/meta_labels.parquet
data/processed/af_task_metadata.parquet
data/processed/norm_task_metadata.parquet
data/processed/af_task_subset.parquet
data/processed/norm_task_subset.parquet
```

These tables contain the ECG metadata, labels, and waveform paths used for downstream analysis.

---

# Downloading ECG Waveforms

Waveforms can be downloaded using the generated `aria2_input.txt` file.

Example command:

```
aria2c -i aria2_input.txt   -j 32   --enable-http-pipelining=true   --min-split-size=1M   --continue=true   --auto-file-renaming=false
```

Downloaded ECG waveforms should be stored in:

```
data/raw_waveforms/
```
