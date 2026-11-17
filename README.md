
# CT Dataset Quality Prediction Using Machine Learning

 
**Presenter:** Arjita Pathak  

## 1. Executive Summary

3D Computed Tomography (CT) scans are extensively used in medical diagnostics and industrial non-destructive testing. However, volumetric CT datasets can suffer from physical corruptions such as missing slices, intensity drift, photon starvation noise, and detector saturation clipping.

Manually inspecting every slice of a 3D CT volume is time-consuming, computationally expensive, and impractical for high-throughput applications.

This project develops an end-to-end automated **CT Dataset Quality Gate** (`src/gate.py`) using Machine Learning.

The pipeline:

- Generates synthetic 3D Shepp-Logan phantom CT volumes.
- Simulates different types and levels of scanner defects.
- Extracts 25 statistical features without using defect labels or injection recipes.
- Trains and evaluates multiple classification and regression models.
- Deploys a command-line quality assessment tool.

The final system predicts whether a CT scan should be classified as:

- **Accept:** The scan is suitable for use.
- **Caution:** The scan requires further inspection.
- **Reject:** The scan has significant quality issues.

It also generates a continuous quality score between 0 and 100 and provides a structured JSON report.

---

## 2. System Architecture & Execution Pipeline

The project follows a modular, sequential workflow consisting of eight major stages.

```text
┌──────────────────────────────────────────────┐
│ 1. Phantom Synthesis                         │
│    src/phantom_generator.py                  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 2. Defect Injection                          │
│    src/defect_injector.py                    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 3. Feature Extraction (25 Metrics)           │
│    src/feature_extractor.py                  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 4. Dataset Assembly & Preprocessing          │
│    src/dataset_builder.py                    │
└──────────────────────┬───────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
┌──────────────────────┐  ┌──────────────────────┐
│ 5. Classification    │  │ 6. Regression        │
│    Training          │  │    Training          │
│ train_classifier.py  │  │ train_regressor.py   │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│ 7. System Evaluation & Benchmarking          │
│    src/evaluate.py                           │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 8. Primary Testing Entry Point               │
│    src/gate.py                               │
└──────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```text
CT_Dataset_Quality_Prediction/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── Generated .npz CT volumes (600 scans)
│   │
│   └── processed/
│       └── Extracted 25-feature datasets (CSV)
│
├── models/
│   ├── xgb_classifier.joblib
│   └── xgb_regressor.joblib
│
└── src/
    ├── __init__.py
    ├── phantom_generator.py
    ├── defect_injector.py
    ├── feature_extractor.py
    ├── dataset_builder.py
    ├── train_classifier.py
    ├── train_regressor.py
    ├── evaluate.py
    └── gate.py
```

### Directory Description

| Directory/File | Purpose |
|---|---|
| `README.md` | Project documentation and execution guide |
| `requirements.txt` | Required Python dependencies |
| `data/raw/` | Stores generated 3D CT volumes in `.npz` format |
| `data/processed/` | Stores extracted features in CSV format |
| `models/` | Stores trained machine learning models |
| `src/` | Contains the core Python source code |

---

## 4. File-by-File Roles & Responsibilities

### 4.1 `src/phantom_generator.py`

**Role:** Synthetic 3D Data Synthesis

Uses the ASTRA Tomography Toolbox to create synthetic 3D Shepp-Logan phantom volumes.

- Sets up a 2D parallel-beam Radon projection geometry.
- Uses 180 angular projection steps.
- Reconstructs clean phantom slices using Filtered Backprojection (FBP).
- Generates volumes of dimensions `128 × 128 × 128`.

These clean volumes serve as the reference datasets for further processing.

### 4.2 `src/defect_injector.py`

**Role:** Corruption & Defect Engine

Introduces artificial defects into clean CT volumes to simulate different scanner-related quality problems.

Four defect types are implemented across seven severity levels:

1. **Missing Slices:** Removes selected blocks of slices by setting their values to zero, simulating slice acquisition failures.

2. **Intensity Drift:** Applies a linear brightness variation along the z-axis to simulate gradual intensity changes.

3. **Gaussian Noise:** Adds Gaussian noise to simulate noisy images associated with low-dose scanning and photon starvation.

4. **Dynamic Clipping:** Limits or saturates pixel intensities to simulate detector saturation and loss of intensity information.

The system also supports single and compound defect combinations.

### 4.3 `src/feature_extractor.py`

**Role:** Blind 25-Feature Statistical Engine

Extracts 25 statistical features from each CT volume.

The feature extraction process is blind to defect labels and injection recipes. It uses only the actual volume data.

The features are organized into five groups:

**1. Brightness Features**
- Mean intensity
- Standard deviation
- Dynamic range

**2. Noise Features**
- Local variance
- Median Absolute Deviation (MAD)

**3. Drift Detection Features**
- Linear regression slope
- Coefficient of determination (R²), represented by `drift_r2`

**4. Structural Integrity Features**
- Missing slice counts
- Zero-variance slice transitions

**5. Neighbour Similarity Features**
- Structural Similarity Index (SSIM) between adjacent slices
- Minimum adjacent SSIM (`adjacent_ssim_min`)
- Mean adjacent SSIM (`adjacent_ssim_mean`)

These features help the machine learning models identify patterns associated with different types of CT scan degradation.

### 4.4 `src/dataset_builder.py`

**Role:** Dataset Generation & Corpus Orchestration

- Generates a total of 600 synthetic CT volumes.
- Includes 100 clean reference volumes and 500 defective volumes.
- Creates three operational quality categories: Accept, Caution, and Reject.
- Generates continuous quality scores between 0 and 100.
- Coordinates feature extraction and dataset preparation.
- Saves the processed dataset as a CSV file.

**Output:** `data/processed/features.csv`

### 4.5 `src/train_classifier.py`

**Role:** Classification Model Training

Trains and evaluates four classification algorithms to predict the operational quality category of a CT scan.

Models evaluated:

- Logistic Regression Classifier
- Random Forest Classifier
- XGBoost Classifier
- MLP Neural Network Classifier

The selected classification model is saved as:

`models/xgb_classifier.joblib`

### 4.6 `src/train_regressor.py`

**Role:** Regression Model Training

Trains and evaluates three regression algorithms to predict the continuous CT scan quality score.

Models evaluated:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The selected regression model is saved as:

`models/xgb_regressor.joblib`

### 4.7 `src/evaluate.py`

**Role:** System Evaluation & Benchmarking

Evaluates the performance of the trained machine learning models.

Major responsibilities include:

- Comparing ML models against rule-based baselines.
- Evaluating classification and regression performance.
- Analysing feature importance.
- Testing model performance on unseen compound defect combinations (Out-of-Distribution Testing).

Important features identified include:

- `drift_r2`
- `adjacent_ssim_min`

### 4.8 `src/gate.py`

**Role:** Operational CLI Quality Gate & Main Testing Entry Point

This is the primary script used to test an individual CT scan.

It performs the following operations:

1. Loads the pre-trained classification and regression models.
2. Reads the input CT volume in `.npz` format.
3. Extracts the 25 statistical features.
4. Predicts the operational quality category.
5. Calculates the continuous quality score.
6. Displays a structured JSON report.

The script acts as the final interface between the trained machine learning models and the user.

---

## 5. How to Test Scans & Evaluate Models

The primary entry point for testing an individual CT scan is:

`src/gate.py`

### Quick Testing Command

Run the following command from the project root directory:

```bash
python src/gate.py --input data/raw/scan_001.npz
```

### What Happens During Testing?

When the command is executed:

1. The system loads the pre-trained models:
   - `models/xgb_classifier.joblib`
   - `models/xgb_regressor.joblib`

2. It extracts the 25 statistical features from the input CT scan.

3. The classification model predicts one of three operational grades:
   - Accept
   - Caution
   - Reject

4. The regression model predicts a continuous quality score between 0 and 100.

5. A structured JSON report is displayed in the terminal.

---

## 6. Models Evaluated & Quantitative Results

### 6.1 Classification Models

**Task:** Predict the CT scan quality category (Accept / Caution / Reject).

| Classification Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Rule-Based Baseline | 44.05% | 0.41 | 0.44 | 0.40 |
| Logistic Regression | 97.62% | 0.97 | 0.97 | 0.97 |
| Random Forest | 100.00% | 1.00 | 1.00 | 1.00 |
| MLP Neural Network | 100.00% | 1.00 | 1.00 | 1.00 |
| XGBoost (Selected Engine) | 98.81% | 0.99 | 0.99 | 0.99 |

### 6.2 Regression Models

**Task:** Predict a continuous CT scan quality score between 0 and 100.

**Course Requirement:** Mean Absolute Error (MAE) ≤ 6.0 points.

| Regression Model | MAE (Points) | RMSE (Points) | R² Score |
|---|---:|---:|---:|
| Rule-Based Threshold Baseline | 12.40 | 16.85 | 0.42 |
| Linear Regression | 2.15 | 3.10 | 0.94 |
| MLP Neural Network | 0.81 | 1.24 | 0.98 |
| Random Forest | 0.64 | 1.02 | 0.99 |
| XGBoost (Selected Engine) | 0.58 | 0.91 | 0.99 |

The selected XGBoost regressor achieves an MAE of **0.58 points**, which is within the course requirement of 6.0 points.

*Note: The reported results are based on the synthetic dataset and evaluation setup used in this project. They should not be interpreted as guaranteed performance on real-world clinical CT scans.*

---

## 7. Full Pipeline Re-Building Guide

The complete dataset generation and model training pipeline can be executed from scratch using the following steps.

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate the 600-Scan Synthetic Dataset

```bash
python src/dataset_builder.py --num_scans 600 --output_dir data/
```

This generates the synthetic CT volumes and extracts their statistical features.

### Step 3: Train the Classification and Regression Models

**Classification:**

```bash
python src/train_classifier.py --data data/processed/features.csv
```

**Regression:**

```bash
python src/train_regressor.py --data data/processed/features.csv
```

The trained models are saved in the `models/` directory.

### Step 4: Run System Evaluation

```bash
python src/evaluate.py --data data/processed/features.csv
```

This evaluates the models and performs benchmarking.

### Step 5: Test an Individual CT Scan

```bash
python src/gate.py --input data/raw/scan_001.npz
```

---

## 8. Operational CLI Output Example

An example of the JSON report generated by `src/gate.py` is shown below:

```json
{
  "scan_id": "scan_001.npz",
  "status": "Accept",
  "operational_code": 0,
  "quality_score": 98.42,
  "metrics": {
    "drift_r2": 0.012,
    "adjacent_ssim_min": 0.984,
    "mean_intensity": 0.421
  }
}
```

### Understanding the Output

| Field | Description |
|---|---|
| `scan_id` | Name of the tested CT scan |
| `status` | Predicted operational category |
| `operational_code` | Numeric code corresponding to the predicted category |
| `quality_score` | Predicted quality score between 0 and 100 |
| `metrics` | Selected statistical features extracted from the CT volume |

---

## 9. Limitations & Future Scope

Although the developed framework demonstrates an automated approach to CT dataset quality assessment, certain limitations remain.

### Limitations

- The current dataset is synthetically generated using Shepp-Logan phantoms.
- Synthetic defects cannot fully represent every physical scanner artifact.
- Real-world CT scans may contain additional problems such as beam hardening, metal artifacts, detector imperfections, and patient motion.
- Model performance on synthetic data does not guarantee equivalent performance on real clinical datasets.

### Future Scope

- Validate the framework using real medical and industrial CT datasets.
- Include additional physical scanner artifacts.
- Improve model generalization across different scanners and acquisition protocols.
- Fine-tune the models using real-world CT data.
- Extend the framework to support larger datasets and practical deployment environments.

---

