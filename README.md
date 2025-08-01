# Barbell Exercise Tracking & Classification

This project implements a complete machine learning pipeline for sensor-based tracking, feature engineering, and classification of barbell exercises using wearable device data. The workflow covers data ingestion, cleaning, feature extraction, model training, and evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Pipeline](#data-pipeline)
  - [1. Data Ingestion & Preprocessing](#1-data-ingestion--preprocessing)
  - [2. Outlier Detection & Removal](#2-outlier-detection--removal)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training & Evaluation](#4-model-training--evaluation)
  - [5. Repetition Counting](#5-repetition-counting)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Results & Insights](#results--insights)

---

## Project Overview

The goal is to automatically classify barbell exercises (e.g., squat, bench, deadlift) and count repetitions using time-series data from accelerometer and gyroscope sensors. The pipeline includes robust data cleaning, advanced feature extraction (temporal, frequency, PCA, clustering), and model selection for accurate activity recognition.

---

## Data Pipeline

### 1. Data Ingestion & Preprocessing (`make_dataset.py`)

- Loads raw CSV files from wearable sensors (MetaMotion) for multiple participants and exercise sessions.
- Extracts metadata (participant, label, category, set) from filenames.
- Merges accelerometer and gyroscope data into a unified dataframe.
- Resamples data to a consistent frequency (200ms) and splits by day.
- Exports processed data for further analysis.

### 2. Outlier Detection & Removal (`remove_outliers.py`)

- Visualizes sensor data distributions and boxplots by exercise label.
- Implements three outlier detection methods:
  - Interquartile Range (IQR)
  - Chauvenet's Criterion (statistical)
  - Local Outlier Factor (LOF, distance-based)
- Removes outliers by replacing them with NaN, grouped by exercise label.
- Exports cleaned dataset for feature engineering.

### 3. Feature Engineering (`build_features.py`)

- Handles missing values via interpolation.
- Calculates set durations and aggregates statistics.
- Applies Butterworth lowpass filter to smooth sensor signals.
- Performs Principal Component Analysis (PCA) for dimensionality reduction.
- Generates derived features (sum of squares, temporal rolling means/stds).
- Extracts frequency-domain features using Fourier Transform.
- Reduces redundancy by handling overlapping windows.
- Applies clustering (KMeans) to group similar movement patterns.
- Exports final feature-rich dataset for modeling.

### 4. Model Training & Evaluation (`train_model.py`)

- Splits data into training and test sets (random and participant-based).
- Defines multiple feature sets (basic, engineered, PCA, temporal, frequency, cluster, selected).
- Performs forward feature selection using decision trees.
- Trains and compares classifiers:
  - Neural Network (MLP)
  - Random Forest
  - K-Nearest Neighbors
  - Decision Tree
  - Naive Bayes
- Uses grid search and cross-validation for hyperparameter tuning.
- Visualizes model performance with grouped bar plots and confusion matrices.
- Evaluates generalization to unseen participants.

### 5. Repetition Counting (`count_repetitions.py`)

- Calculates vector magnitudes for accelerometer and gyroscope data.
- Visualizes sensor signals for each exercise type.
- Applies lowpass filtering and peak detection to count repetitions.
- Benchmarks predicted reps against ground truth and computes error metrics.

---

## Technologies Used

- Python 3
- Pandas, NumPy
- Scikit-learn (ML, outlier detection, clustering)
- Matplotlib, Seaborn (visualization)
- SciPy (signal processing)
- Jupyter Notebooks (recommended for exploration)

---

## How to Run

1. Place raw sensor CSV files in `data/raw/MetaMotion/`.
2. Run scripts in order:
   - `make_dataset.py` → `remove_outliers.py` → `build_features.py` → `train_model.py` → `count_repetitions.py`
3. Inspect outputs in `data/interim/` and generated plots for analysis.

---

## Project Structure

```
data/
  raw/           # Raw sensor CSV files
  interim/       # Processed, cleaned, and feature datasets
src/
  data/
    make_dataset.py
  features/
    remove_outliers.py
    build_features.py
    count_repetitions.py
  models/
    train_model.py
    LearningAlgorithms.py
```

---

## Results & Insights

- Robust preprocessing and outlier removal significantly improve data quality.
- Advanced feature engineering (temporal, frequency, PCA, clustering) boosts model accuracy.
- Random Forest and Neural Network models achieve high classification performance.
- Participant-based evaluation demonstrates model generalization to new users.
- Automated repetition counting provides reliable exercise metrics for training analysis.

---
