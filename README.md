# Anomaly Detection and Data Quality Correction for Transformer Load Time Series

**Topic:** Electricity Demand Forecasting  
**Credit Load:** 6 ECTS (150 hours)

## Background and Motivation
Municipal electricity distribution grids consist of hundreds of transformer (trafo) stations that continuously record load measurements. In practice, these recordings are frequently affected by sensor malfunctions, communication dropouts, meter resets, and physically implausible spikes or flatlines. If left uncorrected, such anomalies directly degrade the accuracy of any forecasting model trained on the data. Before advanced forecasting techniques can be meaningfully applied to real-world trafo data, a robust and reproducible data quality pipeline must be established.
Your work addresses this foundational challenge. You will investigate whether state-of-the-art deep learning models for **anomaly detection** and **imputation** — as implemented in the Time Series Library (TSLib) — can be used to automatically identify and correct faulty measurements in transformer load time series, and whether doing so leads to measurably better forecasting outcomes.


## The Time Series Library (TSLib)
TSLib is an open-source deep learning library developed at Tsinghua University, designed as a unified benchmark and development platform for time series analysis. It supports five core tasks under a single codebase:
* **Long-term forecasting**
* **Short-term forecasting**
* **Imputation** (reconstruction of missing or masked values)
* **Anomaly detection**
* **Classification**
All tasks share a common entry point (`run.py`) and are configured via command-line arguments or shell scripts. The library includes a wide range of model architectures, from simple linear baselines (DLinear) to advanced Transformer-based models (iTransformer, TimesNet, FEDformer, Autoformer, Non-stationary Transformer). You are expected to become familiar with the library's structure, including its data loading pipeline (`data_provider/`), experiment classes (`exp/`), and evaluation utilities (`utils/metrics.py`).

## Dataset
You will be provided with a **synthetic dataset** that simulates transformer station load measurements alongside co-located **weather data** (e.g., temperature, solar irradiance, cloud cover). The dataset has been designed to reflect realistic characteristics of trafo load profiles from a municipal grid operator, including injected anomalies of different types (spikes, flatlines, missing intervals). The exact format and variable descriptions will be provided separately.

## Your Task
Your work is structured in three connected phases:

**Phase 1 — Anomaly Detection**  
Using TSLib's anomaly detection task, you will benchmark at least **three models** from the library (suggested starting points: TimesNet, FEDformer, Autoformer) on the provided dataset. Your goal is to evaluate how well each model flags anomalous intervals in the load time series. You will experiment with the `--anomaly_ratio` parameter to understand how the models' sensitivity affects detection performance, and you will document the precision, recall, and F1 score for each configuration.

**Phase 2 — Imputation (Value Correction)**  
Once anomalies have been flagged, the next step is to replace the identified faulty values with plausible reconstructed ones. You will use TSLib's imputation task to reconstruct the flagged intervals, benchmarking at least **two imputation models** (suggested: Non-stationary Transformer, TimesNet). You will evaluate reconstruction quality using MSE and MAE against the known ground-truth values in the synthetic dataset.

**Phase 3 — Impact Assessment on Forecasting**  
To demonstrate the practical value of your data correction pipeline, you will train a short-term forecasting model (e.g., DLinear or TimesNet) on two versions of the dataset: the raw (uncorrected) data and the corrected (post-imputation) data. By comparing forecasting accuracy (MSE, MAE) across both versions, you will quantify the downstream benefit of the anomaly detection and correction pipeline.

## Deliverables
You are expected to submit the following:
1. **Written Report** — A structured scientific report (~4,000–6,000 words) covering:
    * Introduction and motivation
    * Related work (brief literature review on anomaly detection in energy time series)
    * Description of the dataset and preprocessing steps
    * Methodology: models used, experimental configurations, evaluation metrics
    * Results: tables and figures comparing model performance across all three phases
    * Discussion: interpretation of results, limitations, and practical implications
    * Conclusion
2. **Code Repository** — A clean, documented version of all scripts used for your experiments, structured so that results can be reproduced. This should be based on the TSLib framework and submitted alongside the report.
3. **Presentation** — A 15–20 minute oral presentation of your findings, including a live or recorded demo of the TSLib pipeline running on the provided dataset.

## Recommended Starting Points
* Work through the TSLib tutorial notebook (`tutorial/TimesNet_tutorial.ipynb`) before starting experiments
* Study the `exp/exp_anomaly_detection.py` and `exp/exp_imputation.py` files to understand the task pipelines
* Review the example scripts under `scripts/anomaly_detection/` and `scripts/imputation/` to understand how experiments are configured
* Familiarize yourself with the evaluation metrics in `utils/metrics.py`
