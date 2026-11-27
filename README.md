# LLM-Based Time-Series Data Quality Assessment

![Status](https://img.shields.io/badge/Status-Ongoing-orange)
![Project](https://img.shields.io/badge/Project-Master's%20Thesis-blue)
![Research](https://img.shields.io/badge/Research-Data--Centric%20AI-purple)

## Overview

Data quality is a fundamental requirement for reliable time-series analytics. However, real-world sensor and telemetry data frequently suffer from missing values, noise, outliers, communication failures, and temporal or concept drift.

This research project investigates the capabilities of Large Language Models (LLMs) in detecting, localizing, quantifying, and interpreting data quality defects in time-series data.

The main objective is to develop a systematic and reproducible framework for benchmarking LLM-based approaches to time-series data quality assessment, with a potential extension toward targeted data quality mitigation.

## Research Objectives

### 1. Benchmark Curation and Controlled Defect Injection

- Review existing approaches to LLM-based time-series analytics and data quality assessment.
- Select and preprocess representative real-world time-series datasets.
- Develop controlled data quality defect injection pipelines.
- Characterize and quantify different types and severities of data quality defects.

### 2. LLM-Based Quality Detection and Rating

- Investigate the capabilities of open-source LLMs in identifying and localizing data quality defects.
- Evaluate their ability to estimate defect severity.
- Generate structured and interpretable data quality ratings.
- Assess diagnostic reliability and hallucination rates.
- Compare LLM-based approaches against traditional baseline methods.

### 3. Data Quality Mitigation (Optional Extension)

- Investigate targeted correction and imputation strategies.
- Explore the mitigation of selected data quality defects.
- Evaluate improvements in data quality after correction.

## Proposed Framework

The proposed methodology consists of the following stages:

1. Real-world time-series dataset collection and preprocessing.
2. Controlled defect injection and ground-truth generation.
3. LLM-based data quality assessment.
4. Defect detection, localization, and severity estimation.
5. Structured quality rating and diagnostic interpretation.
6. Performance evaluation and comparison against baseline methods.
7. Targeted quality mitigation (optional).

## Data Quality Dimensions

| Defect | Description |
|---|---|
| Missing Values | Missing observations within a time series. |
| Sensor Noise | Unwanted perturbations affecting recorded measurements. |
| Outliers | Abnormal observations deviating from expected patterns. |
| Communication Failures | Data disruptions caused by transmission or connectivity issues. |
| Temporal Drift | Changes in the temporal characteristics of the data. |
| Concept Drift | Changes in the underlying data-generating relationships. |

## Evaluation

The proposed framework will be evaluated across several dimensions:

- Defect detection accuracy.
- Anomaly localization performance.
- Defect severity estimation.
- Data quality rating accuracy.
- Hallucination rates.
- Semantic diagnostic reliability.
- Comparative performance against traditional methods.

## Research Roadmap

| Phase | Period | Focus |
|---|---|---|
| Phase 1 | October – November | Literature review, dataset curation, and baseline preparation. |
| Phase 2 | December – February | Defect injection, LLM evaluation, and framework development. |
| Phase 3 | March – April | Optional mitigation, comparative experiments, and framework finalization. |
| Phase 4 | Late April – Late May | Research manuscript, thesis writing, and reproducibility. |

The roadmap remains flexible and may evolve according to experimental findings and research progress.

## Expected Contributions

- A systematic benchmark for LLM-based time-series data quality assessment.
- Reproducible data quality defect generation pipelines.
- A structured framework for defect detection, localization, and quality rating.
- An empirical investigation of LLM capabilities and limitations in time-series data quality assessment.
- A potential targeted data quality mitigation approach.
- A documented open-source research artifact.

## Related Work

- **TSRating:** Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment.
- **TSQAgent:** Rating Time Series Data Quality via Dedicated Agentic Reasoning.
- **Delving into Large Language Models for Effective Time-Series Anomaly Detection.**

## Supervision

**Co-supervisor:** Dr. Mohamed Reda Bouadjenek  
Senior Lecturer in Applied Artificial Intelligence  
Deakin University, Australia

## Project Status

This repository accompanies an ongoing Master's thesis research project. The implementation, experimental results, datasets, and documentation will be progressively updated throughout the research.

## Keywords

Large Language Models · Time-Series Analytics · Data Quality Assessment · Anomaly Detection · Data-Centric AI · Time-Series Imputation · Edge Sensor Telemetry
