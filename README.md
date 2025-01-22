# Automatic Evaluation of Dialogues Using Large Language Models (LLMs)

## Project Overview

This repository contains the implementation of a study inspired by the framework presented in the paper: A Comprehensive Analysis of the Effectiveness of Large Language Models as Automatic Dialogue Evaluators. The goal is to explore and implement a multi-dimensional system for automatic dialogue evaluation using LLMs, analyzing their robustness and effectiveness across various dimensions.

## Objectives

- Understand the Framework: Study the evaluation metrics at both turn-level and dialogue-level;
- Implement overall evaluation: utilize LLMs (e.g., GPT-4, LLaMA-2) to evaluate dialogues;
- Test the system on provided datasets and compare LLM scores to human judgments;

## Methodology

### 1. Framework Analysis
- Study the metric for evaluating the overall quality of dialogues;
- Analyze the effectiveness of LLMs in producing scores aligned with human judgments;
- Investigate the impact of adversarial perturbations to test robustness in evaluating overall quality.

### 2. Environment Setup
- Configure a Python environment with essential libraries:
  - PyTorch
  - Transformers
  - NumPy
- Use the ChatGPT API for experiments requiring external access to GPT4-based models;
- Execute five open-source models locally for testing and evaluation, ensuring reproducibility and flexibility.

### 3. Evaluation
- Conduct experiments using provided datasets.
- Compute correlations between LLM scores and human judgments using statistical metrics:
  - Cohen's Kappa
  - Spearman
  - Pearson
  - Kendall-Tau



