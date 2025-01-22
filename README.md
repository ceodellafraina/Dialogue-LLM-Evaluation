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

### 4. Results Analysis
- Implement a system to evaluate open-domain dialogues (turn-level and dialogue-level) based on their overall quality;
- Analyze the correlations between LLM scores and human judgments focused exclusively on overall dialogue quality;
- Evaluate the effectiveness of ensemble techniques in improving alignment with human evaluations for the general quality metric.

## Results

The project delivers:

- An evaluation system for open-domain dialogues;
- Insights into LLM evaluation effectiveness compared to human judgments.

## Resources

- Base Paper: [A Comprehensive Analysis of the Effectiveness of Large Language Models as Automatic Dialogue Evaluators](https://ojs.aaai.org/index.php/AAAI/article/view/29923/31613)
  
- Models tested:
  - [Chatgpt-4](https://platform.openai.com/docs/api-reference/introduction)
  - [Chatglm3-6b-base](https://huggingface.co/THUDM/chatglm3-6b-base)
  - [Llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
  - [Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)
  - [vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)
  - [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)


## How to Use

1. Clone this repository;
2. Set up the Python environment:
```bash
pip install -r requirements.txt
```
3. Run evaluation script in "scoring" folders with the provided datasets.
