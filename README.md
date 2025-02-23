# ShickShack: Jailbreak Detection in Large Language Models (LLMs) 

A robust, multi-layered anomaly detection system designed to identify and prevent jailbreak attempts in Large Language Models (LLMs). This project combines heuristics-based pre-filtering, vector similarity analysis, and deep intent understanding to effectively detect both simple and sophisticated manipulation tactics.

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Methodology](#methodology)
  - [Heuristics-Based Pre-Filtering](#heuristics-based-pre-filtering)
  - [Vector Similarity Check (FAISS + SBERT)](#vector-similarity-check-faiss--sbert)
  - [Intent & Context Analysis (Mistral AI)](#intent--context-analysis-mistral-ai)
  - [Feedback Loop & Continuous Learning](#feedback-loop--continuous-learning)
- [Technology Stack](#technology-stack)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Future Prospects](#future-prospects)

## Overview

This project aims to secure Large Language Models (LLMs) against jailbreak attempts using a multi-layered approach. By integrating anomaly detection techniques with advanced NLP models, the system identifies potential jailbreaks through both structural and contextual analysis.

## Objectives

- Detect jailbreak attempts in LLMs with high accuracy.
- Implement a multi-layered detection system combining heuristics, vector similarity, and deep intent analysis.
- Develop a feedback loop for continuous learning and system improvement.
- Provide an efficient and scalable solution adaptable to evolving jailbreak tactics.

## Methodology

### Heuristics-Based Pre-Filtering

The first layer uses lightweight heuristics to pre-filter inputs based on easily identifiable characteristics such as:

- Token count and character length
- Markdown usage and obfuscation tactics
- Presence of emotionally charged or manipulative language

This step ensures computational efficiency by filtering out clearly benign inputs before deeper analysis.

### Vector Similarity Check (FAISS + SBERT)

Prompts that pass the heuristic filter are analyzed using FAISS and SBERT embeddings to check for semantic similarities with known jailbreak samples. This layer effectively detects paraphrased or subtly modified jailbreak attempts.

### Intent & Context Analysis (Mistral AI)

For complex cases, the system employs Mistral AI to perform deep semantic and contextual analysis. This layer uncovers hidden malicious intent and linguistic manipulation that simpler methods might miss.

### Feedback Loop & Continuous Learning

Detected jailbreaks trigger a feedback loop that updates:

- The dataset with new samples.
- The FAISS index to improve future detection.
- Training data analysis to identify evolving patterns over time.

## Technology Stack

- **Langchain** – LLM pipeline management
- **FAISS** – Vector similarity search
- **SBERT** – Sentence embeddings
- **Mistral AI** – Deep intent analysis
- **MongoDB** – Data storage (FAISS index, embeddings, and datasets)
- **Streamlit** – Interactive UI for system interaction
- **Python** – Core development language

## Exploratory Data Analysis (EDA)

EDA revealed key factors contributing to jailbreak detection:

- Increased token and character counts
- Frequent use of markdown and obfuscated content
- Emotionally charged language
- Critical keywords identified through TF-IDF and n-gram analysis

These findings guided the design of the heuristics and vector similarity layers.

## Future Prospects

- **Heuristics-First Approach**: Enhance efficiency by refining heuristics-based pre-filtering to minimize reliance on resource-heavy vector analysis.
