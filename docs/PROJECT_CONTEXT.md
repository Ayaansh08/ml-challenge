# Project Context

## Problem Statement
Amazon ML Challenge 2026: Business Entity Resolution. The objective is to resolve reference business entities from Source 1 (S1) against noisy records in Source 2 (S2) and Source 3 (S3).

## Data Sources
- **Source 1 (S1):** Reference dataset (clean).
- **Source 2 (S2):** Noisy dataset.
- **Source 3 (S3):** Noisy dataset.

## Primary Metric
- **Macro-averaged F0.5 per S1 entity.** (We prioritize precision over recall).

## Hard Constraints
- **No external lookups/APIs:** Models and algorithms must run completely offline.
- **Open-set Country Field:** Test data contains unseen countries (e.g., France is unseen in training).
- **Model Size:** Final model parameters must be <= 8B.
- **License Compliance:** Dependencies and components must satisfy open-source fair-play requirements (e.g., MIT, Apache 2.0).
- **Submission Limit:** Maximum 5 submissions per day.
- **Compute/Time:** Limited local budget; pipeline must be memory-safe and resumable.

## Current Objective
Prioritize establishing a fast, measurable end-to-end working pipeline over rewriting components from scratch. Get to a first valid baseline F0.5 score as fast as possible.

## Terminology
- **S1:** Source 1 (Reference)
- **S2 / S3:** Sources 2 & 3 (Noisy)
- **Blocking:** Fast candidate pair generation (filtering millions of pairs to thousands).
- **Bi-Encoder:** Dense retrieval model producing candidate embeddings.
- **Cross-Encoder:** Heavy pair-wise classification model for final scoring.
