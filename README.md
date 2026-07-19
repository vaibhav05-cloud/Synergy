#  DeploySense AI

**AI-Powered Pre-Deployment Risk Intelligence Platform**

> Predict. Explain. Prevent.

DeploySense AI looks at a deployment before it ships and tells engineers three things: how risky it is, *why* it's risky, and what to do to make it safer — all before the code goes live.

Built for **Synergy 2026** (Manipal University Jaipur × HPE) — HPE Problem Statement 09: *Deployment Risk Scorer*.

---

## The Problem

Engineering teams deploy code many times a day. Whether a given deployment is risky is usually decided by gut feeling, not data. One bad deployment at the wrong time — a big change, off-hours, on a critical service, with a history of past incidents — can cause a real outage.

**The challenge:** train a classifier on deployment metadata that scores every deployment **Low / Medium / High** risk before it happens, and explain that score in a way an engineer can actually trust and act on.

## What Makes This Different

DeploySense AI does **not** replace your CI/CD pipeline. Jenkins and GitHub Actions keep doing exactly what they already do — build, test, deploy. DeploySense adds one intelligent checkpoint in between:

```
Code Push → Build → Test → [ DeploySense AI Gate ] → Deploy
```

| | Jenkins / GitHub Actions | DeploySense AI |
|---|---|---|
| **Behaves like** | A security guard — follows a fixed checklist | An experienced detective — learns from history |
| Memory of past deployments | None | Learns from every past deployment & incident |
| Explains *why* something is risky | No | Yes, via SHAP |
| Tells you how to reduce the risk | No | Yes, via AI-generated recommendations |

Compared to enterprise AIOps tools (Harness, Dynatrace, Datadog): those watch production *after* code is live and need a full observability stack. DeploySense works entirely *before* deployment, straight off Git/PR metadata — lightweight, and explainable by design.

## Main Flow

```
Developer wants to deploy
        ↓
GitHub data fetch (commits, PRs, workflow, deployment metadata)
        ↓
ML model predicts risk (Low / Medium / High + %)
        ↓
SHAP explains WHY
        ↓
AI suggests HOW to reduce the risk
        ↓
Engineer modifies the deployment
        ↓
Risk decreases → Deploy confidently
```

## Features

### Core (must-have)
- 🎯 **Deployment Risk Prediction** — Low / Medium / High + risk %, from repo, branch, files changed, commits, deployment time, service, review count
- 🔍 **SHAP Explainability** — every prediction comes with the specific factors that drove it (e.g. *+ night deployment, + large PR, + previous failures, + payment-critical service*)
- 🔗 **GitHub Integration** — commits, workflow runs, PRs and deployment metadata are fetched automatically via the GitHub REST API, no manual CSV upload
- 📊 **Deployment Dashboard** — risk gauge, risk badge, SHAP chart, deployment summary, all in one screen

### AI-powered differentiators
- 🤖 **AI Recommendation Engine** — turns SHAP's top risk factors into plain-English actions ("wait until business hours", "split this PR", "add a reviewer")
- 🎚️ **What-if Simulator** — drag deployment time or file count and watch the risk score update live, using the same prediction API with different inputs

### Dashboard extras
- ✅ **Smart Deployment Checklist** — auto-generated from the same feature data (tests passed, CI success, reviewer approved, weekend deployment, etc.)
- 💯 **Deployment Health Score** — the risk score expressed as a single 0–100 number

## Architecture & Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Data source | GitHub REST API | Deployment metadata, commit/PR history |
| ML model | Random Forest / XGBoost | Predicts Low / Medium / High risk |
| Explainability | SHAP (TreeExplainer) | Explains why a prediction was made |
| Recommendations | LLM prompted with SHAP output | Turns risk factors into action items |
| Backend | FastAPI *(planned)* | Serves the `/predict` endpoint |
| Frontend | React *(planned)* | Dashboard, charts, what-if simulator |

## Project Structure

```
deploysense-ai/
├── data/
│   └── processed/
│       ├── train.csv          # 1,036 rows
│       ├── validation.csv     # 223 rows
│       └── test.csv           # 223 rows
├── model/
│   ├── preprocessing.py       # feature engineering + encoding pipeline
│   ├── train_model.py         # trains & evaluates Random Forest / XGBoost
│   ├── evaluate_report.py     # generates the full HTML evaluation report
│   ├── requirements.txt
│   └── saved_model/           # trained model, encoders, charts, report (generated)
└── backend/                   # FastAPI service (in progress)
```

## Current Results

The model is trained and evaluated on a clean, leakage-free dataset (46 engineered features, no missing values, no train/val/test overlap).

| Metric | Score |
|---|---|
| Test Accuracy | ~75% |
| Test Macro-F1 | 0.76 |
| High-risk precision | 0.85 |
| High-risk recall | 0.76 |

**Safety highlight:** in the confusion matrix, **zero High-risk deployments were ever predicted as Low-risk** — the model's mistakes are conservative (confusing High with Medium), never reckless.

Medium-risk is the hardest class to call — which mirrors how engineers themselves often disagree on borderline changes.

## Getting Started

```bash
# 1. Clone the repo
git clone <this-repo-url>
cd deploysense-ai

# 2. Install dependencies
cd model
pip install -r requirements.txt

# 3. Train the model and generate the evaluation report
python train_model.py
```

This produces `model/saved_model/model_pipeline.pkl` (the trained model) and `model/saved_model/training_report.html` — open the report in a browser to see every chart: class distribution, model comparison, confusion matrix, ROC curves, feature importance, and SHAP plots.

## Roadmap

**Phase 1 — Essential** ✅ *in progress*
- [x] Dataset prepared & validated
- [x] Feature engineering
- [x] Random Forest / XGBoost model trained & evaluated
- [x] SHAP explainability
- [ ] FastAPI backend (`/predict` endpoint)
- [ ] GitHub API integration
- [ ] React dashboard

**Phase 2 — Differentiators**
- [ ] AI-generated deployment recommendations
- [ ] Interactive What-if Simulator
- [ ] Deployment Health Score
- [ ] Smart Deployment Checklist

## About

Built by **Bucks and bugs 2026** for the Synergy 2026 hackathon at Manipal University Jaipur, in collaboration with HPE, E-Cell MUJ, and the Directorate of Corporate Relations & Placements.