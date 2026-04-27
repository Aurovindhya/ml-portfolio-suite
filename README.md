# ml-portfolio-suite

> A full-stack Machine Learning platform spanning classical ML, deep learning, Generative AI, computer vision, and agentic workflows — unified behind a single FastAPI orchestration layer.

This is not a collection of notebooks. It is a **production-structured ML system** built to demonstrate end-to-end engineering across every major discipline of modern applied AI.

---

## Platform Architecture

```
                        ┌─────────────────────────────────┐
                        │      FastAPI Gateway Layer       │
                        │   /api/v1/{module}/{endpoint}    │
                        └────────────┬────────────────────┘
                                     │
          ┌──────────────────────────┼───────────────────────────┐
          │                          │                           │
          ▼                          ▼                           ▼
   ┌─────────────┐          ┌──────────────────┐       ┌─────────────────┐
   │  Regression │          │  Classification  │       │  Generative AI  │
   │  ─────────  │          │  & Clustering    │       │  ─────────────  │
   │  Flight fare│          │  ────────────── │       │  Gemini chatbot │
   │  House price│          │  Chest disease   │       │  PDF assistant  │
   │  Gold price │          │  Heart disease   │       │  Medical tool   │
   └─────────────┘          │  Spam filter     │       └────────┬────────┘
                            └──────────────────┘                │
                                                                 │ Langfuse
          ┌──────────────────────────────────────────────────────┤
          │                                                       │
          ▼                                                       ▼
   ┌─────────────────┐                                  ┌──────────────────┐
   │ Computer Vision │                                  │ Agentic Workflow │
   │ ─────────────── │                                  │ ──────────────── │
   │ Hand tracking   │                                  │ Market insights  │
   │ (OpenCV + MP)   │                                  │ (multi-agent)    │
   └─────────────────┘                                  └──────────────────┘
```

Every module is independently importable and testable. The gateway composes them.

---

## Modules

### Regression
**Predicting continuous outcomes with classical and ensemble ML**

| Module | Model | Dataset | Key Metric |
|--------|-------|---------|------------|
| [Flight Fare](modules/regression/flight_fare/) | XGBoost + feature engineering | EaseMyTrip (~300k flights) | MAE ~₹1,200 |
| [House Prices](modules/regression/house_prices/) | Ridge + Lasso + ElasticNet ensemble | Boston / Kaggle Housing | RMSE ~0.12 (log scale) |
| [Gold Prices](modules/regression/gold_prices/) | Random Forest + time-series features | 10 years of daily OHLC | R² ~0.98 |

Techniques: polynomial features, target encoding, log transforms, cross-validated hyperparameter tuning, SHAP explainability.

---

### Classification & Clustering
**Binary and multi-class prediction with interpretability**

| Module | Model | Dataset | Key Metric |
|--------|-------|---------|------------|
| [Chest Disease Detection](modules/classification/chest_disease/) | CNN (EfficientNet-B0) + Grad-CAM | NIH Chest X-Ray | AUC ~0.87 |
| [Heart Disease](modules/classification/heart_disease/) | LightGBM + SHAP | UCI Heart Disease | F1 ~0.84 |
| [Spam Filter](modules/classification/spam_filter/) | TF-IDF + Logistic Regression / Naive Bayes | SpamAssassin + SMS | Precision ~0.99 |

Techniques: SMOTE for class imbalance, Grad-CAM visualization, SHAP waterfall plots, threshold tuning for recall vs. precision tradeoff.

---

### Generative AI
**LLM-powered tools built on Gemini Pro — all traced with Langfuse**

| Module | Description | Stack |
|--------|-------------|-------|
| [Chatbot](modules/generative/chatbot/) | Multi-turn conversational agent with memory and system prompt control | Gemini Pro + LangChain |
| [PDF Assistant](modules/generative/pdf_assistant/) | Upload any PDF, ask questions — RAG pipeline with semantic chunking | Gemini Pro + FAISS + LangChain |
| [Medical Tool](modules/generative/medical_tool/) | Symptom checker with structured output, guardrails, and disclaimer injection | Gemini Pro + Pydantic |

All GenAI modules emit traces to Langfuse: input tokens, output tokens, latency, and structured eval scores.

---

### Computer Vision
**Real-time video processing with OpenCV and MediaPipe**

| Module | Description | Stack |
|--------|-------------|-------|
| [Hand Tracking](modules/computer_vision/hand_tracking/) | 21-landmark hand detection at 30fps, gesture classification, finger counting | OpenCV + MediaPipe |

Supports webcam feed and static image input. Outputs annotated frames and landmark JSON.

---

### Agentic Workflows
**Multi-step autonomous pipelines with tool use — traced end-to-end**

| Module | Description | Stack |
|--------|-------------|-------|
| [Market Insights](modules/agentic/market_insights/) | Autonomous agent that fetches financial news, runs sentiment analysis, cross-references price data, and produces a structured market brief | LangChain Agents + Gemini Pro + yfinance + Langfuse |

The agent uses tool-calling to: search news → extract entities → query stock data → reason over signals → format output.

---

## Project Structure

```
ml-portfolio-suite/
├── gateway/
│   └── main.py                  # FastAPI gateway — routes all modules
├── core/
│   ├── config.py                # Centralised settings (env vars, paths)
│   └── response.py              # Shared response schemas
├── modules/
│   ├── regression/
│   │   ├── flight_fare/
│   │   │   ├── model.py         # Training + inference
│   │   │   ├── features.py      # Feature engineering pipeline
│   │   │   └── router.py        # FastAPI router
│   │   ├── house_prices/
│   │   └── gold_prices/
│   ├── classification/
│   │   ├── chest_disease/
│   │   ├── heart_disease/
│   │   └── spam_filter/
│   ├── generative/
│   │   ├── chatbot/
│   │   ├── pdf_assistant/
│   │   └── medical_tool/
│   ├── computer_vision/
│   │   └── hand_tracking/
│   └── agentic/
│       └── market_insights/
├── observability/
│   └── langfuse_client.py       # Shared Langfuse tracer (GenAI + agentic)
├── notebooks/                   # Colab notebooks for each module
├── scripts/
│   └── download_datasets.py     # One script to fetch all datasets
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/Aurovindhya/ml-portfolio-suite
cd ml-portfolio-suite
pip install -r requirements.txt
cp .env.example .env          # Add your API keys
uvicorn gateway.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) — Swagger UI lists every endpoint across all modules.

---

## API Overview

| Method | Path | Module |
|--------|------|--------|
| `POST` | `/api/v1/regression/flight-fare/predict` | Flight fare |
| `POST` | `/api/v1/regression/house-price/predict` | House prices |
| `POST` | `/api/v1/regression/gold-price/predict` | Gold prices |
| `POST` | `/api/v1/classification/chest-disease/predict` | Chest X-ray |
| `POST` | `/api/v1/classification/heart-disease/predict` | Heart disease |
| `POST` | `/api/v1/classification/spam/predict` | Spam filter |
| `POST` | `/api/v1/gen/chat` | Gemini chatbot |
| `POST` | `/api/v1/gen/pdf-qa` | PDF assistant |
| `POST` | `/api/v1/gen/medical` | Medical tool |
| `POST` | `/api/v1/vision/hand-tracking/analyze` | Hand tracking |
| `POST` | `/api/v1/agentic/market-insights` | Market insights |
| `GET`  | `/api/v1/health` | Platform health |

---

## Observability

Generative AI and agentic modules emit structured traces to [Langfuse](https://langfuse.com):

- **Input / output** logged per call
- **Token counts** and **latency** tracked per model call
- **Eval scores** attached for medical and market modules (relevance, groundedness)
- Traces degrade gracefully to no-op if `LANGFUSE_PUBLIC_KEY` is not set

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Classical ML** | scikit-learn, XGBoost, LightGBM, Random Forest |
| **Deep Learning** | PyTorch, EfficientNet (timm), Grad-CAM |
| **Generative AI** | Google Gemini Pro, LangChain, FAISS |
| **Computer Vision** | OpenCV, MediaPipe |
| **Agentic** | LangChain Agents, tool-calling, yfinance |
| **API Layer** | FastAPI, Pydantic v2, Uvicorn |
| **Observability** | Langfuse |
| **Explainability** | SHAP, Grad-CAM |
| **Training** | Google Colab (GPU) |

---

## Notebooks

Each module has a standalone Colab notebook covering data loading, EDA, training, evaluation, and inference.

| # | Notebook | Module | Key concepts |
|---|----------|--------|--------------|
| 01 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aurovindhya/ml-portfolio-suite/blob/main/notebooks/01_FlightFare_Colab.ipynb) | Flight Fare | XGBoost, feature engineering, SHAP importance |
| 02 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aurovindhya/ml-portfolio-suite/blob/main/notebooks/02_HousePrices_Colab.ipynb) | House Prices | Stacked ensemble, OOF predictions, log-transform |
| 03 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aurovindhya/ml-portfolio-suite/blob/main/notebooks/03_GoldPrices_Colab.ipynb) | Gold Prices | TimeSeriesSplit CV, RSI, MACD, lag features |
| 04 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aurovindhya/ml-portfolio-suite/blob/main/notebooks/04_ChestDisease_Colab.ipynb) | Chest Disease | EfficientNet, multi-label BCE, Grad-CAM |
| 05 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aurovindhya/ml-portfolio-suite/blob/main/notebooks/05_HeartDisease_Colab.ipynb) | Heart Disease | LightGBM, SHAP waterfall, threshold tuning |
| 06 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aurovindhya/ml-portfolio-suite/blob/main/notebooks/06_SpamFilter_Colab.ipynb) | Spam Filter | Char n-grams, soft ensemble, precision vs recall |
| 07 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aurovindhya/ml-portfolio-suite/blob/main/notebooks/07_GenerativeAI_Colab.ipynb) | Generative AI | Chatbot, RAG pipeline, medical guardrails |
| 08 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aurovindhya/ml-portfolio-suite/blob/main/notebooks/08_HandTracking_Colab.ipynb) | Hand Tracking | MediaPipe landmarks, gesture rules, Grad-CAM |
| 09 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aurovindhya/ml-portfolio-suite/blob/main/notebooks/09_MarketInsights_Colab.ipynb) | Market Insights | Agentic workflow, tool-calling, eval scores |

---

## Environment Variables

```bash
# Google Gemini
GEMINI_API_KEY=your_key_here

# Langfuse (optional — tracing degrades gracefully without it)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```
