# AI Learning Assistant – Emotion Detection & Personalized Learning Support Platform

An end-to-end, academic-grade AI web application that understands student emotional states from written descriptions of coursework difficulties and delivers tailored pedagogical support via Google Gemini Generative AI and dual deep learning models (PyTorch BiLSTM and BERT Transformer).

---

## 🌟 Key Highlights & Features

1. **Dual Deep Learning Architecture**:
   - **Model A (PyTorch BiLSTM)**: Custom bidirectional Long Short-Term Memory neural network with embedding, dropout, and classification head.
   - **Model B (BERT / Transformer)**: Contextual sequence representations capturing semantic and syntactic nuances.
2. **Rule-Based Emotion Enhancement & Mixed Emotion Engine**:
   - Detects contrasting clauses (e.g. *"I am confused about recursion, but excited to learn it"*) and isolates **Primary** and **Secondary** emotions.
3. **Google Gemini Generative AI Integration**:
   - Delivers structured 6-part pedagogical plans (Difficulty reflection, Supportive message, Personalized study tips, Step-by-step next actions, Recommended study methodology, Encouraging conclusion).
   - High-fidelity offline pedagogical engine active when API credentials are unset or offline.
4. **Interactive Multi-Page SaaS Interface (Streamlit + Modern CSS)**:
   - Deep Navy glassmorphism, responsive across desktop, tablet, and mobile.
   - 6 functional pages: Home, Student Assistant, Analytics Dashboard, Model Comparison, History Ledger, and Responsible AI Governance.
5. **Plotly Telemetry & Analytics Dashboard**:
   - Real-time interaction history visualization: Emotion distribution donuts, frequency bars, longitudinal trendlines, and confusion matrix heatmaps.
6. **Audit Trail & CSV Export**:
   - Searchable, filterable history ledger with 1-click CSV export and history clearing.

---

## 🏷️ The 5 Academic Emotional Dimensions

| Emotion | Core Indicator | Pedagogical Support Strategy |
| :--- | :--- | :--- |
| **🥱 BORED** | Lack of engagement, disinterest, passive fatigue | Active learning sprints, project-driven micro-tasks, gamification |
| **🌟 CONFIDENT** | Conceptual mastery, readiness for advanced stretch | Boundary conditions, teaching peers, advanced architecture projects |
| **🧩 CONFUSED** | Cognitive overload, conceptual roadblocks | Feynman Technique, visual mental models, minimal number tracing |
| **💡 CURIOUS** | Inquisitive drive, interest in underlying mechanisms | First-principles research, source code inspection, toy prototypes |
| **🔥 FRUSTRATED** | High stress, repetitive bugs, feeling blocked | Cognitive reset, Rubber Duck debugging, divide-and-conquer tracing |

---

## 🏗️ Project Architecture & Directory Structure

```
AI_Learning_Assistant/
│
├── app.py                         # Application entrypoint & global navbar routing
├── requirements.txt               # Dependencies optimized for local & Streamlit Cloud
├── README.md                      # Comprehensive project guide & documentation
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
│
├── .streamlit/
│   └── config.toml                # Dark theme configuration (Navy & Cyan/Violet)
│
├── pages/
│   ├── 1_🏠_Home.py               # Landing page with hero, workflow, feature cards
│   ├── 2_🎓_Learning_Assistant.py # Main interactive emotion detection & Gemini study plan
│   ├── 3_📊_Analytics_Dashboard.py# Plotly analytics, distribution, trends, filters
│   ├── 4_⚖️_Model_Comparison.py   # BiLSTM vs BERT architectures, metrics & live head-to-head tester
│   ├── 5_📜_Interaction_History.py# History table, search, filters, CSV export, delete
│   └── 6_🛡️_Responsible_AI.py    # Privacy, limitations, mental health disclaimer, ethics
│
├── models/
│   ├── __init__.py
│   ├── bilstm_model.py            # PyTorch BiLSTM neural network class & tokenizer
│   ├── bert_model.py              # Transformer/BERT emotion classification handler
│   ├── model_loader.py            # Cached model loading & singleton inference
│   └── trained/
│       ├── bilstm_model.pt        # Trained PyTorch checkpoint
│       ├── vocab.json             # Vocabulary mappings (963 tokens)
│       └── metrics.json           # Real evaluation metrics & confusion matrices
│
├── training/
│   ├── __init__.py
│   ├── preprocessing.py           # Text cleaning, normalization, tokenization, sequence padding
│   ├── train_bilstm.py            # BiLSTM training pipeline with validation and weight export
│   └── evaluate.py                # Evaluation suite: Accuracy, Precision, Recall, F1, Confusion Matrix
│
├── services/
│   ├── __init__.py
│   ├── emotion_detection.py       # Hybrid pipeline: BiLSTM + BERT + Rule-based enhancement
│   ├── gemini_service.py          # Google Gemini integration with prompt engineering & fallbacks
│   └── recommendation.py          # Pedagogical strategies & study tips mappings
│
├── data/
│   ├── dataset/
│   │   ├── academic_emotions.csv  # Curated, balanced 5-class dataset (train/val/test splits)
│   │   └── generate_dataset.py    # Dataset creation script
│   └── interaction_history/
│       └── interactions.csv       # Persistent CSV storage for student queries and predictions
│
├── utils/
│   ├── __init__.py
│   ├── config.py                  # Colors, emotion metadata, emojis, constants
│   ├── storage.py                 # Thread-safe CSV read/write and export utilities
│   └── metrics.py                 # Metric calculations and Plotly visualization helpers
│
└── assets/
    └── styles.css                 # Custom CSS (glassmorphism, gradients, navy-violet theme)
```

---

## ⚡ Quickstart & Local Execution

### 1. Clone or Open the Repository
```bash
cd AI_Learning_Assistant
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set Google Gemini API Key
To enable live generative AI responses from Google Gemini:
```bash
# On Windows (PowerShell)
$env:GEMINI_API_KEY="your_actual_gemini_api_key"

# On Linux/macOS
export GEMINI_API_KEY="your_actual_gemini_api_key"
```
*Note: You can also enter the API key directly into the sidebar text field inside the running web app!*

### 4. Launch the Web Application
```bash
streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🧠 Model Training & Evaluation Suite

The platform includes full source scripts to retrain the neural network and re-evaluate on test datasets:

### Retrain the BiLSTM Neural Network
```bash
python training/train_bilstm.py
```
*Trains for 30 epochs with AdamW optimizer, gradient clipping, learning rate scheduler, and saves the best model checkpoint to `models/trained/bilstm_model.pt`.*

### Run Empirical Evaluation (BiLSTM vs. BERT)
```bash
python training/evaluate.py
```
*Evaluates both models on held-out test data, computing Accuracy, Precision, Recall, F1-Score, and Confusion Matrices saved to `models/trained/metrics.json`.*

### Real Empirical Performance Benchmark
* Evaluated across verified student expressions:
  - **BiLSTM (PyTorch)**: Accuracy ~51.11% | F1-Score: 0.4860 | Latency: 12.4 ms (3.9x faster)
  - **BERT (Transformer)**: Accuracy: 60.00% | F1-Score: 0.6179 | Latency: 48.6 ms

---

## ☁️ Streamlit Community Cloud Deployment Guide

Follow these steps to deploy this application online:

### Step 1: Push Code to GitHub
1. Create a new GitHub repository named `ai-learning-assistant`.
2. Commit and push the project code:
   ```bash
   git init
   git add .
   git commit -m "Initial release of AI Learning Assistant"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ai-learning-assistant.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
2. Click **"New app"**.
3. Select your repository: `YOUR_USERNAME/ai-learning-assistant`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Advanced settings"** -> **Secrets**:
   Add your Gemini key:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
7. Click **"Deploy!"**. Streamlit Cloud will install dependencies from `requirements.txt` and launch your live public URL within 2 minutes!

---

## 🛡️ Responsible AI & Ethical Disclaimer

- **Strictly Educational**: This platform is designed solely to analyze text sentiment regarding academic obstacles.
- **Non-Clinical Boundary**: It **does not diagnose clinical depression, anxiety, or mental health conditions**. Students experiencing severe emotional distress must consult campus counseling or licensed healthcare professionals.
- **Privacy First**: No student names, student IDs, IP addresses, or biometrics are collected. Interactions are stored in an anonymized local CSV ledger that can be exported or purged at any time.

---

## 🎓 Academic Evaluation Project Summary
Developed as an end-to-end Academic Capstone Demonstration. Combines classical sequence recurrent networks (BiLSTM), state-of-the-art transformer contextual representations (BERT), rule-based postprocessing, and large language model pedagogical planning (Google Gemini).
