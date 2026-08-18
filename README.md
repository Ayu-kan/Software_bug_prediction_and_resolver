# Software Bug Prediction & Solution Generation System

An enterprise-grade, multi-language software bug risk prediction platform powered by static code analysis, NetworkX graph dependency risk modeling, ML predictions, and LLM solution generation.

---

## 🚀 Key Features

- **Multi-Language Static Code Analysis**: Static code metrics parsing for Python (`.py`), JavaScript (`.js`/`.jsx`), TypeScript (`.ts`/`.tsx`), and Java (`.java`).
- **Smart File Filtering**: Intelligent exclusion of `.venv`, `node_modules`, binary files, datasets, and logs with focus on high-impact API, Auth, and DB modules.
- **NetworkX Dependency Risk Graph**: Calculates file coupling, fan-in, fan-out, and betweenness centrality.
- **Architectural Risk Layering**: Auto-classifies code files into Authentication & Security, Database Models, Backend APIs, Core Services, and Frontend components.
- **Hybrid Risk Scoring**: Combines Machine Learning bug predictions (XGBoost/Random Forest), Dependency Risk, and Architectural Criticality.
- **LLM AI Solution Engine**: Automatic secret redaction (stripping API keys and secrets) followed by AI fix recommendations.
- **Streamlit Enterprise Dashboard**: Interactive progress tracking, metric cards, ranked tables, file deep dives, and downloadable CSV reports.
- **Historical Run Persistence**: SQLite database storage for analysis runs and user feedback history.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd bug_prediction_project
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

To launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`. Enter a GitHub repository URL or a local folder path to initiate the complete analysis pipeline.

---

## 🧪 Running Tests

Execute the unit test suite:

```bash
python -m unittest discover -s tests
```
