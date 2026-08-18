"""
dashboard/app.py
-----------------
Enhanced Modern Streamlit Dashboard for Bug Risk Prediction Pipeline.
Supports Phases 1 - 12 (Filtering, Multi-language analysis, Hybrid Risk, Scenario generator, LLM solutions, Run history).
"""

import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repo.validator import validate_repo_input, TemporaryClone
from extract_features import extract
from llm.llm_service import LLMSolutionEngine
from history.persistence import record_analysis_run, get_analysis_history, record_solution_feedback

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Bug Risk Intelligence Platform", layout="wide", page_icon="🐞")

# Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🐞 Enterprise Bug Risk Intelligence Platform")
st.caption("AI-Powered Multi-Language Code Analysis, Graph Dependency Risk & LLM Solution Generation")

tabs = st.tabs(["🚀 Repository Analysis", "📜 Analysis History", "⚙️ Configuration"])

def load_best_model():
    for name, fname, needs_scaler in [
        ("XGBoost", "xgboost.joblib", False),
        ("Random Forest", "random_forest.joblib", False),
        ("Logistic Regression", "logistic_regression.joblib", True),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            model = joblib.load(path)
            scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib")) if needs_scaler else None
            return name, model, scaler
    return None, None, None

with tabs[0]:
    repo_input = st.text_input(
        "Repository Path or GitHub URL",
        placeholder="https://github.com/psf/requests OR F:\\MyProjects\\repo",
    )
    col_a, col_b = st.columns([1, 4])
    with col_a:
        run_btn = st.button("⚡ Run Deep Analysis", type="primary", use_container_width=True)

    if run_btn:
        is_valid, err_msg, meta = validate_repo_input(repo_input)
        if not is_valid:
            st.error(err_msg)
            st.stop()

        model_name, model, scaler = load_best_model()
        if model is None:
            st.error("No trained ML model found. Please train models first.")
            st.stop()

        try:
            feature_cols = json.load(open(os.path.join(MODELS_DIR, "feature_columns.json")))
        except Exception:
            feature_cols = ["loc", "complexity", "function_count", "avg_function_size", 
                            "max_function_size", "dependency_count", "commit_count", 
                            "developer_count", "lines_added", "lines_deleted", 
                            "code_churn", "recent_commit_count", "days_since_last_change", 
                            "previous_bug_count"]

        status_box = st.status("Initializing Analysis Pipeline...", expanded=True)
        tmp_csv = "_dashboard_tmp.csv"

        try:
            if meta.get("is_url"):
                status_box.update(label="1/5 Fetching repository (Shallow temp clone)...")
                with TemporaryClone(repo_input) as clone_dir:
                    status_box.update(label="2/5 Filtering files & performing static code analysis...")
                    extract(clone_dir, tmp_csv, cutoff_ratio=1.0)
            else:
                status_box.update(label="1/5 Validated local repository path.")
                status_box.update(label="2/5 Filtering files & running static code analysis...")
                extract(repo_input, tmp_csv, cutoff_ratio=1.0)

            status_box.update(label="3/5 Building dependency graph & architectural roles...")
            df = pd.read_csv(tmp_csv)

            status_box.update(label="4/5 Running ML model & Hybrid Risk scoring...")
            
            # Predict
            missing_cols = [c for c in feature_cols if c not in df.columns]
            for c in missing_cols:
                df[c] = 0
                
            X = df[feature_cols].fillna(0)
            X_input = scaler.transform(X) if scaler is not None else X
            
            ml_probs = model.predict_proba(X_input)[:, 1] if hasattr(model, "predict_proba") else [0.5]*len(df)
            df["ml_probability"] = ml_probs

            # Hybrid Risk Score calculation
            dep_risk = df.get("dependency_risk", 0) / 100.0
            arch_risk = df.get("architecture_risk", 10) / 100.0
            
            df["hybrid_risk_score"] = (
                0.50 * df["ml_probability"] +
                0.25 * dep_risk +
                0.25 * arch_risk
            )
            df["risk_%"] = (df["hybrid_risk_score"] * 100).round(1)

            # Categorize Risk
            def get_risk_tier(r):
                if r >= 70:
                    return "🔴 High Risk"
                elif r >= 40:
                    return "🟡 Medium Risk"
                return "🟢 Low Risk"

            df["risk_category"] = df["risk_%"].apply(get_risk_tier)
            df = df.sort_values("hybrid_risk_score", ascending=False).reset_index(drop=True)

            # Compute Risk Cause Explanation for each file
            def compute_risk_cause(row):
                causes = []
                if row.get("complexity", 0) > 15:
                    causes.append(f"High cyclomatic complexity ({row['complexity']})")
                if row.get("code_churn", 0) > 100:
                    causes.append(f"Frequent code churn ({row['code_churn']} lines altered)")
                if row.get("previous_bug_count", 0) > 0:
                    causes.append(f"History of bug fixes ({row['previous_bug_count']} past bugs)")
                if row.get("dependency_risk", 0) > 40:
                    causes.append(f"High dependency coupling (Fan-in: {row.get('fan_in', 0)})")
                if row.get("architecture_risk", 0) >= 80:
                    causes.append(f"Critical architecture role ({row.get('architecture_role', 'Security/DB')})")
                return " | ".join(causes) if causes else "Moderate metric thresholds"

            df["risk_cause_description"] = df.apply(compute_risk_cause, axis=1)

            status_box.update(label="5/5 Analysis complete!", state="complete")

            # Persist run
            high_risk_cnt = int((df["risk_%"] >= 70).sum())
            record_analysis_run(repo_input, len(df), high_risk_cnt, df.head(10).to_dict(orient="records"))

            # Metrics display
            st.subheader("📊 Repository Risk Summary")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Code Files", len(df))
            with m2:
                st.metric("🔴 High Risk Files (≥70%)", high_risk_cnt)
            with m3:
                med_cnt = int(((df["risk_%"] >= 40) & (df["risk_%"] < 70)).sum())
                st.metric("🟡 Medium Risk Files", med_cnt)
            with m4:
                st.metric("Average Risk Score", f"{df['risk_%'].mean():.1f}%")

            # Top Risky Files Chart
            st.subheader("🔥 Top Highest Bug Risk Files")
            top10 = df.head(10)
            st.bar_chart(top10.set_index("file")["risk_%"])

            # Detailed Data Table
            st.subheader("📋 Complete Ranked Risk Inventory")
            show_cols = ["file", "risk_category", "risk_%", "risk_cause_description", "architecture_role", "complexity", "dependency_count", "code_churn"]
            st.dataframe(
                df[show_cols].rename(columns={
                    "risk_category": "Risk Level",
                    "risk_%": "Risk Score (%)",
                    "risk_cause_description": "Why File Is Risky / Causes Error",
                    "architecture_role": "Layer Role",
                    "complexity": "Complexity",
                    "dependency_count": "Dependencies",
                    "code_churn": "Churn Lines"
                }),
                use_container_width=True
            )

            # File Deep Dive & LLM Solutions
            st.divider()
            st.subheader("🧠 Deep Dive & AI Solution Generation")
            selected_file = st.selectbox("Select file to analyze & generate fix for:", df["file"].tolist())
            
            if selected_file:
                file_row = df[df["file"] == selected_file].iloc[0]
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Architecture Role:** `{file_row.get('architecture_role', 'N/A')}`")
                    st.write(f"**Cyclomatic Complexity:** `{file_row.get('complexity', 0)}`")
                    st.write(f"**Lines of Code:** `{file_row.get('loc', 0)}`")
                    st.write(f"**Why File Is Risky:** `{file_row.get('risk_cause_description', 'N/A')}`")
                with c2:
                    st.write(f"**Dependencies:** `{file_row.get('dependency_count', 0)}`")
                    st.write(f"**Historical Churn:** `{file_row.get('code_churn', 0)} lines`")
                    st.write(f"**Previous Bug Fixes:** `{file_row.get('previous_bug_count', 0)}`")

                # Future Development & Current Defect Projections
                st.subheader("🔮 Bug Forecast: Current Defects vs Future Development Risks")
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.markdown("##### 🐛 Current Code Defect Triggers")
                    st.info(
                        f"- **Complexity Bottleneck**: Cyclomatic score of `{file_row.get('complexity', 0)}` indicates high branching density.\n"
                        f"- **Historical Defects**: `{file_row.get('previous_bug_count', 0)}` previous bug fixes in git history.\n"
                        f"- **Code Churn Volatility**: `{file_row.get('code_churn', 0)}` total lines modified recently."
                    )
                with b_col2:
                    st.markdown("##### ⚡ Future Development Risk Scenarios")
                    st.warning(
                        f"- **Ripple Regression**: Modifying `{file_row['file']}` could break `{file_row.get('fan_in', 0)}` dependent module(s).\n"
                        f"- **Architecture Drift**: Operates as `{file_row.get('architecture_role', 'Module')}`—changes here impact system security/DB boundaries.\n"
                        f"- **Maintenance Overhead**: High churn makes future feature additions prone to unexpected merge conflicts."
                    )

                api_key_input = st.text_input(
                    "Optional LLM API Key (OpenAI / Gemini) for Live Fix Generation",
                    type="password",
                    help="Leave blank to use the built-in intelligent engine, or enter your API key for live AI generation."
                )

                if st.button("🤖 Generate AI Fix Solution"):
                    llm = LLMSolutionEngine(api_key=api_key_input.strip() if api_key_input.strip() else None)
                    solution = llm.generate_solution(
                        selected_file,
                        file_row.get("last_source_code", ""),
                        file_row["risk_cause_description"],
                        file_row["hybrid_risk_score"],
                        row_data=file_row.to_dict()
                    )
                    st.info(f"**Problem Diagnosis & Root Cause:** {solution['problem_summary']}")
                    st.markdown(f"**Suggested Solution Steps:**\n{solution['suggested_fix']}")
                    st.subheader("🛠️ AI Refactored Code Fix:")
                    st.code(solution["improved_code"], language="python")
                    st.warning(f"**Potential Side Effects & Regression Risks:**\n{solution['possible_side_effects']}")

                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        if st.button("👍 Accept Solution"):
                            record_solution_feedback(selected_file, "Accepted")
                            st.success("Recorded positive feedback!")
                    with col_f2:
                        if st.button("👎 Reject Solution"):
                            record_solution_feedback(selected_file, "Rejected")
                            st.error("Recorded feedback.")

            # CSV Download
            st.download_button(
                "📥 Download Risk CSV Report",
                df.to_csv(index=False).encode("utf-8"),
                "bug_risk_analysis.csv",
                "text/csv",
                type="primary"
            )

        finally:
            if os.path.exists(tmp_csv):
                try:
                    os.remove(tmp_csv)
                except Exception:
                    pass

with tabs[1]:
    st.subheader("📜 Historical Repository Analyses")
    history = get_analysis_history()
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.info("No analysis history recorded yet.")

with tabs[2]:
    st.subheader("⚙️ System Configuration")
    st.json({
        "Supported Languages": ["Python", "JavaScript", "TypeScript", "Java"],
        "Max Repository Size": "500 MB",
        "Hybrid Risk Weights": {
            "ML Prediction": "50%",
            "Dependency Risk": "25%",
            "Architecture Risk": "25%"
        }
    })
