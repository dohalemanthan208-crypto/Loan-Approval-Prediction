import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from train_model import ensure_model

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "loan_app.db"
MODEL_PATH = APP_DIR / "models" / "loan_model.joblib"

st.set_page_config(page_title="LoanWise | Loan Prediction", page_icon="🏦", layout="wide")

# ---------- Database and authentication ----------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                created_at TEXT NOT NULL,
                applicant_income REAL,
                coapplicant_income REAL,
                loan_amount REAL,
                loan_term REAL,
                credit_history INTEGER,
                education TEXT,
                self_employed TEXT,
                property_area TEXT,
                prediction TEXT,
                approval_probability REAL
            )
        """)

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000).hex()

def create_user(username, password):
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    salt = secrets.token_hex(16)
    pwd_hash = hash_password(password, salt)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users(username, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, salt, pwd_hash, datetime.now().isoformat(timespec="seconds"))
            )
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username already exists."

def verify_user(username, password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    if not row:
        return False
    salt, expected = row
    return hmac.compare_digest(hash_password(password, salt), expected)

def save_prediction(username, values, prediction, probability):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                username, created_at, applicant_income, coapplicant_income,
                loan_amount, loan_term, credit_history, education,
                self_employed, property_area, prediction, approval_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username, datetime.now().isoformat(timespec="seconds"),
            values["ApplicantIncome"], values["CoapplicantIncome"],
            values["LoanAmount"], values["Loan_Amount_Term"],
            values["Credit_History"], values["Education"],
            values["Self_Employed"], values["Property_Area"],
            prediction, float(probability)
        ))

def load_history(username):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """SELECT created_at AS Date, applicant_income AS Applicant_Income,
                      coapplicant_income AS Coapplicant_Income, loan_amount AS Loan_Amount,
                      loan_term AS Loan_Term_Months, credit_history AS Credit_History,
                      education AS Education, self_employed AS Self_Employed,
                      property_area AS Property_Area, prediction AS Prediction,
                      approval_probability AS Approval_Probability
               FROM predictions WHERE username = ? ORDER BY id DESC""",
            conn, params=(username,)
        )

init_db()

# Create or load model and metrics. First run trains models.
with st.spinner("Preparing the machine-learning models..."):
    bundle = ensure_model(MODEL_PATH)

model = bundle["best_model"]
metrics_df = bundle["metrics"]
data_source = bundle["data_source"]
feature_names = bundle["feature_names"]
test_y = bundle["test_y"]
test_pred = bundle["test_pred"]

# ---------- Session state ----------
if "user" not in st.session_state:
    st.session_state.user = None

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#246a73); color: white;}
.hero h1 {color: white; margin-bottom: .25rem;}
.small-note {font-size: .9rem; color: #667085;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# ---------- Login / Register ----------
if st.session_state.user is None:
    st.markdown('<div class="hero"><h1>🏦 LoanWise</h1><p>Loan Approval Prediction System powered by Machine Learning</p></div>', unsafe_allow_html=True)
    st.write("")
    st.info("Educational project only. Predictions are estimates from a model and are not actual bank decisions.")
    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            login_name = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Log in", use_container_width=True)
        if login_submit:
            if verify_user(login_name, login_password):
                st.session_state.user = login_name.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    with register_tab:
        with st.form("register_form"):
            new_name = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password (at least 8 characters)", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            register_submit = st.form_submit_button("Register", use_container_width=True)
        if register_submit:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = create_user(new_name, new_password)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
    st.stop()

# ---------- Main application ----------
user = st.session_state.user
with st.sidebar:
    st.markdown("## 🏦 LoanWise")
    st.caption(f"Logged in as **{user}**")
    page = st.radio("Navigate", ["Home & Predict", "Model Dashboard", "Prediction History"], label_visibility="visible")
    st.divider()
    st.caption("SMLDS Mini Project")
    st.caption(f"Dataset mode: {data_source}")
    if st.button("Log out", use_container_width=True):
        st.session_state.user = None
        st.rerun()

if page == "Home & Predict":
    st.markdown('<div class="hero"><h1>Loan Approval Prediction</h1><p>Enter applicant details to receive a model-based estimate.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.warning("This prediction is for learning and demonstration only. Do not use it as a real credit decision.")
    with st.form("applicant_form"):
        st.subheader("Applicant information")
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_income = st.number_input("Applicant monthly income (₹)", min_value=0.0, max_value=10_000_000.0, value=50000.0, step=1000.0)
            coapplicant_income = st.number_input("Co-applicant monthly income (₹)", min_value=0.0, max_value=10_000_000.0, value=0.0, step=1000.0)
            loan_amount = st.number_input("Loan amount (₹ thousands)", min_value=1.0, max_value=100_000.0, value=150.0, step=10.0, help="Enter amount in thousands, matching the common public dataset format.")
        with c2:
            loan_term = st.selectbox("Loan term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360], index=8)
            credit_history = st.selectbox("Credit history", [1, 0], format_func=lambda x: "Good / available (1)" if x == 1 else "Poor / unavailable (0)")
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with c3:
            self_employed = st.selectbox("Self-employed", ["No", "Yes"])
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        submitted = st.form_submit_button("Predict loan outcome", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": int(credit_history),
            "Property_Area": property_area
        }])
        try:
            prediction = int(model.predict(row)[0])
            probabilities = model.predict_proba(row)[0]
            class_list = list(model.classes_)
            probability = float(probabilities[class_list.index(1)]) if 1 in class_list else 0.0
            result = "Likely Approved" if prediction == 1 else "Likely Rejected"
            save_prediction(user, {
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Education": education,
                "Self_Employed": self_employed,
                "Property_Area": property_area
            }, result, probability)
            st.divider()
            st.subheader("Prediction result")
            col1, col2 = st.columns([1, 1])
            with col1:
                if prediction == 1:
                    st.success("### ✅ Likely Approved")
                else:
                    st.error("### ❌ Likely Rejected")
                st.metric("Model-estimated approval probability", f"{probability:.1%}")
                st.progress(max(0.0, min(1.0, probability)))
            with col2:
                st.markdown("**Applicant summary**")
                st.write(f"- Applicant income: ₹{applicant_income:,.0f}")
                st.write(f"- Co-applicant income: ₹{coapplicant_income:,.0f}")
                st.write(f"- Loan amount: ₹{loan_amount * 1000:,.0f}")
                st.write(f"- Loan term: {loan_term} months")
                st.write(f"- Credit history: {'Good / available' if credit_history == 1 else 'Poor / unavailable'}")
            st.caption("The probability is the model's estimate, not a guarantee. Actual lending decisions consider many additional factors.")
        except Exception as exc:
            st.error(f"Could not make prediction: {exc}")
            st.info("Check that the training dataset uses the expected loan prediction columns.")

elif page == "Model Dashboard":
    st.markdown('<div class="hero"><h1>Model Performance Dashboard</h1><p>Compare classification algorithms on a held-out test set.</p></div>', unsafe_allow_html=True)
    st.caption(f"Dataset used: {data_source}. Metrics are calculated on a test split; values can vary with dataset and random seed.")
    display_metrics = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_metrics[col] = display_metrics[col].round(3)
    st.subheader("Algorithm comparison")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    plot_df = metrics_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"], var_name="Metric", value_name="Score")
    fig = px.bar(plot_df, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1], title="Classification metrics by model")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader(f"Confusion matrix — selected model ({bundle['best_model_name']})")
    cm = confusion_matrix(test_y, test_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual Rejected", "Actual Approved"], columns=["Predicted Rejected", "Predicted Approved"])
    fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Blues", title="Confusion matrix")
    st.plotly_chart(fig_cm, use_container_width=True)
    with st.expander("Explain the metrics"):
        st.markdown("""
- **Accuracy:** proportion of all predictions that were correct.
- **Precision:** among applications predicted as approved, the proportion actually approved.
- **Recall:** among actually approved applications, the proportion correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** shows correct and incorrect predictions for each class.
""")
    st.subheader("Dataset overview")
    st.write(f"Training/evaluation mode: **{data_source}**")
    st.write("The public dataset typically contains demographic, income, loan, credit-history and property-area fields. The synthetic fallback is generated only to demonstrate the app.")

elif page == "Prediction History":
    st.markdown('<div class="hero"><h1>Prediction History</h1><p>Your saved predictions for this account.</p></div>', unsafe_allow_html=True)
    history = load_history(user)
    if history.empty:
        st.info("No predictions saved yet. Go to Home & Predict to create your first prediction.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total predictions", len(history))
        b.metric("Likely approved", int((history["Prediction"] == "Likely Approved").sum()))
        c.metric("Likely rejected", int((history["Prediction"] == "Likely Rejected").sum()))
        st.dataframe(history, use_container_width=True, hide_index=True)
        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button("Download history as CSV", data=csv, file_name="loan_prediction_history.csv", mime="text/csv")
        fig = px.pie(history, names="Prediction", title="Distribution of your predictions")
        st.plotly_chart(fig, use_container_width=True)
