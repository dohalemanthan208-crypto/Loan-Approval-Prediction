# LoanWise — Loan Approval Prediction System

SMLDS mini project using Python, Streamlit and classification algorithms.

## Features
- User registration and login (passwords are stored as salted PBKDF2 hashes).
- Applicant form and prediction result.
- Logistic Regression, Decision Tree and Random Forest comparison.
- Accuracy, precision, recall, F1-score and confusion matrix.
- SQLite prediction history with CSV export.
- Uses a public CSV dataset if `data/train.csv` exists; otherwise creates a synthetic demo dataset.

## 1. Requirements
- Python 3.10 or newer recommended
- VS Code
- Internet connection only for installing Python packages (not required to run after installation)

## 2. Open in VS Code
1. Extract this ZIP file.
2. In VS Code, choose **File → Open Folder** and select `Loan_Approval_Prediction`.
3. Open **Terminal → New Terminal**.

## 3. Create a virtual environment

Windows:
```powershell
py -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

If Windows blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

## 4. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the app
```bash
streamlit run app.py
```
Streamlit should open the app in your browser. If not, open the Local URL printed in the terminal, usually `http://localhost:8501`.

## Dataset instructions

The project runs immediately with a synthetic dataset for demonstration. To use the common public Loan Prediction dataset:
1. Download the training CSV from the Kaggle **Loan Prediction Problem Dataset** (search that exact title on Kaggle).
2. Put the file in the `data` folder and name it `train.csv`.
3. The CSV should contain these columns: `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status`.
4. Stop and restart Streamlit after adding or replacing the CSV.

The app ignores `Loan_ID`. The target `Loan_Status` should contain `Y` and `N`. If your downloaded dataset has different column names, update `FEATURES` and the target mapping in `train_model.py`.

**Important:** the fallback data is synthetic, not a real bank dataset. Label it as a demo dataset in your project report and presentation. Do not claim its results represent real bank performance.

## How to demonstrate
1. Register a username and password.
2. Log in.
3. Open **Home & Predict**, fill out the form and click **Predict loan outcome**.
4. Open **Model Dashboard** and explain model comparison and confusion matrix.
5. Open **Prediction History** to show saved results and CSV export.
6. Explain preprocessing: missing-value imputation, one-hot encoding and scaling.
7. Explain train-test split and classification metrics.

## Suggested viva explanation
- **Problem:** estimate loan application status using applicant attributes.
- **Type:** supervised binary classification.
- **Target:** `Loan_Status` — Y (approved) / N (rejected).
- **Models:** Logistic Regression, Decision Tree, Random Forest.
- **Preprocessing:** median imputation and scaling for numeric fields; most-frequent imputation and one-hot encoding for categorical fields.
- **Evaluation:** accuracy, precision, recall, F1-score and confusion matrix.
- **Storage:** SQLite stores accounts and each user's prediction history.
- **Limitations:** public data may be small or outdated; synthetic demo data is not real; estimates can be biased and must not be used for actual lending.

## Project structure
```text
Loan_Approval_Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── .streamlit/
    └── config.toml
```

## Security note
This is a local college demonstration app, not production authentication. Do not use real financial or identity data, expose it publicly, or reuse important passwords.
