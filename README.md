# Credit Security Risk Relay

A credit-risk modeling project with a Streamlit interface for single-application
assessment, batch CSV scoring, and exploration of the bundled German credit
dataset.

## Run locally

Use Python 3.13, which is compatible with the scikit-learn version used to create
the included model artifact. In PowerShell:

```powershell
cd "credit risk modeling"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`). Keep the model,
encoder, and CSV files in the same folder as `app.py`.

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, choose **Create app** and select this repository.
3. Set the app file path to `credit risk modeling/app.py`.
4. Select Python 3.13 in the app's advanced settings, then deploy.

The dependency file is next to the app at
[`credit risk modeling/requirements.txt`](./credit%20risk%20modeling/requirements.txt).
The model and encoder files are loaded relative to `app.py`; no secrets or external
services are needed.

## Deploy to Vercel

Vercel serves a lightweight web frontend and a FastAPI prediction function; the
Streamlit app remains available for local use. The Vercel function uses the
root [`requirements.txt`](./requirements.txt) and loads the trained pipeline
from `credit risk modeling/credit_risk_pipeline.joblib`.

To deploy with the Vercel CLI from the repository root:

```powershell
npx vercel
npx vercel --prod
```

Follow the CLI prompts to sign in and link the project. Keep the Vercel project root set to the repository root so it can find
`main.py`, `vercel_app.py`, `requirements.txt`, `vercel.json`, the `public`
frontend, and the model artifact. The live prediction
API is `POST /api/predict`; interactive API documentation is available at
`/docs`.

## Model notes

The Extra Trees model expects nine inputs: `Age`, `Sex`, `Job`, `Housing`,
`Saving accounts`, `Checking account`, `Credit amount`, `Duration`, and `Purpose`.
To retrain it from the bundled CSV and print
held-out evaluation metrics, run:

```powershell
python train_model.py
```

The training script evaluates on a stratified holdout, then trains the final
pipeline on all available rows and saves `credit_risk_pipeline.joblib`. Missing
savings-account values are imputed by the fitted preprocessing pipeline.
Predictions are demonstrations only and must not be used as the sole basis for
a real credit decision.