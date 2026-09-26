# Startup Success Predictor

A small Flask web app that predicts whether a startup is likely to be
**acquired** or **closed**, trained on the Kaggle dataset
["Startup Success Prediction"](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction)
by manishkc06.

## Project structure

```
startup-success-predictor/
├── app.py                # Flask web app (serves the form + predictions)
├── train_model.py        # Trains the model from the CSV
├── requirements.txt
├── Procfile              # Tells Render how to start the app
├── data/
│   └── startup_data.csv  # <- you add this (see step 1 below)
├── model/                 # created by train_model.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## 1. Get the dataset

1. Go to the Kaggle page: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction
2. Click **Download** (you'll need a free Kaggle account, and Kaggle may ask
   you to accept the dataset's terms).
3. Unzip it, and rename the CSV file (it's usually called
   `startup data.csv` or similar) to `startup_data.csv`.
4. Put it inside this project's `data/` folder, so you have
   `data/startup_data.csv`.

## 2. Run it locally in VS Code

**Requirements:** Python 3.10+ and VS Code with the Python extension installed.

1. Open the `startup-success-predictor` folder in VS Code
   (`File → Open Folder…`).
2. Open a terminal in VS Code: `Terminal → New Terminal`.
3. Create and activate a virtual environment:

   - macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     venv\Scripts\Activate.ps1
     ```

   When it's active you'll see `(venv)` at the start of your terminal prompt.
   In VS Code, also select this interpreter: press `Ctrl/Cmd+Shift+P` →
   "Python: Select Interpreter" → choose the one inside `venv`.

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Train the model (this reads `data/startup_data.csv` and writes files into
   `model/`):
   ```bash
   python train_model.py
   ```
   You should see a test accuracy printed, plus a classification report.
6. Run the web app:
   ```bash
   python app.py
   ```
7. Open your browser to **http://127.0.0.1:5000** — you'll see the form.
   Fill it in and click **Predict**.

Every time you re-run `train_model.py` it will overwrite the saved model, so
`app.py` will automatically use the newest version next time you restart it.

## 3. Put the project on GitHub

Render deploys from a Git repository, so push this folder to GitHub first.

1. In the VS Code terminal:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: startup success predictor"
   ```
2. Create a new empty repository on GitHub (no README/license — you already
   have files).
3. Connect and push:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git branch -M main
   git push -u origin main
   ```

**Important:** the trained model files in `model/` (and your CSV in `data/`)
are small enough to commit for this project, so just leave the `.gitignore`
as-is (it only excludes virtual-env and cache files). If GitHub warns the CSV
is large, you can instead add `data/` and `model/` to `.gitignore` and have
Render run `python train_model.py` as part of the build — see the note in
step 4.

## 4. Deploy on Render

1. Go to https://render.com and sign in (you can sign in with GitHub).
2. Click **New +** → **Web Service**.
3. Connect your GitHub account if you haven't, then select the repository
   you just pushed.
4. Configure the service:
   - **Name:** anything you like, e.g. `startup-success-predictor`
   - **Region:** whichever is closest to you
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:**
     ```
     pip install -r requirements.txt
     ```
     If you excluded `data/` and `model/` from Git, use instead:
     ```
     pip install -r requirements.txt && python train_model.py
     ```
     (this requires the CSV to also be committed, or fetched some other way,
     since Render's build step has no access to your local machine)
   - **Start Command:**
     ```
     gunicorn app:app
     ```
   - **Instance Type:** Free is fine for testing.
5. Click **Create Web Service**. Render will install dependencies, run the
   build, and start the app. The first deploy takes a few minutes.
6. Once it says **Live**, open the URL Render gives you
   (something like `https://startup-success-predictor.onrender.com`) —
   that's your public app.

Any time you `git push` new commits to `main`, Render will automatically
redeploy.

## Notes on the model

- It's a `RandomForestClassifier` trained on a subset of the dataset's
  columns: funding history, milestones, relationships, state, and industry
  category.
- The target is derived from the dataset's `status` column: `acquired` = 1,
  `closed` = 0. (Rows with other statuses, if any, are dropped during
  training since the `status` filter only checks for "acquired".)
- This is a simple demo model for learning purposes — it shouldn't be used
  for real investment decisions. Startup outcomes depend on many factors a
  spreadsheet can't capture.
