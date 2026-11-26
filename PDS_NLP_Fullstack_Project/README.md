# PDS Complaint NLP Analyzer — Frontend + Backend

## Folder structure
- `backend/app.py` — Flask API and NLP model training
- `backend/requirements.txt` — Python dependencies
- `backend/pds_nlp_1000_records_cleaned (1).csv` — labeled complaint dataset
- `frontend/index.html`, `style.css`, `script.js` — website

## Run on Windows
1. Install Python 3.10+.
2. Open a terminal in this project folder and run:
   ```powershell
   cd backend
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   python app.py
   ```
   If `py` is not recognized, try `python` instead. Keep this terminal open.
3. Open `frontend/index.html` in your browser, preferably through VS Code Live Server.
4. Enter a complaint and click **Analyze complaint**.

Backend runs at `http://127.0.0.1:5000`; API endpoints are `/api/health`, `/api/stats`, and `/api/predict`.

## Use your actual dataset
Replace `backend/pds_nlp_1000_records_cleaned (1).csv` with your cleaned, labeled dataset. Required columns are `original_text`, `category`, and `sentiment`. Common aliases such as `complaint`, `text`, `issue_category`, and `sentiment_label` are also accepted. Keep the file name or update `CSV_PATH` in `app.py`.

The included CSV contains the project's labeled complaints so the application can run immediately. Replace it with your actual labeled dataset for meaningful results. The model trains on all supplied rows at startup; confidence is not a guarantee of correctness. For academic evaluation, assess it separately on a held-out test set.

## Troubleshooting
- **Backend offline:** run `python app.py` in the backend terminal and leave it open.
- **Module missing:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **CSV error:** confirm the file is in `backend` and contains complaint, category and sentiment columns.
- **Browser fetch error:** use VS Code Live Server instead of a `file://` URL.
