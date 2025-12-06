Setup and run (Windows PowerShell)

1. (Optional) Create and activate a virtual environment:

```powershell
cd "C:\Users\Asfa\Documents\Semester_8\IDS\Symptom2Diseases_Prediction_System\GUI_Development"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
python -m pip install --upgrade pip
python -m pip install -r ..\requirements.txt
```

3. Download required NLTK data (the app will also attempt this automatically):

```powershell
python download_nltk.py
```

4. Run the Flask app:

```powershell
python app.py
```

Notes:

- If you see pickle/version incompatibility errors when loading model files in `../Model_training_and_testing`, install the scikit-learn version used to save the models (for example, `pip install scikit-learn==1.0.2`).
- Ensure the model files `tfidf_vectorizer.pkl`, `best_model_svm.pkl` and optionally `label_encoder.pkl` are present in `..\Model_training_and_testing`.
