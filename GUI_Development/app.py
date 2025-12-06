import joblib
import re
import nltk
import os
import sys
import csv
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords

# ----------------------------
# Initialize Flask App
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Paths
# ----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "../Model_training_and_testing")

# ----------------------------
# Ensure required NLTK data and load stopwords
# ----------------------------
def ensure_nltk_stopwords():
    """Make sure the NLTK 'stopwords' corpus is available; download if missing."""
    try:
        # Attempt to access stopwords; this will raise LookupError if missing
        _ = stopwords.words('english')
    except LookupError:
        try:
            print("NLTK stopwords not found. Downloading now...")
            nltk.download('stopwords')
        except Exception as e:
            print(f"Warning: failed to download NLTK stopwords: {e}")


ensure_nltk_stopwords()
stop_words = set(stopwords.words('english'))

# ----------------------------
# Load model, vectorizer, and label encoder
# ----------------------------
print("Loading model, vectorizer, and label encoder...")

# Verify model files exist before attempting to load them; give clear messages if missing
vec_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
model_path = os.path.join(model_dir, 'best_model_svm.pkl')
label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

missing = []
if not os.path.exists(vec_path):
    missing.append(vec_path)
if not os.path.exists(model_path):
    missing.append(model_path)

if missing:
    print("Error: The following model files are missing:")
    for p in missing:
        print(f" - {p}")
    print("Please ensure the trained model files exist in the '../Model_training_and_testing' folder.")
    sys.exit(1)

try:
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)  # Your best SVM model
except Exception as e:
    print(f"Failed to load model or vectorizer: {e}")
    print("Common causes: incompatible scikit-learn version used to create the pickles.\n" \
          "If needed, install a matching scikit-learn version, e.g. `pip install scikit-learn==1.0.2`.")
    sys.exit(1)

# Check if label_encoder exists; optional fallback if missing
if os.path.exists(label_encoder_path):
    try:
        label_encoder = joblib.load(label_encoder_path)
        use_label_encoder = True
    except Exception as e:
        print(f"Warning: failed to load label encoder: {e}")
        use_label_encoder = False
else:
    print("Label encoder not found. Predictions will be numeric.")
    use_label_encoder = False

# Fallback: try to build a mapping from encoded values to disease names using encoded_data.csv
label_mapping = {}
if not use_label_encoder:
    enc_csv = os.path.join(model_dir, 'encoded_data.csv')
    if os.path.exists(enc_csv):
        try:
            with open(enc_csv, newline='', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    # expected columns: 'label' and 'encoded_label'
                    enc = row.get('encoded_label')
                    lab = row.get('label')
                    if enc is None or lab is None:
                        continue
                    try:
                        key = int(enc)
                    except Exception:
                        continue
                    if key not in label_mapping:
                        label_mapping[key] = lab
            if label_mapping:
                print(f"Built fallback label mapping from {enc_csv} ({len(label_mapping)} entries)")
            else:
                print(f"No label mappings found in {enc_csv}")
        except Exception as e:
            print(f"Failed to build label mapping from {enc_csv}: {e}")
    else:
        print(f"Fallback CSV {enc_csv} not found; predictions will remain numeric.")

print("Model, vectorizer, and label encoder loaded successfully!")

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    """
    Clean the input text by:
    - Removing non-alphabetic characters
    - Lowercasing
    - Removing stopwords
    """
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        user_input = request.form.get("user_input", "").strip()
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Clean input
        cleaned_input = clean_text(user_input)
        if not cleaned_input:
            return jsonify({"error": "Input empty after cleaning"}), 400

        # Transform input and predict
        X_tfidf = vectorizer.transform([cleaned_input])
        prediction_encoded = model.predict(X_tfidf)[0]

        # Decode numeric label to disease name if label_encoder exists
        if use_label_encoder:
            prediction_disease = label_encoder.inverse_transform([prediction_encoded])[0]
        else:
            # Try fallback mapping built from encoded_data.csv, otherwise return numeric label
            try:
                key = int(prediction_encoded)
                prediction_disease = label_mapping.get(key, str(prediction_encoded))
            except Exception:
                prediction_disease = str(prediction_encoded)

        # Return predicted disease
        return jsonify({"predicted_disease": prediction_disease})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run Flask App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
