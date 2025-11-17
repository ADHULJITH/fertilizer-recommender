import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# -----------------------------
# File Paths
# -----------------------------
# Ensure these files exist in your project directory
MODEL_PATH = 'decision_tree_model.pkl'
ENCODER_PATH = 'label_encoders.pkl'

app = Flask(__name__)

# -----------------------------
# Global Model + Encoders
# -----------------------------
model = None
label_encoders = None

# Fertilizer name mapping (model output → readable name)
FERTILIZER_MAPPING = {
    0: "Balanced NPK Fertilizer",
    1: "Compost",
    2: "DAP",
    3: "General Purpose Fertilizer",
    4: "Gypsum",
    5: "Lime",
    6: "Muriate of Potash",
    7: "Organic Fertilizer",
    8: "Urea",
    9: "Water Retaining Fertilizer"
}

# -----------------------------
# Load Model + Encoders
# -----------------------------
def load_assets():
    """Load trained model and LabelEncoders."""
    global model, label_encoders
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        with open(ENCODER_PATH, 'rb') as f:
            label_encoders = pickle.load(f)

        print("✔ Model and Label Encoders loaded successfully.\n")

    except FileNotFoundError:
        print(f"✘ Error: One or both files ({MODEL_PATH}, {ENCODER_PATH}) not found.")
        print("Please ensure the model and encoder files are in the same directory as app.py.")
        model = None
        label_encoders = None
    except Exception as e:
        print(f"✘ Error loading assets: {e}")
        model = None
        label_encoders = None


# -----------------------------
# Dropdown Options for HTML
# -----------------------------
def get_categorical_options():
    """Return Soil and Crop options from LabelEncoders."""
    if not label_encoders:
        # Fallback in case of loading error
        return {'Soil': [], 'Crop': []}

    return {
        'Soil': label_encoders['Soil'].classes_.tolist(),
        'Crop': label_encoders['Crop'].classes_.tolist()
    }


# -----------------------------
# ROUTE: Home Page (Landing)
# -----------------------------
@app.route('/', methods=['GET'])
def home():
    """Renders the professional landing page (home.html)."""
    return render_template('home.html')


# -----------------------------
# ROUTE: Recommendation Form Page
# -----------------------------
@app.route('/recommend', methods=['GET'])
def recommend_form():
    """Renders the main form page (recommend.html) with dropdown options."""
    if model is None or label_encoders is None:
        # Graceful error handling for missing assets
        return render_template(
            'error.html', 
            message="Application assets (model/encoders) failed to load. Please check server logs."
        ), 500

    options = get_categorical_options()
    return render_template(
        'recommend.html',
        soil_options=options['Soil'],
        crop_options=options['Crop']
    )


# -----------------------------
# ROUTE: Prediction (API)
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Handles POST request, runs prediction, and returns JSON result."""
    if model is None or label_encoders is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.json

        # --- Input Validation and Casting ---
        # Numerical Inputs (Attempt to convert to float)
        temp = float(data.get('Temperature'))
        moisture = float(data.get('Moisture'))
        ph = float(data.get('PH'))
        nitrogen = float(data.get('Nitrogen'))
        phosphorous = float(data.get('Phosphorous'))
        potassium = float(data.get('Potassium'))
        carbon = float(data.get('Carbon'))

        # Categorical Inputs (Get string values)
        soil = data.get('Soil')
        crop = data.get('Crop')
        
        # Check if categorical inputs are present in the training data
        if soil not in label_encoders['Soil'].classes_:
            return jsonify({'error': f"Unknown Soil Type: '{soil}'"}), 400
        if crop not in label_encoders['Crop'].classes_:
            return jsonify({'error': f"Unknown Crop Type: '{crop}'"}), 400

        # Encode categorical inputs
        soil_enc = label_encoders['Soil'].transform([soil])[0]
        crop_enc = label_encoders['Crop'].transform([crop])[0]

        # Feature Vector (9 features in the correct order)
        features = np.array([[
            temp, moisture, ph, nitrogen, phosphorous,
            potassium, carbon, soil_enc, crop_enc
        ]])

        # Prediction
        encoded_output = model.predict(features)[0]
        fertilizer_name = FERTILIZER_MAPPING.get(encoded_output, "Unknown Fertilizer")

        return jsonify({
            'success': True,
            'fertilizer_recommendation': fertilizer_name
        })

    except (TypeError, ValueError):
        # Catches errors if input values are missing or cannot be converted to float
        return jsonify({'error': 'Invalid input. Ensure all numerical fields are filled correctly.'}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f"Unexpected server error during prediction: {e}"}), 500


# -----------------------------
# START FLASK SERVER
# -----------------------------
if __name__ == '__main__':
    load_assets()

    print("🚀 Fertilizer Recommendation System starting...")
    print("🌐 Open your browser and visit: http://127.0.0.1:5000\n")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )









# import pickle
# import numpy as np
# from flask import Flask, render_template, request, jsonify

# # -----------------------------
# # File Paths
# # -----------------------------
# MODEL_PATH = 'decision_tree_model.pkl'
# ENCODER_PATH = 'label_encoders.pkl'

# app = Flask(__name__)

# # -----------------------------
# # Global Model + Encoders
# # -----------------------------
# model = None
# label_encoders = None

# # Fertilizer name mapping (model output → readable name)
# FERTILIZER_MAPPING = {
#     0: "Balanced NPK Fertilizer",
#     1: "Compost",
#     2: "DAP",
#     3: "General Purpose Fertilizer",
#     4: "Gypsum",
#     5: "Lime",
#     6: "Muriate of Potash",
#     7: "Organic Fertilizer",
#     8: "Urea",
#     9: "Water Retaining Fertilizer"
# }

# # -----------------------------
# # Load Model + Encoders
# # -----------------------------
# def load_assets():
#     """Load trained model and LabelEncoders."""
#     global model, label_encoders
#     try:
#         with open(MODEL_PATH, 'rb') as f:
#             model = pickle.load(f)

#         with open(ENCODER_PATH, 'rb') as f:
#             label_encoders = pickle.load(f)

#         print("✔ Model and Label Encoders loaded successfully.\n")

#     except Exception as e:
#         print(f"✘ Error loading assets: {e}")
#         model = None
#         label_encoders = None


# # -----------------------------
# # Dropdown Options for HTML
# # -----------------------------
# def get_categorical_options():
#     """Return Soil and Crop options from LabelEncoders."""
#     if not label_encoders:
#         return {'Soil': [], 'Crop': []}

#     return {
#         'Soil': label_encoders['Soil'].classes_.tolist(),
#         'Crop': label_encoders['Crop'].classes_.tolist()
#     }


# # -----------------------------
# # ROUTE: Home Page
# # -----------------------------
# @app.route('/', methods=['GET'])
# def index():
#     if model is None or label_encoders is None:
#         return "Error: Model or encoders failed to load.", 500

#     options = get_categorical_options()
#     return render_template(
#         'index.html',
#         soil_options=options['Soil'],
#         crop_options=options['Crop']
#     )


# # -----------------------------
# # ROUTE: Prediction
# # -----------------------------
# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None or label_encoders is None:
#         return jsonify({'error': 'Model not loaded.'}), 500

#     try:
#         data = request.json

#         # Numerical Inputs
#         temp = float(data.get('Temperature'))
#         moisture = float(data.get('Moisture'))
#         ph = float(data.get('PH'))
#         nitrogen = float(data.get('Nitrogen'))
#         phosphorous = float(data.get('Phosphorous'))
#         potassium = float(data.get('Potassium'))
#         carbon = float(data.get('Carbon'))

#         # Categorical Inputs
#         soil = data.get('Soil')
#         crop = data.get('Crop')

#         soil_enc = label_encoders['Soil'].transform([soil])[0]
#         crop_enc = label_encoders['Crop'].transform([crop])[0]

#         # Feature Vector
#         features = np.array([[
#             temp, moisture, ph, nitrogen, phosphorous,
#             potassium, carbon, soil_enc, crop_enc
#         ]])

#         # Prediction
#         encoded_output = model.predict(features)[0]
#         fertilizer_name = FERTILIZER_MAPPING.get(encoded_output, "Unknown Fertilizer")

#         return jsonify({
#             'success': True,
#             'fertilizer_recommendation': fertilizer_name
#         })

#     except ValueError:
#         return jsonify({'error': 'Invalid input. Ensure numbers are correct.'}), 400
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return jsonify({'error': f"Unexpected error: {e}"}), 500


# # -----------------------------
# # START FLASK SERVER
# # -----------------------------
# if __name__ == '__main__':
#     load_assets()

#     print("🚀 Server starting...")
#     print("🌐 Open your browser and visit:")
#     print("👉 http://127.0.0.1:5000\n")

#     app.run(
#         debug=True,
#         host="127.0.0.1",   # Localhost
#         port=5000          # Clickable URL
#     )
