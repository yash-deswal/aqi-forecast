"""
AQI Prediction Flask App
========================
This app loads a pre-trained machine learning model and exposes a /predict
endpoint. The model was trained with one-hot encoded city columns, so we
manually construct that encoding on every incoming request.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -------------------------------------------------------------------
# Load the pre-trained model
# joblib is used because sklearn models are typically saved with it.
# -------------------------------------------------------------------
MODEL_PATH = os.path.join("model", "aqi_model_small.pkl")
model = joblib.load(MODEL_PATH)

# -------------------------------------------------------------------
# MODEL_COLUMNS — the EXACT list of feature columns the model expects,
# in the EXACT order they were present during training.
#
# The first 7 are numeric features.
# Everything starting with "City_" is a one-hot encoded binary column.
# Only ONE of those City_ columns will be set to 1 per prediction;
# all others stay 0.
# -------------------------------------------------------------------
MODEL_COLUMNS = [
    'day', 'month', 'year', 'day_of_week',
    'AQI_lag1', 'AQI_lag2', 'AQI_lag3',
    'City_Agra', 'City_Ahmedabad', 'City_Ahmednagar', 'City_Aizawl',
    'City_Ajmer', 'City_Akola', 'City_Alwar', 'City_Amaravati',
    'City_Ambala', 'City_Amravati', 'City_Amritsar', 'City_Anantapur',
    'City_Angul', 'City_Ankleshwar', 'City_Araria', 'City_Ariyalur',
    'City_Arrah', 'City_Asanol', 'City_Asansol', 'City_Aurangabad',
    'City_Aurangabad\n(Maharashtra)', 'City_Aurangabad (Bihar)',
    'City_Baddi', 'City_Badlapur', 'City_Bagalkot', 'City_Baghpat',
    'City_Bahadurgarh', 'City_Balasore', 'City_Ballabgarh',
    'City_Banswara', 'City_Baran', 'City_Barbil', 'City_Bareilly',
    'City_Baripada', 'City_Barmer', 'City_Barrackpore', 'City_Bathinda',
    'City_Begusarai', 'City_Belapur', 'City_Belgaum', 'City_Bengaluru',
    'City_Bettiah', 'City_Bhagalpur', 'City_Bharatpur', 'City_Bhilai',
    'City_Bhilwara', 'City_Bhiwadi', 'City_Bhiwandi', 'City_Bhiwani',
    'City_Bhopal', 'City_Bhubaneswar', 'City_Bidar', 'City_Bihar Sharif',
    'City_Bikaner', 'City_Bilaspur', 'City_Bileipada', 'City_Boisar',
    'City_Brajrajnagar', 'City_Bulandshahr', 'City_Bundi', 'City_Buxar',
    'City_Byasanagar', 'City_Byrnihat', 'City_Chamarajanagar',
    'City_Chandigarh', 'City_Chandrapur', 'City_Charkhi Dadri',
    'City_Chengalpattu', 'City_Chennai', 'City_Chhal', 'City_Chhapra',
    'City_Chikkaballapur', 'City_Chikkamagaluru', 'City_Chittoor',
    'City_Chittorgarh', 'City_Churu', 'City_Coimbatore', 'City_Cuddalore',
    'City_Cuttack', 'City_Damoh', 'City_Darbhanga', 'City_Dausa',
    'City_Davanagere', 'City_Dehradun', 'City_Delhi', 'City_Dewas',
    'City_Dhanbad', 'City_Dharuhera', 'City_Dharwad', 'City_Dholpur',
    'City_Dhule', 'City_Dindigul', 'City_Dungarpur', 'City_Durgapur',
    'City_Eloor', 'City_Ernakulam', 'City_Faridabad', 'City_Fatehabad',
    'City_Firozabad', 'City_Gadag', 'City_GandhiNagar', 'City_Gandhinagar',
    'City_Gangtok', 'City_Gaya', 'City_Ghaziabad', 'City_Gorakhpur',
    'City_Greater Noida', 'City_Greater_Noida', 'City_Gummidipoondi',
    'City_Gurugram', 'City_Guwahati', 'City_Gwalior', 'City_Hajipur',
    'City_Haldia', 'City_Hanumangarh', 'City_Hapur', 'City_Hassan',
    'City_Haveri', 'City_Hisar', 'City_Hosur', 'City_Howrah',
    'City_Hubballi', 'City_Hyderabad', 'City_Imphal', 'City_Indore',
    'City_Jabalpur', 'City_Jaipur', 'City_Jaisalmer', 'City_Jalandhar',
    'City_Jalgaon', 'City_Jalna', 'City_Jalore', 'City_Jhalawar',
    'City_Jhansi', 'City_Jhunjhunu', 'City_Jind', 'City_Jodhpur',
    'City_Jorapokhar', 'City_Kadapa', 'City_Kaithal', 'City_Kalaburagi',
    'City_Kalaburgi', 'City_Kalyan', 'City_Kanchipuram', 'City_Kannur',
    'City_Kanpur', 'City_Karauli', 'City_Karnal', 'City_Karur',
    'City_Karwar', 'City_Kashipur', 'City_Katihar', 'City_Katni',
    'City_Keonjhar', 'City_Khanna', 'City_Khurja', 'City_Kishanganj',
    'City_Kochi', 'City_Kohima', 'City_Kolar', 'City_Kolhapur',
    'City_Kolkata', 'City_Kollam', 'City_Koppal', 'City_Korba',
    'City_Kota', 'City_Kozhikode', 'City_Kunjemura', 'City_Kurukshetra',
    'City_Latur', 'City_Lucknow', 'City_Ludhiana', 'City_Madikeri',
    'City_Madurai', 'City_Mahad', 'City_Maihar', 'City_Malegaon',
    'City_Mandi Gobindgarh', 'City_Mandideep', 'City_Mandikhera',
    'City_Manesar', 'City_Mangalore', 'City_Manglore', 'City_Manguraha',
    'City_Medikeri', 'City_Meerut', 'City_Milupara', 'City_Mira-Bhayandar',
    'City_Moradabad', 'City_Motihari', 'City_Mumbai', 'City_Munger',
    'City_Muzaffarnagar', 'City_Muzaffarpur', 'City_Mysuru', 'City_Nagaon',
    'City_Nagapattinam', 'City_Nagaur', 'City_Nagpur', 'City_Naharlagun',
    'City_Nalbari', 'City_Namakkal', 'City_Nanded', 'City_Nandesari',
    'City_Narnaul', 'City_Nashik', 'City_Navi Mumbai', 'City_Nayagarh',
    'City_Noida', 'City_Ooty', 'City_Pali', 'City_Palkalaiperur',
    'City_Palwal', 'City_Panchgaon', 'City_Panchkula', 'City_Panipat',
    'City_Parbhani', 'City_Pathardih', 'City_Patiala', 'City_Patna',
    'City_Perundurai', 'City_Pimpri Chinchwad', 'City_Pimpri-Chinchwad',
    'City_Pithampur', 'City_Pratapgarh', 'City_Prayagraj', 'City_Puducherry',
    'City_Pudukottai', 'City_Pune', 'City_Purnia', 'City_Raichur',
    'City_Raipur', 'City_Rairangpur', 'City_Rajamahendravaram',
    'City_Rajgir', 'City_Rajsamand', 'City_Ramanagara',
    'City_Ramanathapuram', 'City_Ranipet', 'City_Ratlam', 'City_Rishikesh',
    'City_Rohtak', 'City_Rourkela', 'City_Rupnagar', 'City_Sagar',
    'City_Saharsa', 'City_Salem', 'City_Samastipur', 'City_Sangli',
    'City_Sasaram', 'City_Satna', 'City_Sawai Madhopur', 'City_Shillong',
    'City_Shivamogga', 'City_Sikar', 'City_Silchar', 'City_Siliguri',
    'City_Singrauli', 'City_Sirohi', 'City_Sirsa', 'City_Sivasagar',
    'City_Siwan', 'City_Solapur', 'City_Sonipat', 'City_Sri Ganganagar',
    'City_Sri Vijaya Puram', 'City_Srinagar', 'City_Suakati', 'City_Surat',
    'City_Talcher', 'City_Tensa', 'City_Thane', 'City_Thanjavur',
    'City_Thiruvananthapuram', 'City_Thoothukudi', 'City_Thrissur',
    'City_Tiruchirappalli', 'City_Tirumala', 'City_Tirunelveli',
    'City_Tirupati', 'City_Tiruppur', 'City_Tirupur', 'City_Tonk',
    'City_Tumakuru', 'City_Tumidih', 'City_Udaipur', 'City_Udupi',
    'City_Ujjain', 'City_Ulhasnagar', 'City_Vapi', 'City_Varanasi',
    'City_Vatva', 'City_Vellore', 'City_Vijayapura', 'City_Vijayawada',
    'City_Virar', 'City_Virudhunagar', 'City_Visakhapatnam',
    'City_Vrindavan', 'City_Yadgir', 'City_Yamuna Nagar',
    'City_Yamunanagar', 'City_vellore'
]


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts JSON with city + date/lag features, returns predicted AQI.

    One-hot encoding logic:
    - All City_* columns are initialised to 0.
    - We build the column name as  "City_" + city_name  (e.g. "City_Delhi").
    - If that column exists in MODEL_COLUMNS, we set it to 1.
    - This mirrors exactly what pd.get_dummies() produced during training.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    # --- Step 1: Create a dict with every model column initialised to 0 ---
    input_dict = {col: 0 for col in MODEL_COLUMNS}

    # --- Step 2: Fill in the numeric features from the request ---
    numeric_fields = ['day', 'month', 'year', 'day_of_week',
                      'AQI_lag1', 'AQI_lag2', 'AQI_lag3']
    for field in numeric_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
        input_dict[field] = data[field]

    # --- Step 3: One-hot encode the city ---
    # The training dataset used pd.get_dummies() which created columns like
    # "City_Delhi", "City_Mumbai", etc.  We replicate that here manually.
    city = data.get("city", "").strip()
    city_col = f"City_{city}"   # e.g. "City_Delhi"

    if city_col not in MODEL_COLUMNS:
        return jsonify({
            "error": f"City '{city}' not recognized. "
                     f"Expected column '{city_col}' not found in model."
        }), 400

    input_dict[city_col] = 1   # set the matching city column to 1

    # --- Step 4: Build DataFrame in EXACT column order ---
    # Column order MUST match training order; we enforce this explicitly.
    input_df = pd.DataFrame([input_dict], columns=MODEL_COLUMNS)

    # --- Step 5: Predict and return ---
    prediction = model.predict(input_df)[0]
    aqi_value = round(float(prediction), 2)

    return jsonify({"aqi": aqi_value})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
