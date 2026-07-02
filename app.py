from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    err = exc.errors()[0]
    field = ".".join(str(loc) for loc in err["loc"][1:]) # skip 'body'
    return JSONResponse(
        status_code=400,
        content={"error": f"Invalid or missing value for '{field}': {err['msg']}"}
    )

class PredictRequest(BaseModel):
    city: str
    day: int
    month: int
    year: int
    day_of_week: int
    AQI_lag1: float
    AQI_lag2: float
    AQI_lag3: float

MODEL_PATH = os.path.join("model", "aqi_model_small.pkl")
model = joblib.load(MODEL_PATH)

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

@app.get("/")
def index():
    """Serve the main HTML page."""
    return FileResponse("templates/index.html")

@app.post("/predict")
def predict(req: PredictRequest):
    input_dict = {col: 0 for col in MODEL_COLUMNS}

    input_dict['day'] = req.day
    input_dict['month'] = req.month
    input_dict['year'] = req.year
    input_dict['day_of_week'] = req.day_of_week
    input_dict['AQI_lag1'] = req.AQI_lag1
    input_dict['AQI_lag2'] = req.AQI_lag2
    input_dict['AQI_lag3'] = req.AQI_lag3

    city = req.city.strip()
    city_col = f"City_{city}"

    if city_col not in MODEL_COLUMNS:
        return JSONResponse(
            status_code=400,
            content={"error": f"City '{city}' not recognized. Expected column '{city_col}' not found in model."}
        )

    input_dict[city_col] = 1

    input_df = pd.DataFrame([input_dict], columns=MODEL_COLUMNS)

    prediction = model.predict(input_df)[0]
    aqi_value = round(float(prediction), 2)

    return {"aqi": aqi_value}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
