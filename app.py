import streamlit as st
import joblib
import pandas as pd
import os
import base64

# Path to model and image
MODEL_PATH = 'output/best_model.joblib'
IMAGE_PATH = 'house.png'  # Ensure this image is in your project folder

# Page configuration (must be the first Streamlit command)
st.set_page_config(page_title='House Price Predictor', layout='centered', page_icon=":house:")

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    .header img {
        width: 50px;
        margin-right: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stNumberInput {
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with image and icon
st.markdown(
    f"""
    <div class="header">
        <img src="data:image/png;base64,{base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode()}" alt="House Icon">
        <h1 style="color: #2c3e50;">House Price Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.write('Enter house features to predict estimated price.')

# Load model
@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f'Could not load model: {e}')
        return None

model = load_model()

# Input form with columns for better layout
with st.form('input_form'):
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.header('Input Features')

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input('Area (sqft)', min_value=0.0, value=1000.0)
        bedrooms = st.number_input('Bedrooms', min_value=0, max_value=20, value=3)
        bathrooms = st.number_input('Bathrooms', min_value=0, max_value=10, value=2)
        stories = st.number_input('Stories', min_value=1, max_value=10, value=1)
        parking = st.number_input('Parking', min_value=0, max_value=10, value=0)

    with col2:
        mainroad = st.selectbox('Main Road', options=['yes', 'no'])
        guestroom = st.selectbox('Guest Room', options=['yes', 'no'])
        basement = st.selectbox('Basement', options=['yes', 'no'])
        hotwaterheating = st.selectbox('Hot Water Heating', options=['yes', 'no'])
        airconditioning = st.selectbox('Air Conditioning', options=['yes', 'no'])
        prefarea = st.selectbox('Preferred Area', options=['yes', 'no'])
        furnishingstatus = st.selectbox('Furnishing Status', options=['furnished', 'semi-furnished', 'unfurnished'])
        location = st.selectbox('Location', options=[
            'Kharghar', 'Sector-13 Kharghar', 'Sector 18 Kharghar', 'Sector 20 Kharghar',
            'Sector 15 Kharghar', 'Dombivali', 'Churchgate', 'Prabhadevi', 'Jogeshwari West',
            'Kalyan East', 'Malad East', 'Kandivali East', 'Taloja', 'Lower Parel', 'Thane',
            'Jankalyan Nagar', 'Badlapur', 'Ambernath West', 'Vakola', 'Ambarnath', 'Badlapur East',
            'Mira Bhayandar', 'Malad West', 'Goregaon West', 'Vasai east', 'Nahur', 'Badlapur West',
            'Thane West', 'Panvel', 'Kalyan', 'Juhu', 'Naigaon East', 'Mira Road East', 'Ulwe',
            'Bandra East', 'Dronagiri', 'Nerul', 'Karanjade', 'Sanpada', 'Sector-18 Ulwe',
            'Sector-8 Ulwe', 'Sector-3 Ulwe', 'Sector 17 Ulwe', 'Sector 21 Kharghar',
            'Sector 12 Kharghar', 'Kamothe', 'Sector 17 Ulwe', 'Kamothe Sector 16', 'ULWE SECTOR 19',
            'Sector 23 Ulwe', 'Bandra West', 'Pali Hill', '15th Road', 'Palghar', 'Chembur',
            'Khar', 'Almeida Park', 'Santacruz West', 'Vivek Vidyalaya Marg', 'Vasai',
            'Nala Sopara', 'Bhiwandi', 'Goregaon East', 'Kandivali West', 'Wadala',
            'Worli South Mumbai', 'Andheri East', 'Breach Candy', 'Worli', 'Ghatkopar',
            'Dadar East', 'Dahisar', 'Asangaon', 'Koparkhairane Station Road'
        ])

    submitted = st.form_submit_button('Predict')

    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    if model is None:
        st.error('Model not loaded. Run the notebook to generate the model at "./output/best_model.joblib"')
    else:
        input_df = pd.DataFrame([{
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': mainroad,
            'guestroom': guestroom,
            'basement': basement,
            'hotwaterheating': hotwaterheating,
            'airconditioning': airconditioning,
            'parking': parking,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus,
            'Location': location
        }])
        try:
            pred = model.predict(input_df)[0]
            st.success(f'Predicted price: ₹{pred:,.2f}')
        except Exception as e:
            st.error(f'Prediction failed: {e}')

st.info('Tip: Ensure input values match the dataset feature ranges and categories.')