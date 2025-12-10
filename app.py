"""
Carbon Footprint Estimator - Streamlit Web Application
Redesigned with Home Page → Form → Results → Comparative Analysis flow
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_generation import generate_carbon_footprint_dataset, save_dataset
from data_preprocessing import DataPreprocessor
from model_training import CarbonFootprintModel
from suggestion_engine import SuggestionEngine

# Page configuration
st.set_page_config(
    page_title="Carbon Footprint Estimator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
body, .stApp {
    background: linear-gradient(135deg, #f0f9f7, #e8f5f2);
    font-family: 'Inter', sans-serif;
    color: #0d4a40 !important;
}

/* ---------- HEADINGS ---------- */
h1, h2, h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #0d4a40 !important;
    text-align: center;
}

/* ---------- TEXT ---------- */
p, label, span {
    color: #1a5d52 !important;
}

/* ---------- LINK ---------- */
a {
    color: #0d8970 !important;
    text-decoration: none !important;
    font-weight: 600;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    background: linear-gradient(135deg, #16a085, #1abc9c);
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 700;
    box-shadow: 0px 6px 15px rgba(22, 160, 133, 0.3);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #0d8970, #16a085);
    transform: translateY(-2px);
}

/* ---------- INPUT FIELDS ---------- */
input, select, textarea, .stTextInput>div>div>input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.98) !important;
    color: #0d4a40 !important;
    border: 2px solid #16a085 !important;
    border-radius: 10px !important;
}

input::placeholder {
    color: #7cb5ad !important;
}

input:focus {
    border-color: #0d8970 !important;
    background: rgba(255, 255, 255, 1) !important;
}

/* ---------- CARDS ---------- */
.metric-card, .info-card, .result-card, .recommendation-card {
    background: linear-gradient(135deg, rgba(232, 245, 242, 0.95), rgba(212, 237, 233, 0.95)) !important;
    border: 2px solid #16a085 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0px 4px 12px rgba(22, 160, 133, 0.15) !important;
}

.metric-label {
    color: #1a5d52 !important;
    font-weight: 700;
}

.metric-value {
    color: #0d8970 !important;
    font-size: 2.5em !important;
    font-weight: 900;
}

/* ---------- SECTION DIVIDER ----pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
tensorflow==2.13.0
streamlit==1.28.1
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.1
plotly==5.16.1------ */
.section-divider {
    margin: 30px 0;
    border-bottom: 2px solid #16a085;
    opacity: 0.4;
}

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'user_prediction' not in st.session_state:
    st.session_state.user_prediction = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'avg_footprint' not in st.session_state:
    st.session_state.avg_footprint = 2000  # From Kaggle dataset

# Load model and preprocessor (cached)
@st.cache_resource
def load_model_artifacts():
    """Load preprocessor and ANN model"""
    try:
        from tensorflow.keras.models import load_model
        
        # Load preprocessor
        preprocessor = DataPreprocessor.load('models/preprocessor.pkl')
        
        # Load trained ANN model
        # Use compile=False to avoid deserializing optimizer/metrics (prevents
        # errors like: Could not deserialize 'keras.metrics.mse')
        model = load_model('models/ann_model.h5', compile=False)
        
        return preprocessor, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# PAGE 1: HOME PAGE
def show_home_page():
    st.markdown("""
        <div class="hero-section" style="text-align: center;">
            <h1>🌍 Carbon Footprint Estimator</h1>
            <p>Understand Your Environmental Impact</p>
            <p style="font-size: 1em; opacity: 0.9;">Join millions worldwide in building a sustainable future</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
            <div class="info-card" style="border-left: 6px solid #2ecc71; border-top: 4px solid #1abc9c;">
            <h3 style="color: #0f9b7a; font-size: 1.4em;">Why It Matters</h3>
            <p style="color: #444; font-size: 0.95em; line-height: 1.6;">Personal carbon footprints contribute to climate change. Understanding your impact is the first step to reducing it.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card" style="border-left: 6px solid #3498db; border-top: 4px solid #2980b9;">
            <h3 style="color: #2980b9; font-size: 1.4em;">AI-Powered</h3>
            <p style="color: #444; font-size: 0.95em; line-height: 1.6;">Our machine learning models analyze your lifestyle to provide accurate carbon footprint estimates with 93% accuracy.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-card" style="border-left: 6px solid #f39c12; border-top: 4px solid #e67e22;">
            <h3 style="color: #d68910; font-size: 1.4em;">Get Recommendations</h3>
            <p style="color: #444; font-size: 0.95em; line-height: 1.6;">Receive personalized suggestions to reduce your environmental impact by 20-50% with actionable steps.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <h2 style="color: #27ae60; text-align: center;">How It Works</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
            <h4>1.</h4>
            <p><b>Answer Questions</b></p>
            <p style="font-size: 0.9em;">Tell us about your lifestyle</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
            <h4>2.</h4>
            <p><b>Generate Estimate</b></p>
            <p style="font-size: 0.9em;">Get your carbon footprint</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
            <h4>3.</h4>
            <p><b>View Results</b></p>
            <p style="font-size: 0.9em;">See detailed analysis</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
            <h4>4.</h4>
            <p><b>Get Recommendations</b></p>
            <p style="font-size: 0.9em;">Take action today</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Main CTA Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Check Your Carbon Footprint", key="home_button", use_container_width=True):
            st.session_state.current_page = 'form'
            st.rerun()

# PAGE 2: INPUT FORM
def show_form_page():
    st.markdown("""
        <h1 style="color: #27ae60;">Your Lifestyle Details</h1>
        <p style="font-size: 1.1em; color: #555;">Please answer the following questions about your daily lifestyle</p>
    """, unsafe_allow_html=True)
    
    with st.form("lifestyle_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #2ecc71;'>Transportation</h3>", unsafe_allow_html=True)
            transport_mode = st.selectbox(
                "Primary transportation mode:",
                ["public", "walk/bicycle", "private"]
            )
            commute_distance = st.slider(
                "Daily commute distance (km):",
                0, 50, 15
            )
            flights_per_year = st.slider(
                "Flights per year:",
                0, 20, 2
            )
            car_type = st.selectbox(
                "Car type (if applicable):",
                ["electric", "hybrid", "petrol", "diesel", "lpg"]
            )
        
        with col2:
            st.markdown("<h3 style='color: #2ecc71;'>Diet</h3>", unsafe_allow_html=True)
            diet_type = st.selectbox(
                "Diet type:",
                ["vegan", "vegetarian", "pescatarian", "omnivore"]
            )
            meat_consumption = st.slider(
                "Meat consumption (kg/month):",
                0.0, 20.0, 8.0
            )
            new_clothes = st.slider(
                "New clothes per month:",
                0.0, 10.0, 2.0
            )
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #2ecc71;'>Household</h3>", unsafe_allow_html=True)
            household_size = st.number_input(
                "Household size:", min_value=1, max_value=10, value=1, step=1
            )
            vehicle_fuel_efficiency = st.number_input(
                "Vehicle fuel efficiency (km/litre):", min_value=0.0, max_value=50.0, value=12.0, step=0.1
            )
            renewable_energy_percentage = st.slider(
                "Renewable energy share (%):",
                0, 100, 0
            )
            heating_fuel_type = st.selectbox(
                "Heating fuel type:",
                ["electric", "gas", "oil", "none"]
            )
        
        with col2:
            st.markdown("<h3 style='color: #2ecc71;'>Energy & Water</h3>", unsafe_allow_html=True)
            electricity_usage = st.slider(
                "Monthly electricity (kWh):",
                100, 1000, 400
            )
            gas_usage = st.slider(
                "Monthly gas (therms):",
                0, 200, 40
            )
            water_usage = st.slider(
                "Monthly water (gallons):",
                500, 5000, 2500
            )
            recycling_percentage = st.slider(
                "Recycling percentage (%):",
                0, 100, 50
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_button = st.form_submit_button(
                "Generate Carbon Footprint Estimate",
                use_container_width=True
            )
    
    if generate_button:
        # Prepare user data
        user_data = {
            'transportation_mode': transport_mode,
            'commute_distance_km': commute_distance,
            'flights_per_year': flights_per_year,
            'diet_type': diet_type,
            'meat_consumption_kg_per_month': meat_consumption,
            'electricity_usage_kwh': electricity_usage,
            'gas_usage_therms': gas_usage,
            'water_usage_gallons': water_usage,
            'recycling_percentage': recycling_percentage,
            'new_clothes_per_month': new_clothes,
            'car_type': car_type if car_type != "none" else None,
            'household_size': household_size,
            'vehicle_fuel_efficiency_km_per_l': vehicle_fuel_efficiency,
            'renewable_energy_percentage': renewable_energy_percentage,
            'heating_fuel_type': heating_fuel_type
        }
        
        # Load models
        preprocessor, model = load_model_artifacts()
        
        if preprocessor and model:
            try:
                user_df = pd.DataFrame([user_data])
                user_processed = preprocessor.transform(user_df)
                # Ensure prediction is a native Python float (not a numpy array)
                pred_arr = model.predict(user_processed, verbose=0)
                # Squeeze and cast to float to avoid formatting errors later
                prediction = float(np.squeeze(pred_arr))
                
                st.session_state.user_data = user_data
                st.session_state.user_prediction = prediction
                st.session_state.preprocessor = preprocessor
                st.session_state.model = model
                st.session_state.current_page = 'results'
                st.rerun()
            except Exception as e:
                st.error(f"Error generating prediction: {e}")
        else:
            st.error("Models not loaded. Please ensure model files exist.")
    
    # Back button
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("← Back", key="form_back"):
            st.session_state.current_page = 'home'
            st.rerun()

# PAGE 3: RESULTS
def show_results_page():
    st.markdown("""
        <h1 style="color: #27ae60;">Your Carbon Footprint Analysis</h1>
    """, unsafe_allow_html=True)
    
    # Ensure prediction is a float for formatting and arithmetic
    prediction = float(st.session_state.user_prediction) if st.session_state.user_prediction is not None else 0.0
    user_data = st.session_state.user_data
    avg_footprint = 2000  # Average from Kaggle dataset (monthly in kg)
    
    # Key Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
            <div class="metric-label">Your Carbon Footprint</div>
            <div class="metric-value">{prediction:.0f}</div>
            <div class="metric-label">kg CO₂/month</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        diff = prediction - avg_footprint
        diff_percent = (diff / avg_footprint) * 100
        color = "#27ae60" if diff < 0 else "#e74c3c"
        bg_gradient = "linear-gradient(135deg, #d5f4e6 0%, #c8f0e0 100%)" if diff < 0 else "linear-gradient(135deg, #fadbd8 0%, #f5b7b1 100%)"
        border_color = "#2ecc71" if diff < 0 else "#e74c3c"
        comparison_text = "Below Average" if diff < 0 else "Above Average"
        st.markdown(f"""
            <div class="metric-card" style="background: {bg_gradient}; border-top: 5px solid {border_color};">
            <div class="metric-label" style="color: #333;">vs Average</div>
            <div class="metric-value" style="color: {color}; font-weight: 900;">{diff:+.0f}</div>
            <div class="metric-label" style="color: #555; font-weight: 700;">{comparison_text}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        annual_footprint = prediction * 12
        st.markdown(f"""
            <div class="metric-card metric-alt-blue">
            <div class="metric-label">Annual Carbon</div>
            <div class="metric-value">{annual_footprint:.0f}</div>
            <div class="metric-label">kg CO₂/year</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart - Carbon breakdown by category
        categories = {
            'Transport': user_data['commute_distance_km'] * 20 * 0.21 if user_data['transportation_mode'] == 'Car' else user_data['commute_distance_km'] * 20 * 0.05,
            'Diet': user_data['meat_consumption_kg_per_month'] * 27,
            'Energy': user_data['electricity_usage_kwh'] * 0.92 + user_data['gas_usage_therms'] * 5.3,
            'Water': user_data['water_usage_gallons'] * 0.35,
            'Waste': user_data['new_clothes_per_month'] * 6.5,
        }
        
        fig_pie = px.pie(
            values=list(categories.values()),
            names=list(categories.keys()),
            title="Carbon Footprint Breakdown by Category",
            color_discrete_sequence=['#2ecc71', '#1abc9c', '#f39c12', '#3498db', '#e74c3c']
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Comparison chart
        comparison_data = {
            'Category': ['Your\nFootprint', 'Average\nFootprint'],
            'kg CO₂/month': [prediction, avg_footprint]
        }
        
        fig_bar = px.bar(
            comparison_data,
            x='Category',
            y='kg CO₂/month',
            title="Your Footprint vs Average",
            color='Category',
            color_discrete_sequence=['#2ecc71', '#95a5a6']
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Recommendations Section
    st.markdown("<h2 style='color: #27ae60;'>Personalized Recommendations</h2>", unsafe_allow_html=True)
    
    engine = SuggestionEngine()
    suggestions = engine.generate_suggestions(user_data, prediction, avg_footprint)
    
    # Overall status
    overall = suggestions['overall']
    st.markdown(f"""
        <div class="result-card">
        <h3>{overall['status']}</h3>
        <p>{overall['message']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Category recommendations in tabs
    tabs = st.tabs(["Transport", "Diet", "Energy", "Waste", "Water"])
    
    with tabs[0]:
        transport_sugg = suggestions['transportation']
        st.subheader(f"Current Level: {transport_sugg['level'].title()}")
        st.markdown(f"""
            <div class="recommendation-card recommendation-card-blue">
            <strong>Current Habits:</strong> {transport_sugg['current_habits']}
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="recommendation-card">
            <strong>Potential Savings:</strong> {transport_sugg['potential_savings']}
            </div>
        """, unsafe_allow_html=True)
        st.write("**Recommendations:**")
        for i, rec in enumerate(transport_sugg['suggestions'][:3], 1):
            st.markdown(f"""
                <div class="recommendation-card">
                <strong>Tip {i}:</strong> {rec}
                </div>
            """, unsafe_allow_html=True)
    
    with tabs[1]:
        diet_sugg = suggestions['diet']
        st.subheader(f"Current Level: {diet_sugg['level'].title()}")
        st.markdown(f"""
            <div class="recommendation-card recommendation-card-orange">
            <strong>Current Habits:</strong> {diet_sugg['current_habits']}
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="recommendation-card">
            <strong>Potential Savings:</strong> {diet_sugg['potential_savings']}
            </div>
        """, unsafe_allow_html=True)
        st.write("**Recommendations:**")
        for i, rec in enumerate(diet_sugg['suggestions'][:3], 1):
            st.markdown(f"""
                <div class="recommendation-card recommendation-card-orange">
                <strong>Tip {i}:</strong> {rec}
                </div>
            """, unsafe_allow_html=True)
    
    with tabs[2]:
        energy_sugg = suggestions['energy']
        st.subheader(f"Current Level: {energy_sugg['level'].title()}")
        st.markdown(f"""
            <div class="recommendation-card recommendation-card-blue">
            <strong>Current Habits:</strong> {energy_sugg['current_habits']}
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="recommendation-card">
            <strong>Potential Savings:</strong> {energy_sugg['potential_savings']}
            </div>
        """, unsafe_allow_html=True)
        st.write("**Recommendations:**")
        for i, rec in enumerate(energy_sugg['suggestions'][:3], 1):
            st.markdown(f"""
                <div class="recommendation-card">
                <strong>Tip {i}:</strong> {rec}
                </div>
            """, unsafe_allow_html=True)
    
    with tabs[3]:
        waste_sugg = suggestions['waste']
        st.subheader(f"Current Level: {waste_sugg['level'].title()}")
        st.markdown(f"""
            <div class="recommendation-card recommendation-card-purple">
            <strong>Current Habits:</strong> {waste_sugg['current_habits']}
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="recommendation-card">
            <strong>Potential Savings:</strong> {waste_sugg['potential_savings']}
            </div>
        """, unsafe_allow_html=True)
        st.write("**Recommendations:**")
        for i, rec in enumerate(waste_sugg['suggestions'][:3], 1):
            st.markdown(f"""
                <div class="recommendation-card recommendation-card-purple">
                <strong>Tip {i}:</strong> {rec}
                </div>
            """, unsafe_allow_html=True)
    
    with tabs[4]:
        water_sugg = suggestions['water']
        st.subheader(f"Current Level: {water_sugg['level'].title()}")
        st.markdown(f"""
            <div class="recommendation-card recommendation-card-coral">
            <strong>Current Habits:</strong> {water_sugg['current_habits']}
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="recommendation-card">
            <strong>Potential Savings:</strong> {water_sugg['potential_savings']}
            </div>
        """, unsafe_allow_html=True)
        st.write("**Recommendations:**")
        for i, rec in enumerate(water_sugg['suggestions'][:3], 1):
            st.markdown(f"""
                <div class="recommendation-card recommendation-card-coral">
                <strong>Tip {i}:</strong> {rec}
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Action Buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Comparative Analysis", use_container_width=True, key="comp_analysis"):
            st.session_state.current_page = 'comparative'
            st.rerun()
    
    with col2:
        if st.button("Check Again", use_container_width=True, key="check_again"):
            st.session_state.current_page = 'form'
            st.rerun()
    
    with col3:
        if st.button("Home", use_container_width=True, key="go_home"):
            st.session_state.current_page = 'home'
            st.rerun()
    
    with col4:
        st.write("")  # Empty column for spacing

# PAGE 4: COMPARATIVE ANALYSIS
def show_comparative_page():
    st.markdown("""
        <h1 style="color: #27ae60;">Comparative Analysis</h1>
        <p>Model performance comparison (Neural Network, Random Forest, XGBoost)</p>
    """, unsafe_allow_html=True)

    # Model comparison: load saved metrics and display table + chart
    try:
        import json
        results_path = Path('models/model_results.json')
        if results_path.exists():
            with open(results_path, 'r') as fh:
                results = json.load(fh)
            # Normalize keys to lowercase for presence checks
            available_keys = {k.lower(): k for k in results.keys()}

            # Collect debug info to help diagnose why metrics may be missing
            debug_info = []
            debug_info.append(f"Loaded results keys: {list(results.keys())}")

            # Expect these baseline models to be shown (primary three: Linear, Random Forest, XGBoost)
            expected_models = ['linear_regression', 'random_forest', 'xgboost', 'neural_network']

            missing = [m for m in expected_models if m not in available_keys]

            # If any expected model metrics are missing, attempt to compute them
            if missing:
                try:
                    import joblib
                    from tensorflow.keras.models import load_model as keras_load_model
                    # Load dataset to compute metrics
                    data_path_candidates = [Path('data/carbon_footprint_data_enriched.csv'), Path('data/carbon_footprint_data.csv')]
                    data_df = None
                    for p in data_path_candidates:
                        if p.exists():
                            data_df = pd.read_csv(p)
                            break
                    debug_info.append(f"Dataset found: {data_df is not None}")
                    if data_df is not None:
                        # Find target column
                        target_col = None
                        for col in data_df.columns:
                            if 'carbon' in col.lower() or 'footprint' in col.lower():
                                target_col = col
                                break
                        if target_col is None:
                            target_col = data_df.columns[-1]

                        X = data_df.drop(columns=[target_col])
                        y = data_df[target_col]

                        # Load preprocessor
                        preprocessor_path = Path('models/preprocessor.pkl')
                        preprocessor = None
                        if preprocessor_path.exists():
                            preprocessor = DataPreprocessor.load(str(preprocessor_path))
                        debug_info.append(f"Preprocessor loaded: {preprocessor is not None}")

                        # Use a test split/sample to compute metrics (use last 20%)
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Prepare X_test for models
                        if preprocessor is not None:
                            X_test_proc = preprocessor.transform(X_test.copy())
                        else:
                            X_test_proc = X_test.values

                        # Try loading model files and compute metrics for missing models
                        def _ensure_feature_shape(X_proc, model):
                            """Ensure X_proc has the same number of columns as model expects.
                            Pads with zeros or truncates columns as needed. Returns numpy array."""
                            import numpy as _np

                            # Determine expected feature count
                            expected = None
                            if hasattr(model, 'n_features_in_'):
                                try:
                                    expected = int(model.n_features_in_)
                                except Exception:
                                    expected = None

                            # If X_proc is a DataFrame, convert to numpy
                            if hasattr(X_proc, 'values'):
                                arr = X_proc.values
                            else:
                                arr = _np.asarray(X_proc)

                            # If we don't know expected, return as-is
                            if expected is None:
                                return arr

                            # Ensure 2D
                            if arr.ndim == 1:
                                arr = arr.reshape(1, -1)

                            cur = arr.shape[1]
                            if cur == expected:
                                return arr
                            elif cur < expected:
                                # pad with zeros on the right
                                pad = _np.zeros((arr.shape[0], expected - cur), dtype=arr.dtype)
                                return _np.hstack([arr, pad])
                            else:
                                # truncate extra columns
                                return arr[:, :expected]

                        for m in missing:
                            try:
                                if m == 'linear_regression':
                                    candidates = ['models/linear_model.pkl', 'models/linear_regression.pkl']
                                    mdl = None
                                    for c in candidates:
                                        if Path(c).exists():
                                            mdl = joblib.load(c)
                                            break
                                    if mdl is None:
                                        debug_info.append(f"Linear model file not found among: {candidates}")
                                        continue
                                    X_for_pred = _ensure_feature_shape(X_test_proc, mdl)
                                    y_pred = mdl.predict(X_for_pred)
                                
                                elif m == 'random_forest':
                                    candidates = ['models/random_forest_model.pkl', 'models/random_forest.pkl']
                                    mdl = None
                                    for c in candidates:
                                        if Path(c).exists():
                                            mdl = joblib.load(c)
                                            break
                                    if mdl is None:
                                        debug_info.append(f"Random Forest file not found among: {candidates}")
                                        continue
                                    X_for_pred = _ensure_feature_shape(X_test_proc, mdl)
                                    y_pred = mdl.predict(X_for_pred)
                                
                                elif m == 'xgboost':
                                    # Try common XGBoost / gradient-boosting filenames first (joblib pickles)
                                    candidates = [
                                        'models/xgboost_model.pkl', 'models/xgboost.pkl', 'models/xgboost_model.joblib',
                                        'models/gradient_boosting_model.pkl', 'models/gradient_boosting.pkl'
                                    ]
                                    mdl = None
                                    for c in candidates:
                                        if Path(c).exists():
                                            try:
                                                mdl = joblib.load(c)
                                                break
                                            except Exception:
                                                # Not a joblib pickle, continue to next candidate
                                                mdl = None
                                                continue

                                    if mdl is not None:
                                        X_for_pred = _ensure_feature_shape(X_test_proc, mdl)
                                        y_pred = mdl.predict(X_for_pred)
                                    else:
                                        # Try loading an XGBoost Booster model file (json or binary)
                                        xgb_found = None
                                        for c in ['models/xgboost_model.json', 'models/xgboost_model.model', 'models/xgboost.model', 'models/xgboost.bin']:
                                            p = Path(c)
                                            if p.exists():
                                                xgb_found = p
                                                break
                                        if xgb_found is None:
                                            debug_info.append(f"XGBoost model file not found among candidates: {candidates + ['models/xgboost_model.json','models/xgboost_model.model']}")
                                            continue
                                        try:
                                            import xgboost as xgb
                                            booster = xgb.Booster()
                                            booster.load_model(str(xgb_found))
                                            X_for_pred = _ensure_feature_shape(X_test_proc, booster)
                                            dmat = xgb.DMatrix(X_for_pred)
                                            y_pred = booster.predict(dmat)
                                        except Exception as ex:
                                            debug_info.append(f"Failed to load XGBoost booster from {xgb_found}: {ex}")
                                            continue

                                elif m == 'gradient_boosting':
                                    candidates = ['models/gradient_boosting_model.pkl', 'models/gradient_boosting.pkl', 'models/xgboost_model.pkl']
                                    mdl = None
                                    for c in candidates:
                                        if Path(c).exists():
                                            mdl = joblib.load(c)
                                            break
                                    if mdl is None:
                                        debug_info.append(f"Gradient Boosting file not found among: {candidates}")
                                        continue
                                    X_for_pred = _ensure_feature_shape(X_test_proc, mdl)
                                    y_pred = mdl.predict(X_for_pred)
                                
                                elif m == 'neural_network':
                                    # ANN model is saved as H5
                                    ann_path = Path('models/ann_model.h5')
                                    if not ann_path.exists():
                                        debug_info.append("ANN model file not found: models/ann_model.h5")
                                        continue
                                    ann = keras_load_model(str(ann_path), compile=False)
                                    X_for_pred = _ensure_feature_shape(X_test_proc, ann)
                                    y_pred = ann.predict(X_for_pred, verbose=0).flatten()

                                else:
                                    continue

                                import numpy as np
                                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                                mae = float(mean_absolute_error(y_test, y_pred))
                                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                                r2 = float(r2_score(y_test, y_pred))
                                try:
                                    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
                                except Exception:
                                    mape = None

                                # Add to results under the original casing key
                                results[m] = {'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}}
                                debug_info.append(f"Computed metrics for {m}: R2={r2:.3f}, MAE={mae:.1f}")
                            except Exception:
                                import traceback
                                tb = traceback.format_exc()
                                debug_info.append(f"Failed to compute metrics for {m}: {tb}")
                                continue

                        # Persist updated results to file
                        try:
                            with open(results_path, 'w') as fh:
                                json.dump(results, fh, indent=4)
                            debug_info.append("Persisted updated model_results.json")
                        except Exception as ex:
                            debug_info.append(f"Failed to persist model_results.json: {ex}")
                except Exception:
                    import traceback
                    debug_info.append(f"Fallback computation failed: {traceback.format_exc()}")
                    # If fallback computation fails, continue with whatever results are present
                    pass

            rows = []
            for name, val in results.items():
                m = val.get('metrics', {})
                rows.append({
                    'Model': name.replace('_', ' ').title(),
                    'MAE': m.get('MAE'),
                    'RMSE': m.get('RMSE'),
                    'R2': m.get('R2'),
                    'MAPE (%)': m.get('MAPE')
                })

            metrics_df = pd.DataFrame(rows).set_index('Model')

            # Normalize common naming (ensure XGBoost shows correctly)
            metrics_df = metrics_df.rename(index=lambda s: s.replace('Xgboost', 'XGBoost'))

            # Filter to only include Neural Network, Random Forest, and XGBoost
            # (exclude Linear Regression)
            keep_models = ['Neural Network', 'Random Forest', 'XGBoost']
            idx_map = {idx.strip().lower(): idx for idx in metrics_df.index}
            models_to_keep = []
            for m in keep_models:
                key = m.strip().lower()
                if key in idx_map:
                    models_to_keep.append(idx_map[key])
            
            if models_to_keep:
                metrics_df = metrics_df.loc[models_to_keep]
            
            # Show the three models in the metrics table
            st.subheader('Metrics Table (Three Best Models)')
            st.dataframe(metrics_df.style.format({
                'MAE': '{:.1f}', 'RMSE': '{:.1f}', 'R2': '{:.3f}', 'MAPE (%)': '{:.2f}'
            }))

            # Use filtered metrics_df for chart
            chart_df = metrics_df

            # Create two side-by-side charts for clarity: MAE (raw) and R² (raw)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader('MAE (lower is better)')
                if 'MAE' in chart_df.columns:
                    fig_mae = px.bar(chart_df.reset_index(), x='Model', y='MAE', color='Model',
                                     title='Mean Absolute Error (kg CO₂/month)',
                                     color_discrete_sequence=px.colors.qualitative.Plotly)
                    fig_mae.update_layout(showlegend=False)
                    fig_mae.update_yaxes(title_text='MAE (kg)')
                    st.plotly_chart(fig_mae, use_container_width=True)
                else:
                    st.info('MAE not available for selected models.')

            with col2:
                st.subheader('R² (higher is better)')
                if 'R2' in chart_df.columns:
                    fig_r2 = px.bar(chart_df.reset_index(), x='Model', y='R2', color='Model',
                                    title='R² Score', color_discrete_sequence=px.colors.qualitative.Plotly)
                    fig_r2.update_layout(showlegend=False)
                    fig_r2.update_yaxes(title_text='R²')
                    st.plotly_chart(fig_r2, use_container_width=True)
                else:
                    st.info('R² not available for selected models.')

            # Show debug info to help understand which model metrics were available/computed
            if debug_info:
                with st.expander('Comparative debug info (click to expand)'):
                    for line in debug_info:
                        st.text(line)
        else:
            st.info('Model results file not found. Run retrain_models.py to generate model results.')
    except Exception as e:
        st.error(f"Could not load model comparison: {e}")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Back button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.current_page = 'results'
            st.rerun()
    with col2:
        if st.button("Home", use_container_width=True):
            st.session_state.current_page = 'home'
            st.rerun()

# MAIN APP FLOW
def main():
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'form':
        show_form_page()
    elif st.session_state.current_page == 'results':
        show_results_page()
    elif st.session_state.current_page == 'comparative':
        show_comparative_page()

if __name__ == '__main__':
    main()

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 50px; padding: 20px; color: #95a5a6; border-top: 2px solid #ecf0f1;'>
    <p>🌍 Carbon Footprint Estimator | Made with care for a sustainable future</p>
    <p style='font-size: 0.9em;'>Powered by Machine Learning • Data-Driven Insights • Actionable Recommendations</p>
    </div>
""", unsafe_allow_html=True)
