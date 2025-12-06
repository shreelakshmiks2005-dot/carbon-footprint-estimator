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
    background: linear-gradient(145deg, #0D2F3A, #0B4748);
    font-family: 'Inter', sans-serif;
    color: #E6F4F1 !important;
}

/* ---------- HEADINGS ---------- */
h1, h2, h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    color: #E8FFFE !important;
    text-align: center;
}

/* ---------- TEXT ---------- */
p, label, span {
    color: #CFE8E6 !important;
}

/* ---------- LINK ---------- */
a {
    color: #A8E6DF !important;
    text-decoration: none !important;
    font-weight: 500;
}
a:hover {
    color: #74DACC !important;
}

/* ---------- SECTION TITLE ---------- */
.section-title {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* ---------- CARDS ---------- */
.card, .metric-card, .result-card, .recommendation-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(120, 200, 190, 0.18);
    backdrop-filter: blur(10px);
    box-shadow: 0px 6px 16px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}
.card:hover, .recommendation-card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 10px 24px rgba(0,0,0,0.25);
}

/* ---------- ICON COLORS ---------- */
.icon-green  { color: #47E2A1 !important; }
.icon-gold   { color: #F4D57C !important; }
.icon-blue   { color: #42C6E8 !important; }

/* ---------- BUTTONS ---------- */
.stButton>button {
    background: linear-gradient(135deg, #27A7B5, #34D1A1);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 10px 22px;
    font-size: 16px;
    font-weight: 600;
    transition: 0.25s ease;
}
.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(135deg, #1F9BA9, #2BC790);
}

/* ---------- INPUT FIELDS ---------- */
input, select, textarea, .stTextInput>div>div>input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,0.08) !important;
    color: #E6F4F1 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(100,180,170,0.2) !important;
}
input:focus {
    border-color: #40CFC4 !important;
}

/* ---------- FOOTER ---------- */
.footer {
    margin-top: 40px;
    text-align: center;
    color: #A8D9D4;
    font-size: 14px;
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
    try:
        preprocessor = DataPreprocessor.load('models/preprocessor.pkl')
        model = CarbonFootprintModel.load('models/xgboost_model.pkl')
        return preprocessor, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# PAGE 1: HOME PAGE
def show_home_page():
    st.markdown("""
        <div class="hero-section">
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
                prediction = model.predict(user_processed)[0]
                
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
    
    prediction = st.session_state.user_prediction
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
        <p>Model performance comparison (Linear, Random Forest, XGBoost)</p>
    """, unsafe_allow_html=True)

    # Model comparison: load saved metrics and display table + chart
    try:
        import json
        results_path = Path('models/model_results.json')
        if results_path.exists():
            with open(results_path, 'r') as fh:
                results = json.load(fh)

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
            st.dataframe(metrics_df.style.format({
                'MAE': '{:.1f}', 'RMSE': '{:.1f}', 'R2': '{:.3f}', 'MAPE (%)': '{:.2f}'
            }))

            # Bar chart for R2 and MAE
            chart_df = metrics_df.reset_index()
            fig_metrics = px.bar(
                chart_df.melt(id_vars='Model', value_vars=['R2', 'MAE']),
                x='Model', y='value', color='variable', barmode='group',
                title='Model Comparison (R2 and MAE)'
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
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
