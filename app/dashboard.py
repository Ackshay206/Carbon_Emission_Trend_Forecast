import streamlit as st
import pandas as pd
from utils import load_model, forecast_future, LSTMModel,GRUModel
import plotly.graph_objs as go
import time
import sys
import os

# Add parent directory to path to find the src module
sys.path.append('..')

# Configure the page
st.set_page_config(
    page_title="Carbon Emission Forecasting Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom colors
PRIMARY_COLOR = "#1B5E20"  # Dark green
SECONDARY_COLOR = "#4CAF50"  # Medium green
TEXT_COLOR = "#263238"  # Dark gray
LIGHT_BG = "#F5F7F9"  # Very light gray/blue
CARD_BG = "#FFFFFF"  # White

# Apply custom CSS
st.markdown(f"""
<style>
    .stApp {{
        background-color: {LIGHT_BG};
    }}
    .main .block-container {{
        padding-top: 2rem;
    }}
    h1 {{
        color: {PRIMARY_COLOR} !important;
        font-weight: 600;
    }}
    h2, h3 {{
        color: {TEXT_COLOR};
        font-weight: 500;
    }}
    p, li {{
        color: {TEXT_COLOR};
    }}

    h1 {{

        font-family: Monaco, monospace !important;
    }}   
    .stButton>button {{
        background-color: {SECONDARY_COLOR};
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: {PRIMARY_COLOR};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {LIGHT_BG};
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: {TEXT_COLOR};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {SECONDARY_COLOR};
        color: white;
    }}
    .metric-card {{
        background-color: {CARD_BG};
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px;
    }}
    .metric-value {{
        font-size: 24px;
        font-weight: 600;
        color: {PRIMARY_COLOR};
    }}
    .metric-label {{
        font-size: 14px;
        color: {TEXT_COLOR};
        margin-top: 5px;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG};
        border-right: 1px solid #E0E0E0;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 2rem;
    }}
    section[data-testid="stSidebar"] h3 {{
        padding-left: 1rem;
        color: {PRIMARY_COLOR};
        font-weight: 600;
    }}
    div[data-testid="stExpander"] {{
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        background-color: {CARD_BG};
    }}
    div[data-testid="stMetricValue"] {{
        color: {PRIMARY_COLOR};
        font-weight: 600;
    }}
    .content-card {{
        background-color: {CARD_BG};
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color:{PRIMARY_COLOR}
    }}
    .success-box {{
        padding: 10px 15px;
        background-color: #E8F5E9;
        border-left: 5px solid {PRIMARY_COLOR};
        border-radius: 4px;
        margin-bottom: 1rem;
        color: {TEXT_COLOR};
    }}
    .coming-soon {{
        padding: 40px 20px;
        text-align: center;
        background-color: {CARD_BG};
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .tip-box {{
        background-color: #E8F5E9;
        border-left: 4px solid {PRIMARY_COLOR};
        padding: 10px 15px;
        border-radius: 4px;
        margin: 20px 0;
        color: {TEXT_COLOR};
    }}
    footer {{
        background-color: {CARD_BG};
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-top: 2rem;
        color: {TEXT_COLOR};
    }}
    .welcome-card {{
        background-color: {CARD_BG};
        color: {TEXT_COLOR};
        padding: 40px 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    .welcome-card h2 {{
        text-align: center;
        margin-bottom: 30px;
        color: {PRIMARY_COLOR};
    }}
    .welcome-card p {{
        font-size: 17px;
        margin-bottom: 25px;
        color: {TEXT_COLOR};
    }}
    .welcome-card h3 {{
        color: {TEXT_COLOR};
    }}
    .welcome-card ol {{
        font-size: 16px;
        margin-bottom: 25px;
        color: {TEXT_COLOR};
    }}
    .settings-card {{
        background-color: {CARD_BG};
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        color: {PRIMARY_COLOR}
    }}
    .nav-button {{
        margin-bottom: 0.5rem;
        width: 100%;
    }}

</style>
""", unsafe_allow_html=True)

# Add caching for session state initialization
@st.cache_resource
def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'page' not in st.session_state:
        st.session_state.page = "Forecasting"
    if 'last_model_choice' not in st.session_state:
        st.session_state.last_model_choice = None
    if 'last_emission_choice' not in st.session_state:
        st.session_state.last_emission_choice = None
    if 'last_region_choice' not in st.session_state:
        st.session_state.last_region_choice = None
    if 'last_forecast_years' not in st.session_state:
        st.session_state.last_forecast_years = None
    if 'cached_forecast_results' not in st.session_state:
        st.session_state.cached_forecast_results = None
    if 'last_execution_time' not in st.session_state:
        st.session_state.last_execution_time = None

# Initialize session state
initialize_session_state()

# Define page navigation functions
def navigate_to_forecasting():
    st.session_state.page = "Forecasting"

def navigate_to_statistics():
    st.session_state.page = "Statistics"

def navigate_to_clustering():
    st.session_state.page = "Clustering"

# Sidebar navigation
with st.sidebar:
    st.markdown("### Navigation")
    
    # Navigation buttons
    if st.button("🔮 Forecasting", key="nav_forecasting", 
                 use_container_width=True, 
                 type="primary" if st.session_state.page == "Forecasting" else "secondary"):
        navigate_to_forecasting()
        
    if st.button("📊 Emission Statistics", key="nav_statistics", 
                 use_container_width=True,
                 type="primary" if st.session_state.page == "Statistics" else "secondary"):
        navigate_to_statistics()
        
    if st.button("📈 Clustering", key="nav_clustering", 
                 use_container_width=True,
                 type="primary" if st.session_state.page == "Clustering" else "secondary"):
        navigate_to_clustering()
    
    # Display caching information if available
    if st.session_state.last_execution_time is not None:
        st.markdown("---")
        st.markdown("### Performance")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.last_execution_time:.2f}s</div>
            <div class="metric-label">Last Execution Time</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a button to clear the cache
        if st.button("Clear Cache", use_container_width=True):
            # Reset session state variables
            st.session_state.last_model_choice = None
            st.session_state.last_emission_choice = None
            st.session_state.last_region_choice = None
            st.session_state.last_forecast_years = None
            st.session_state.cached_forecast_results = None
            st.session_state.last_execution_time = None
            st.success("Cache cleared successfully!")
            st.experimental_rerun()

# Main dashboard area
st.title("🌎 Carbon Emission Forecasting Dashboard")
st.markdown("---")

# Display the selected page
if st.session_state.page == "Forecasting":
    
    # Check if forecast already exists
    has_forecast = (st.session_state.cached_forecast_results is not None)
    
    # Create a two-column layout with settings and content
    if not has_forecast:
        # If no forecast yet, show welcome message and settings side by side
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Welcome message
            st.markdown("""
            <div class="welcome-card">
                <h2>Welcome to the Carbon Emissions Forecasting Tool</h2>
                <p>
                    This interactive dashboard allows you to generate forecasts for future carbon emissions 
                    using various machine learning models.
                </p>
                <div style="max-width: 600px; margin: 0 auto;">
                    <h3>Getting Started:</h3>
                    <ol>
                        <li>Select a <strong>model type</strong> from the dropdown menu</li>
                        <li>Choose the <strong>emission type</strong> you want to forecast</li>
                        <li>Select a <strong>region</strong> for the forecast</li>
                        <li>Set the <strong>forecast horizon</strong> in years</li>
                        <li>Click the <strong>Generate Forecast</strong> button to view results</li>
                    </ol>
                </div>
                <div class="tip-box">
                    <strong>💡 Tip:</strong> For most accurate results, ARIMA models are recommended for short-term forecasts, while neural network models (LSTM, GRU) may provide better long-term predictions.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Forecast settings
            st.markdown("""
            <div class="settings-card">
                <h3>Forecast Settings</h3>
            </div>
            """, unsafe_allow_html=True)
            
            model_choice = st.selectbox("Choose Model", ["LSTM", "GRU", "ARIMA", "ARIMA+LSTM"])
            emission_choice = st.selectbox("Emission Type", ["CO2", "GHG"])
            region_choice = st.selectbox("Region", ["United states", "World"])
            years_to_forecast = st.number_input("Forecast Horizon (years)", 
                                              min_value=1, max_value=100, value=10, step=1)
            
            generate_button = st.button("Generate Forecast", use_container_width=True)
    else:
        # If there's already a forecast, show settings at the top
        st.markdown("""
        <div class="settings-card">
            <h3>Forecast Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col1:
            model_choice = st.selectbox("Model", ["LSTM", "GRU", "ARIMA", "ARIMA+LSTM"], 
                                        index=["LSTM", "GRU", "ARIMA", "ARIMA+LSTM"].index(st.session_state.last_model_choice))
        with col2:
            emission_choice = st.selectbox("Emission Type", ["CO2", "GHG"],
                                          index=["CO2", "GHG"].index(st.session_state.last_emission_choice))
        with col3:
            region_choice = st.selectbox("Region", ["United states", "World"],
                                        index=["United states", "World"].index(st.session_state.last_region_choice))
        with col4:
            years_to_forecast = st.number_input("Forecast Horizon", 
                                              min_value=1, max_value=100, value=st.session_state.last_forecast_years, step=1)
        with col5:
            generate_button = st.button("Update Forecast", use_container_width=True)
    
    # Handle forecast generation
if generate_button or has_forecast:
    # Use existing forecast if no button press but we have cached results
    if not generate_button and has_forecast:
        model_choice = st.session_state.last_model_choice
        emission_choice = st.session_state.last_emission_choice
        region_choice = st.session_state.last_region_choice
        years_to_forecast = st.session_state.last_forecast_years
    
    # Check if we've already computed this exact forecast
    same_parameters = (
        st.session_state.last_model_choice == model_choice and
        st.session_state.last_emission_choice == emission_choice and
        st.session_state.last_region_choice == region_choice and
        st.session_state.last_forecast_years == years_to_forecast and
        st.session_state.cached_forecast_results is not None
    )
    
    if same_parameters and not generate_button:
        # Use cached results (silently)
        forecast_df, forecast_fig, metrics = st.session_state.cached_forecast_results
    elif same_parameters and generate_button:
        # Use cached results (with notification)
        forecast_df, forecast_fig, metrics = st.session_state.cached_forecast_results
        st.markdown(f"""
        <div class="success-box">
            <strong>⚡ Retrieved cached forecast in {st.session_state.last_execution_time:.2f} seconds</strong>
        </div>
        """, unsafe_allow_html=True)
    elif generate_button:
        # Generate new forecast
        with st.spinner('Loading model and generating forecast...'):
            try:
                start_time = time.time()

                
                            
                # Load the selected model
                model_data = load_model(model_choice, emission_choice, region_choice)
                
                # Generate forecast
                forecast_df, forecast_fig, metrics = forecast_future(
                    model_choice, model_data, years_to_forecast, emission_choice, region_choice
                )
                
                # Calculate execution time
                execution_time = time.time() - start_time
                st.session_state.last_execution_time = execution_time
                
                # Store results and parameters in session state
                st.session_state.cached_forecast_results = (forecast_df, forecast_fig, metrics)
                st.session_state.last_model_choice = model_choice
                st.session_state.last_emission_choice = emission_choice
                st.session_state.last_region_choice = region_choice
                st.session_state.last_forecast_years = years_to_forecast
                
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ Generated forecast in {execution_time:.2f} seconds</strong>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during forecasting: {str(e)}")
                st.info("Please check that the required model files exist and the data paths are correct.")
                st.stop()
    
    # Display the forecast header
    st.markdown(f"""
    <div class="content-card">
        <h2>Forecasted {emission_choice} Emissions for {region_choice}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. DISPLAY METRICS FIRST
    st.markdown("""
    <div class="content-card">
        <h3>Summary Metrics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display metrics in a nice card layout
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['avg_annual_pct_change']:.2f}%</div>
            <div class="metric-label">Average Annual % Change</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['total_pct_change']:.2f}%</div>
            <div class="metric-label">Total % Change</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['min_annual_pct_change']:.2f}%</div>
            <div class="metric-label">Min Annual % Change</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['max_annual_pct_change']:.2f}%</div>
            <div class="metric-label">Max Annual % Change</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add interpretation of the forecast
    st.markdown("""<h3 style="margin: 30px 0 15px 0;">💡 Forecast Interpretation</h3>""", unsafe_allow_html=True)
    
    # Create expandable section for interpretation
    with st.expander("View Forecast Analysis", expanded=True):
        # Provide a basic interpretation based on the metrics
        if metrics['total_pct_change'] > 0:
            trend_description = f"increasing trend with a total increase of {metrics['total_pct_change']:.2f}% over the forecast period"
        elif metrics['total_pct_change'] < 0:
            trend_description = f"decreasing trend with a total decrease of {abs(metrics['total_pct_change']):.2f}% over the forecast period"
        else:
            trend_description = "relatively stable trend over the forecast period"
        
        st.write(f"The forecast shows a {trend_description}. The average annual percentage change is {metrics['avg_annual_pct_change']:.2f}%.")
        
        if model_choice == "ARIMA":
            st.write("This forecast was generated using an ARIMA model, which is effective at capturing time series patterns and seasonality in the data.")
        elif model_choice == "ARIMA+LSTM":
            st.write("This forecast combines the strengths of both ARIMA (for trend and seasonality) and neural networks (for complex patterns).")
        elif model_choice in ["LSTM", "GRU"]:
            st.write(f"This forecast was generated using a {model_choice} neural network, which excels at learning complex patterns in time series data.")
    
    # 2. DISPLAY VISUALIZATION NEXT
    st.markdown("""
    <div class="content-card" style="margin-top: 30px;">
        <h3>Visualization</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Make the plot larger and update the configuration for interactivity
    forecast_fig.update_layout(
        height=600,  # Make plot taller
        width=None,  # Let it scale with the container width
        hovermode="x unified",  # Show all values at same x position
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Enable various interactions with the plot
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'editable': True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{region_choice}_{emission_choice}_forecast',
            'height': 600,
            'width': 1200,
            'scale': 2
        }
    }
    
    # Display the plot with the enhanced configuration
    st.plotly_chart(forecast_fig, use_container_width=True, config=config)
    
    # 3. DISPLAY DATA TABLE LAST
    st.markdown("""
    <div class="content-card" style="margin-top: 30px;">
        <h3>Forecast Data</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a clean table with the most relevant forecast data
    yearly_data = pd.DataFrame({
        'Year': forecast_df['Year'].astype(int),
        f'Forecasted {emission_choice}': forecast_df['Forecasted_CO2'].round(2),
        'Annual % Change': forecast_df['pct_change'].round(2).apply(lambda x: f"{x}%" if not pd.isna(x) else "N/A"),
        'Cumulative % Change': forecast_df['pct_change_from_last_historical'].round(2).apply(lambda x: f"{x}%")
    })
    
    # Use Streamlit's data editor for a nicer display
    st.dataframe(
        yearly_data,
        column_config={
            'Year': st.column_config.NumberColumn("Year", format="%d"),
            f'Forecasted {emission_choice}': st.column_config.NumberColumn(f"Forecasted {emission_choice}", format="%.2f"),
            'Annual % Change': st.column_config.TextColumn("Annual % Change"),
            'Cumulative % Change': st.column_config.TextColumn("Cumulative % Change")
        },
        hide_index=True,
        use_container_width=True,
        height=400
    )
    

                
            

elif st.session_state.page == "Statistics":
    # Emission Statistics Page (Coming Soon)
    st.markdown("""
    <div class="coming-soon">
        <h2>📊 Emission Statistics Module</h2>
        <p style="font-size: 18px; margin: 20px 0; color: #263238;">Coming Soon!</p>
        <p style="font-size: 16px; max-width: 600px; margin: 0 auto 30px auto; color: #263238;">
            This section will allow users to view historical emission summaries, trends, and analytics.
            We're working hard to bring you comprehensive data visualization and analysis tools.
        </p>
        <div class="tip-box" style="display: inline-block; text-align: left; margin: 0 auto;">
            <strong>Features to expect:</strong> Historical trends, country comparisons, sectoral breakdowns, and interactive data exploration.
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "Clustering":
    # Clustering Page (Coming Soon)
    st.markdown("""
    <div class="coming-soon">
        <h2>📈 Clustering Module</h2>
        <p style="font-size: 18px; margin: 20px 0; color: #263238;">Coming Soon!</p>
        <p style="font-size: 16px; max-width: 600px; margin: 0 auto 30px auto; color: #263238;">
            This section will enable clustering of countries or regions based on emissions profiles.
            Our data scientists are developing sophisticated algorithms to identify patterns and groupings.
        </p>
        <div class="tip-box" style="display: inline-block; text-align: left; margin: 0 auto;">
            <strong>Features to expect:</strong> Dynamic clustering, similarity analysis, trend-based grouping, and interactive visualizations.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add footer
st.markdown("""
<footer>
    <h3>About This Dashboard</h3>
    <p>This dashboard provides carbon emission forecasting capabilities using multiple machine learning models:</p>
    <ul>
        <li><strong>LSTM and GRU:</strong> Neural network models that excel at capturing complex patterns in time series data</li>
        <li><strong>ARIMA:</strong> Statistical model that handles seasonality and trends well</li>
        <li><strong>ARIMA+LSTM:</strong> Hybrid approach combining the strengths of both statistical and neural network models</li>
    </ul>
    <p style="font-size: 12px; color: #666; margin-top: 20px;">
        Built with Streamlit • Last updated: April 2025
    </p>
</footer>
""", unsafe_allow_html=True)