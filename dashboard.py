import streamlit as st
import pandas as pd
from utils import load_model, forecast_future, LSTMModel, GRUModel
import plotly.graph_objs as go
import plotly.express as px
import time
import sys
import os
import os
from datetime import datetime
from llm_utils import generate_forecast_summary  # Added import for AI summary

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
    h1, h2, h3, h4, h5, h6 {{
        color: {PRIMARY_COLOR} !important;
        font-weight: 600;
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
    section[data-testid="stSidebar"] .stRadio {{
        border-radius: 8px;
        overflow: hidden;
    }}
    div[data-testid="stExpander"] {{
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        background-color: {CARD_BG};
    }}
    div[data-testid="stExpander"] h1, 
    div[data-testid="stExpander"] h2, 
    div[data-testid="stExpander"] h3, 
    div[data-testid="stExpander"] h4 {{
        color: {PRIMARY_COLOR} !important;
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
        color: {TEXT_COLOR};
    }}
    .content-card h1, 
    .content-card h2, 
    .content-card h3, 
    .content-card h4 {{
        color: {PRIMARY_COLOR} !important;
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
    .tip-box strong {{
        color: {PRIMARY_COLOR};
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
        color: {PRIMARY_COLOR} !important;
    }}
    .welcome-card p {{
        font-size: 17px;
        margin-bottom: 25px;
        color: {TEXT_COLOR};
    }}
    .welcome-card h3 {{
        color: {PRIMARY_COLOR} !important;
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
    }}
    .settings-card h3 {{
        color: {PRIMARY_COLOR} !important;
    }}
    .nav-button {{
        margin-bottom: 0.5rem;
        width: 100%;
    }}
    /* Fix for markdown headings in AI-generated content */
    .markdown-text h1, 
    .markdown-text h2, 
    .markdown-text h3, 
    .markdown-text h4 {{
        color: {PRIMARY_COLOR} !important;
    }}

    #policy-recommendations h1,
    #policy-recommendations h2,
    #policy-recommendations h3,
    #policy-recommendations h4,
    #policy-recommendations h5,
    #policy-recommendations h6 {{
        color: {PRIMARY_COLOR} !important;
        font-weight: 600;
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
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'ai_summary' not in st.session_state:
        st.session_state.ai_summary = None
    if 'show_ai_summary' not in st.session_state:
        st.session_state.show_ai_summary = False

# Initialize session state
initialize_session_state()

# Define page navigation functions
def navigate_to_forecasting():
    st.session_state.page = "Forecasting"

def navigate_to_statistics():
    st.session_state.page = "Statistics"

def navigate_to_llm_integration():
    st.session_state.page = "LLM Integration"

# Add API key settings to sidebar
with st.sidebar:
    st.markdown("### API Settings")
    use_own_key = st.checkbox("Use my own OpenAI API key")
    
    if use_own_key:
        user_api_key = st.text_input("OpenAI API Key", type="password")
        if user_api_key:
            st.success("API key provided!")
            # Store in session state for persistence
            st.session_state["user_api_key"] = user_api_key
            # Add a note about security
            st.info("Your API key is stored only in your current browser session.")
        else:
            st.warning("Please enter your API key")

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
        
    if st.button("🤖 Policy Recommendations", key="nav_llm", 
         use_container_width=True,
         type="primary" if st.session_state.page == "LLM Integration" else "secondary"):
        navigate_to_llm_integration()
    
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
            st.session_state.ai_summary = None
            st.session_state.show_ai_summary = False
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
                    st.session_state.ai_summary = None  # Clear previous AI summary when new forecast is generated
                    st.session_state.show_ai_summary = False
                    
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
        interpretation_cols = st.columns([1, 3])
        with interpretation_cols[0]:
            st.markdown(f"""
            <h3 style="margin: 30px 0 15px 0; color: {PRIMARY_COLOR};">💡 Forecast Interpretation</h3>
            """, unsafe_allow_html=True)
            # Add AI analysis button
            if st.button("Generate AI Analysis", key="generate_ai_analysis", use_container_width=True):
                with st.spinner("Generating AI forecast analysis..."):
                    try:
                        # Generate AI summary
                        analysis = generate_forecast_summary(
                            forecast_df, 
                            metrics, 
                            region_choice, 
                            emission_choice, 
                            model_choice
                        )
                        st.session_state.ai_summary = analysis
                        st.session_state.show_ai_summary = True
                    except Exception as e:
                        st.error(f"Error generating AI analysis: {str(e)}")

        with interpretation_cols[1]:
            # Check if we should show AI summary
            if st.session_state.show_ai_summary and st.session_state.ai_summary:
                # Create expandable section for AI interpretation
                with st.expander("View AI Analysis", expanded=True):
                    # Wrap the markdown content in a div with a class we can target with CSS
                    st.markdown(f"""
                    <div class="markdown-text">
                    {st.session_state.ai_summary}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("Download Analysis", key="download_analysis"):
                        st.download_button(
                            label="Download Analysis as Markdown",
                            data=st.session_state.ai_summary,
                            file_name=f"{region_choice}_{emission_choice}_forecast_analysis.md",
                            mime="text/markdown",
                        )
            else:
                # Create expandable section for basic interpretation
                with st.expander("View Basic Analysis", expanded=True):
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
    st.markdown("""
    <div class="content-card">
        <h2>📊 Global Emission Statistics</h2>
        <p>Analyze and compare greenhouse gas emissions across different countries and regions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different visualization options
    stats_tabs = st.tabs(["Country Comparison", "Historical Trends", "Emission Breakdown"])
    
    with stats_tabs[0]:
        st.markdown("""
        <div class="settings-card">
            <h3>Country Comparison Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar settings
        col1, col2 = st.columns(2)
        
        with col1:
            # Gas selection
            gas_type = st.selectbox(
                "Select Gas Type",
                ["CO2", "Total GHG", "Methane", "Nitrous Oxide"],
                index=0
            )
            
            # Year selection
            available_years = list(range(1990, 2024))
            selected_year = st.slider("Select Year", min_value=min(available_years), 
                                     max_value=max(available_years), value=2023)
            
            # Normalization options
            normalization = st.radio(
                "View As",
                ["Total Emissions", "Per Capita", "Per GDP", "% of Global"]
            )
        
        with col2:
            # Allow multi-selection of countries with default top emitters
            # We'll use this later to filter the data
            default_countries = ["China", "United States", "India", "Russia", "Japan", 
                               "Germany", "Iran", "South Korea", "Saudi Arabia", "Canada"]
            
            num_countries = st.slider("Number of Top Emitters", min_value=5, max_value=20, value=10)
            
            show_regions = st.checkbox("Include Regions", value=True)
            
            # Option to sort results
            sort_order = st.radio(
                "Sort Order",
                ["Descending", "Ascending"]
            )
        
        # Load data function (to be implemented using the actual data path in your project)
        @st.cache_data
        def load_emission_data():
            """Load and prepare the emissions data from the CSV"""
            import pandas as pd
            import os
            
            # Try to find the data directory relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "data", "raw_data", "owid-co2-data.csv")
            
            try:
                df = pd.read_csv(data_path)
            except FileNotFoundError:
                try:
                    # Alternate path
                    df = pd.read_csv("../data/raw_data/owid-co2-data.csv")
                except FileNotFoundError:
                    # As a fallback, try to locate the file in the working directory
                    df = pd.read_csv("owid-co2-data.csv")
            
            return df
        
        try:
            # Load the data
            emissions_df = load_emission_data()
            
            # Map the selected gas type to the corresponding column in the dataset
            gas_column_mapping = {
                "CO2": "co2",
                "Total GHG": "total_ghg",
                "Methane": "methane",
                "Nitrous Oxide": "nitrous_oxide"
            }
            
            normalization_mapping = {
                "Total Emissions": "",
                "Per Capita": "_per_capita",
                "Per GDP": "_per_gdp",
                "% of Global": "share_global_"
            }
            
            # Determine which column to use based on user selections
            base_column = gas_column_mapping[gas_type]
            if normalization == "Total Emissions":
                column_to_use = base_column
            elif normalization == "% of Global":
                if base_column == "co2":
                    column_to_use = "share_global_co2"
                elif base_column == "total_ghg":
                    # There might not be a direct share column, so calculate it
                    column_to_use = base_column
                    calculate_share = True
                else:
                    column_to_use = base_column
                    calculate_share = True
            else:
                column_to_use = base_column + normalization_mapping[normalization]
            
            # Filter the data for the selected year
            year_data = emissions_df[emissions_df["year"] == selected_year].copy()
            
            # Remove rows with NaN values in the column we're plotting
            year_data = year_data.dropna(subset=[column_to_use])
            
            # Remove rows with 0 values in the selected column
            year_data = year_data[year_data[column_to_use] > 0]
            
            # If we need to calculate share, do it here
            if normalization == "% of Global" and base_column != "co2":
                global_total = year_data[base_column].sum()
                year_data["calculated_share"] = (year_data[base_column] / global_total) * 100
                column_to_use = "calculated_share"
            
            # Remove regions if not showing regions
            if not show_regions:
                # Filter out regions - this is a simplistic approach, you might want to improve it
                regions = ["World", "Africa", "Asia", "Europe", "North America", "South America",
                          "European Union", "High-income countries", "Upper-middle-income countries",
                          "Lower-middle-income countries", "Low-income countries", "OECD", "Non-OECD"]
                year_data = year_data[~year_data["country"].isin(regions)]
            
            # Sort the data
            ascending = True if sort_order == "Ascending" else False
            year_data = year_data.sort_values(by=column_to_use, ascending=ascending)
            
            # Get the top N countries
            top_countries_data = year_data.head(num_countries)
            
            # Create plot
            import plotly.express as px
            import plotly.graph_objs as go
            
            # Determine the title and y-axis label based on the selections
            if normalization == "Total Emissions":
                title = f"Top {num_countries} {gas_type} Emitters in {selected_year}"
                y_label = f"{gas_type} Emissions (million tonnes)"
            elif normalization == "Per Capita":
                title = f"Top {num_countries} {gas_type} Emitters Per Capita in {selected_year}"
                y_label = f"{gas_type} Emissions (tonnes per person)"
            elif normalization == "Per GDP":
                title = f"Top {num_countries} {gas_type} Emitters Per GDP in {selected_year}"
                y_label = f"{gas_type} Emissions (kg per international-$)"
            else:  # % of Global
                title = f"Top {num_countries} Contributors to Global {gas_type} Emissions in {selected_year}"
                y_label = f"Share of Global {gas_type} Emissions (%)"
            
            # Create a horizontal bar chart
            fig = go.Figure()
            
            # Add a bar trace
            fig.add_trace(go.Bar(
                x=top_countries_data[column_to_use],
                y=top_countries_data["country"],
                orientation='h',
                marker=dict(
                    color=top_countries_data[column_to_use],
                    colorscale='Reds',
                    showscale=False
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=y_label,
                yaxis_title="Country",
                height=600,
                template="plotly_white",
                yaxis=dict(autorange="reversed")  # This ensures the bars are sorted correctly
            )
            
            # Set config for interactive features
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'editable': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{gas_type}_emissions_{selected_year}',
                    'height': 600,
                    'width': 1200,
                    'scale': 2
                }
            }
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            # Display the data table with more details
            st.markdown("""
            <div class="content-card" style="margin-top: 30px;">
                <h3>Detailed Data</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Prepare the table with formatted data
            display_df = top_countries_data[["country", column_to_use]].copy()
            display_df.columns = ["Country", f"{gas_type} Emissions"]
            
            # Use Streamlit's data editor for a nicer display
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
        except Exception as e:
            st.error(f"An error occurred while loading or processing the data: {str(e)}")
            st.info("Please ensure the emissions data file is available in the correct location.")
    
    with stats_tabs[1]:
        st.markdown("""
        <div class="settings-card">
            <h3>Historical Trend Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Countries selection (multi-select)
            countries_for_trends = st.multiselect(
                "Select Countries/Regions",
                ["World", "China", "United States", "India", "EU-27", "Russia"],
                default=["World", "China", "United States", "India"]
            )
            
            # Gas selection
            trend_gas_type = st.selectbox(
                "Select Gas Type",
                ["CO2", "Total GHG", "Methane", "Nitrous Oxide"],
                key="trend_gas_type"
            )
        
        with col2:
            # Time range selection
            year_range = st.slider(
                "Select Year Range",
                min_value=1750,
                max_value=2023,
                value=(1950, 2023),
                key="year_range"
            )
            
            # Normalization options
            trend_normalization = st.radio(
                "View As",
                ["Total Emissions", "Per Capita", "Per GDP", "% of Global"],
                key="trend_normalization"
            )
        
        # Display placeholder for historical trend graph
        st.markdown("""
        <div class="content-card">
            <h3>Historical Emission Trends</h3>
            <p>Select countries and metrics to visualize emission trends over time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Load the data (using the cached function)
            emissions_df = load_emission_data()
            
            # Map the selected gas type to the corresponding column in the dataset
            gas_column_mapping = {
                "CO2": "co2",
                "Total GHG": "total_ghg",
                "Methane": "methane",
                "Nitrous Oxide": "nitrous_oxide"
            }
            
            normalization_mapping = {
                "Total Emissions": "",
                "Per Capita": "_per_capita",
                "Per GDP": "_per_gdp",
                "% of Global": "share_global_"
            }
            
            # Determine which column to use based on user selections
            base_column = gas_column_mapping[trend_gas_type]
            if trend_normalization == "Total Emissions":
                column_to_use = base_column
            elif trend_normalization == "% of Global":
                if base_column == "co2":
                    column_to_use = "share_global_co2"
                else:
                    # For other gases, we need to calculate the share
                    calculate_share = True
                    column_to_use = base_column
            else:
                column_to_use = base_column + normalization_mapping[trend_normalization]
            
            # Filter the data for the selected year range and countries
            filtered_df = emissions_df[
                (emissions_df["year"] >= year_range[0]) & 
                (emissions_df["year"] <= year_range[1]) & 
                (emissions_df["country"].isin(countries_for_trends))
            ].copy()
            
            # If we need to calculate share, do it for each year
            if trend_normalization == "% of Global" and base_column != "co2":
                # Group by year and calculate global total for each year
                year_totals = emissions_df.groupby("year")[base_column].sum().reset_index()
                year_totals.columns = ["year", "global_total"]
                
                # Merge with filtered data
                filtered_df = filtered_df.merge(year_totals, on="year", how="left")
                
                # Calculate the share
                filtered_df["calculated_share"] = (filtered_df[base_column] / filtered_df["global_total"]) * 100
                column_to_use = "calculated_share"
            
            # Create plot
            import plotly.graph_objs as go
            
            # Determine the title and y-axis label based on the selections
            if trend_normalization == "Total Emissions":
                title = f"{trend_gas_type} Emissions Over Time ({year_range[0]}-{year_range[1]})"
                y_label = f"{trend_gas_type} Emissions (million tonnes)"
            elif trend_normalization == "Per Capita":
                title = f"{trend_gas_type} Emissions Per Capita Over Time ({year_range[0]}-{year_range[1]})"
                y_label = f"{trend_gas_type} Emissions (tonnes per person)"
            elif trend_normalization == "Per GDP":
                title = f"{trend_gas_type} Emissions Per GDP Over Time ({year_range[0]}-{year_range[1]})"
                y_label = f"{trend_gas_type} Emissions (kg per international-$)"
            else:  # % of Global
                title = f"Share of Global {trend_gas_type} Emissions Over Time ({year_range[0]}-{year_range[1]})"
                y_label = f"Share of Global {trend_gas_type} Emissions (%)"
            
            # Create the line chart
            fig = go.Figure()
            
            # Add a line for each country
            for country in countries_for_trends:
                country_data = filtered_df[filtered_df["country"] == country]
                
                # Skip countries with no data
                if country_data.empty or column_to_use not in country_data.columns:
                    continue
                
                fig.add_trace(go.Scatter(
                    x=country_data["year"],
                    y=country_data[column_to_use],
                    mode='lines',
                    name=country
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Year",
                yaxis_title=y_label,
                legend_title="Countries/Regions",
                template="plotly_white",
                height=600
            )
            
            # Set config for interactive features
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'editable': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{trend_gas_type}_emissions_trend_{year_range[0]}-{year_range[1]}',
                    'height': 600,
                    'width': 1200,
                    'scale': 2
                }
            }
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            # Add insights about the trends
            st.markdown("""
            <div class="content-card">
                <h3>💡 Trend Insights</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View Trend Analysis", expanded=True):
                # Generate some basic insights
                st.write("This section shows historical trends in emissions data over the selected time period.")
                st.write("You can observe how emissions have changed over time for different countries and regions.")
                
                # Add specific insights based on the selected data
                # (This would need more logic to generate meaningful insights from the actual data patterns)
                if "World" in countries_for_trends and column_to_use in filtered_df.columns:
                    world_data = filtered_df[filtered_df["country"] == "World"]
                    if not world_data.empty:
                        first_year_value = world_data.iloc[0][column_to_use]
                        last_year_value = world_data.iloc[-1][column_to_use]
                        percent_change = ((last_year_value - first_year_value) / first_year_value) * 100
                        
                        st.write(f"Global {trend_gas_type} emissions have {'increased' if percent_change > 0 else 'decreased'} " 
                                f"by approximately {abs(percent_change):.1f}% from {year_range[0]} to {year_range[1]}.")
        
        except Exception as e:
            st.error(f"An error occurred during trend analysis: {str(e)}")
            st.info("Please check that the data file is available and contains the required columns.")
    
    with stats_tabs[2]:
        st.markdown("""
        <div class="content-card">
            <h2>CO₂ Emissions by Source</h2>
            <p>Visualize the breakdown of carbon dioxide emissions by source for different countries and years.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a two-column layout for settings
        col1, col2 = st.columns(2)
        
        with col1:
            # Country selection
            breakdown_country = st.selectbox(
                "Select Country/Region",
                ["World", "China", "United States", "India", "Russia", "Japan", 
                 "Germany", "United Kingdom", "Brazil", "Canada", "Australia"],
                index=0
            )
        
        with col2:
            # Year selection with a slider
            available_years = list(range(1990, 2024))
            breakdown_year = st.slider(
                "Select Year",
                min_value=min(available_years),
                max_value=max(available_years),
                value=2023,
                key="source_breakdown_year_slider"
            )
        
        try:
            # Load the data (using the cached function)
            emissions_df = load_emission_data()
            
            # Define source columns and corresponding labels
            source_columns = ["coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2", "other_industry_co2"]
            source_labels = ["Coal", "Oil", "Gas", "Cement", "Flaring", "Other Industry"]
            
            # Filter for the selected country and year
            country_data = emissions_df[
                (emissions_df["country"] == breakdown_country) & 
                (emissions_df["year"] == breakdown_year)
            ]
            
            if not country_data.empty:
                # Extract values for each source
                values = []
                labels = []
                
                for col, label in zip(source_columns, source_labels):
                    if col in country_data.columns and not country_data[col].isna().all():
                        value = country_data[col].values[0]
                        if not pd.isna(value) and value > 0:
                            values.append(value)
                            labels.append(label)
                
                if values:
                    import plotly.graph_objs as go
                    
                    # Create a stylish donut chart
                    fig = go.Figure()
                    
                    # Custom color palette - environmental theme
                    colors = ['#1A5E63', '#028090', '#00A896', '#02C39A', '#F0F3BD', '#96C0B7']
                    
                    # Add pie chart with custom styling
                    fig.add_trace(go.Pie(
                        labels=labels,
                        values=values,
                        textposition='inside',
                        textinfo='percent+label',
                        insidetextfont=dict(color='white', size=14, family='Arial, sans-serif'),
                        hoverinfo='label+percent+value',
                        textfont=dict(size=14, family='Arial, sans-serif'),
                        marker=dict(
                            colors=colors[:len(labels)],
                            line=dict(color='white', width=2)
                        ),
                        hole=0.3,  # Create a donut chart effect
                        rotation=90  # Start from the top
                    ))
                    
                    # Add title with custom styling
                    fig.update_layout(
                        title={
                            'text': f"{breakdown_country} CO₂ Emissions by Source ({breakdown_year})",
                            'y': 0.95,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': dict(
                                size=22,
                                color=PRIMARY_COLOR,
                                family='Arial, sans-serif'
                            )
                        },
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.1,
                            xanchor="center",
                            x=0.5,
                            font=dict(
                                size=12,
                                color=TEXT_COLOR
                            )
                        ),
                        height=600,
                        margin=dict(l=50, r=50, t=100, b=50),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    # Configure interactive features
                    config = {
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'editable': True,
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{breakdown_country}_emissions_by_source_{breakdown_year}',
                            'height': 600,
                            'width': 1000,
                            'scale': 2
                        }
                    }
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True, config=config)
                    
                    # Display the data table
                    st.markdown("""
                    <div class="content-card" style="margin-top: 20px;">
                        <h3>Source Data</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a DataFrame for display
                    source_df = pd.DataFrame({
                        'Source': labels,
                        'Emissions (million tonnes)': values,
                        'Percentage': [v/sum(values)*100 for v in values]
                    })
                    
                    # Format the percentage column
                    source_df['Percentage'] = source_df['Percentage'].apply(lambda x: f"{x:.2f}%")
                    
                    # Display the table
                    st.dataframe(
                        source_df,
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Add insights about the breakdown
                    st.markdown("""
                    <div class="content-card" style="margin-top: 20px;">
                        <h3>💡 Source Analysis</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View Source Analysis", expanded=True):
                        # Identify the main contributor
                        max_idx = values.index(max(values))
                        max_source = labels[max_idx]
                        max_percent = source_df['Percentage'][max_idx]
                        
                        st.markdown(f"""
                        <div class="tip-box">
                            <strong>Key Insight:</strong> The largest source of CO₂ emissions in {breakdown_country} during {breakdown_year} was <strong>{max_source}</strong>, 
                            accounting for {max_percent} of total emissions from fossil fuels and industry.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Calculate total emissions
                        total_emissions = sum(values)
                        st.write(f"Total CO₂ emissions from all sources: {total_emissions:.2f} million tonnes.")
                        
                        # Add comparison to typical patterns if World is selected
                        if breakdown_country == "World":
                            st.write(f"Globally, fossil fuels (coal, oil, and gas) remain the dominant sources of CO₂ emissions, with industrial processes like cement production contributing a smaller portion.")
                        
                        # Add specific analysis based on the country
                        if "coal_co2" in country_data.columns and "gas_co2" in country_data.columns:
                            coal_value = country_data["coal_co2"].values[0] if not pd.isna(country_data["coal_co2"].values[0]) else 0
                            gas_value = country_data["gas_co2"].values[0] if not pd.isna(country_data["gas_co2"].values[0]) else 0
                            
                            if coal_value > gas_value:
                                ratio = coal_value / gas_value if gas_value > 0 else float('inf')
                                if ratio != float('inf'):
                                    st.write(f"Coal emissions are approximately {ratio:.1f}x higher than natural gas emissions in {breakdown_country}.")
                            else:
                                ratio = gas_value / coal_value if coal_value > 0 else float('inf')
                                if ratio != float('inf'):
                                    st.write(f"Natural gas emissions are approximately {ratio:.1f}x higher than coal emissions in {breakdown_country}.")
                else:
                    st.warning(f"No source breakdown data available for {breakdown_country} in {breakdown_year}")
            else:
                st.warning(f"No data available for {breakdown_country} in {breakdown_year}")
        
        except Exception as e:
            st.error(f"An error occurred during source breakdown analysis: {str(e)}")
            st.info("Please check that the data file contains the required source breakdown columns.")

# Add the LLM Integration page
elif st.session_state.page == "LLM Integration":
    # Import and display the LLM page
    try:
        from llm_integration import display_llm_page
        display_llm_page()
    except Exception as e:
        st.error(f"Error loading the Policy Recommendations page: {str(e)}")
        st.info("Make sure you've added the llm_integration.py and llm_utils.py files to your project.")