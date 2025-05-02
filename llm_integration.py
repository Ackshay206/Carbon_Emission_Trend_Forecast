import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from utils import load_model, forecast_future
from llm_utils import (
    generate_policy_recommendations, 
    create_recommendation_chart
)
import os
from datetime import datetime

def display_llm_page():
    """
    Display the LLM Integration page of the dashboard focusing on policy recommendations
    """
    # Access PRIMARY_COLOR from the main script
    PRIMARY_COLOR = "#1B5E20"  # Default to dark green if not accessible
    if "PRIMARY_COLOR" in globals():
        PRIMARY_COLOR = globals()["PRIMARY_COLOR"]
    
    st.title("🤖 AI-Powered Policy Recommendations")
    st.markdown("---")
    
    st.markdown(f"""
    <div class="content-card">
        <h2 style="color: {PRIMARY_COLOR};">📝 AI-Generated Policy Recommendations</h2>
        <p>Generate evidence-based policy recommendations to meet emission reduction targets based on forecast data and sector analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings for policy recommendations
    st.markdown(f"""
    <div class="settings-card">
        <h3 style="color: {PRIMARY_COLOR};">Target Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_region = st.selectbox(
            "Region",
            ["World", "United_States"],
            index=1
        )
    
    with col2:
        reduction_target = st.slider(
            "Emission Reduction Target (%)",
            min_value=5,
            max_value=80,
            value=30,
            step=5
        )
    
    with col3:
        target_year = st.selectbox(
            "Target Year",
            list(range(2030, 2051, 5)),
            index=0
        )
    
    # Generate button
    generate_policy = st.button("Generate Policy Recommendations", use_container_width=True, key="generate_policy")
    
    if generate_policy:
        with st.spinner('Analyzing emissions data and generating evidence-based recommendations...'):
            try:
                # 1. Get historical and forecasted data for the region
                model_choice = "ARIMA"  # Use the most accurate model for your case
                emission_choice = "CO2"
                region_choice = target_region
                
                # Load model and generate forecast
                try:
                    model_data = load_model(model_choice, emission_choice, region_choice)
                    forecast_df, forecast_fig, metrics = forecast_future(
                        model_choice, model_data, target_year - datetime.now().year, emission_choice, region_choice
                    )
                except Exception as e:
                    st.error(f"Error loading model or generating forecast: {str(e)}")
                    # Use some default values for demonstration purposes
                    metrics = {
                        'avg_annual_pct_change': 1.2,
                        'total_pct_change': 15.0,
                        'min_annual_pct_change': 0.5,
                        'max_annual_pct_change': 2.1
                    }
                    forecast_df = pd.DataFrame({
                        'Year': list(range(datetime.now().year, target_year + 1)),
                        'Forecasted_CO2': [10000 * (1.012 ** i) for i in range(target_year - datetime.now().year + 1)]
                    })
                
                # 2. Generate recommendations based on the forecast
                recommendations, trajectories = generate_policy_recommendations(
                    target_region,
                    reduction_target,
                    target_year,
                    forecast_df,
                    metrics
                )
                
                # 3. Create a two-column layout for visualization and recommendations
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    # Display the trajectory comparison chart
                    st.markdown(f"""<h3 style="color: {PRIMARY_COLOR};">Emission Reduction Trajectories</h3>""", unsafe_allow_html=True)
                    trajectory_chart = create_recommendation_chart(trajectories)
                    st.plotly_chart(trajectory_chart, use_container_width=True)
                    
                    # Add key metrics in a nice card format
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.2f}%</div>
                        <div class="metric-label">Required Annual Reduction Rate</div>
                    </div>
                    """.format(trajectories['annual_reduction_needed']), unsafe_allow_html=True)
                    

                
                with col2:
                    # Display the recommendations with an HTML wrapper to apply styles
                    st.markdown(f"""<h3 style="color: {PRIMARY_COLOR};">Policy Recommendations</h3>""", unsafe_allow_html=True)
                    
                    # Create a container for the recommendations content
                    recommendations_container = st.container()
                    
                    # Add an ID to the container for CSS targeting
                    st.markdown("""
                    <style>
                    /* Target the next container's markdown content */
                    .main .block-container > div:nth-last-child(2) h1,
                    .main .block-container > div:nth-last-child(2) h2,
                    .main .block-container > div:nth-last-child(2) h3,
                    .main .block-container > div:nth-last-child(2) h4,
                    .main .block-container > div:nth-last-child(2) h5,
                    .main .block-container > div:nth-last-child(2) h6 {
                        color: """ + PRIMARY_COLOR + """ !important;
                        font-weight: 600;
                    }
                    
                    /* Target strong and bold tags inside the recommendations */
                    .main .block-container > div:nth-last-child(2) strong,
                    .main .block-container > div:nth-last-child(2) b {
                        color: """ + PRIMARY_COLOR + """;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display the processed recommendations with a class for the content
                    with recommendations_container:
                        st.markdown(f"""
                        <div class="recommendations-text" style="color: "#263238" !important;">
                        {recommendations}
                        </div>
                        """, unsafe_allow_html=True)
                    
                                
            except Exception as e:
                st.error(f"An error occurred while generating recommendations: {str(e)}")