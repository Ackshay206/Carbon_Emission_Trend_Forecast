import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import plotly.graph_objs as go
import plotly.express as px
import re
import openai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def get_openai_client():
    """Get OpenAI client using the appropriate API key"""
    # First check if user provided their own key in session state
    if "user_api_key" in st.session_state and st.session_state["user_api_key"]:
        return openai.OpenAI(api_key=st.session_state["user_api_key"])
    
    # Next try to get from Streamlit secrets (for cloud deployment)
    try:
        return openai.OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])
    except:
        # Finally fall back to environment variables
        return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def find_project_data_directory():
    """Find the data directory relative to the current file location"""
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the data directory
    data_dir = os.path.join(current_dir, "data", "raw_data")
    
    print(f"Project root directory: {current_dir}")
    print(f"Data directory: {data_dir}")
    
    return data_dir

def load_emissions_data():
    """Load the emissions data from CSV"""
    data_dir = find_project_data_directory()
    data_path = os.path.join(data_dir, "owid-co2-data.csv")
    
    try:
        # Try current directory first
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        # Then try project structure
        elif os.path.exists('../data/raw_data/owid-co2-data.csv'):
            return pd.read_csv('../data/raw_data/owid-co2-data.csv')
        else:
            # Last resort, try current directory
            return pd.read_csv('owid-co2-data.csv')
    except FileNotFoundError:
        raise FileNotFoundError("Could not find the emissions data file. Please ensure it's available.")

def generate_policy_recommendations(region, reduction_target, target_year, forecast_data, metrics):
    """Generate policy recommendations using OpenAI API"""
    client = get_openai_client()

    # Get current emissions (start of forecast)
    latest_emission = forecast_data['Forecasted_CO2'].iloc[0]
    
    # Calculate target emission level
    target_emission = latest_emission * (1 - reduction_target/100)
    
    # Calculate years to target
    current_year = datetime.now().year
    years_to_target = target_year - current_year
    
    # Calculate required annual reduction rate
    annual_reduction_needed = ((latest_emission - target_emission) / latest_emission) / years_to_target * 100
    
    # Load emissions data for sector breakdown
    try:
        emissions_df = load_emissions_data()

        # Get the latest year data for the region
        region_data = emissions_df[emissions_df['country'] == region].sort_values('year', ascending=False).iloc[0]
        
        # Extract sector data if available
        sector_data = {
            'coal': region_data.get('coal_co2', 'N/A'),
            'oil': region_data.get('oil_co2', 'N/A'),
            'gas': region_data.get('gas_co2', 'N/A'),
            'cement': region_data.get('cement_co2', 'N/A'),
            'trade': region_data.get('trade_co2', 'N/A')
        }
        
        # Calculate percentages
        if region_data.get('co2') and region_data.get('co2') > 0:
            total_co2 = region_data.get('co2')
            sector_percentages = {
                'coal': (region_data.get('coal_co2', 0) / total_co2) * 100,
                'oil': (region_data.get('oil_co2', 0) / total_co2) * 100,
                'gas': (region_data.get('gas_co2', 0) / total_co2) * 100,
                'cement': (region_data.get('cement_co2', 0) / total_co2) * 100,
                'trade': (region_data.get('trade_co2', 0) / total_co2) * 100
            }
        else:
            sector_percentages = {k: 'N/A' for k in sector_data.keys()}
    except Exception as e:
        print(f"Error loading sector data: {str(e)}")
        sector_data = {"Error": "Could not load sector breakdown"}
        sector_percentages = {"Error": "Could not calculate percentages"}
    
    # Create a prompt for the LLM with more detailed sector data
    prompt = f"""
    You are an expert climate policy advisor. Generate detailed, evidence-based policy recommendations for {region}
    to reduce CO2 emissions by {reduction_target}% by {target_year}.
    
    CONTEXT:
    - Current annual emissions: {latest_emission:.2f} million tonnes CO2
    - Target emissions: {target_emission:.2f} million tonnes CO2
    - Required annual reduction rate: {annual_reduction_needed:.2f}%
    - Current projected trend: {metrics['avg_annual_pct_change']:.2f}% annual change
    
    SOURCE BREAKDOWN:
    - Coal: {sector_data['coal']} million tonnes CO2 ({sector_percentages['coal']:.1f}% of total)
    - Oil: {sector_data['oil']} million tonnes CO2 ({sector_percentages['oil']:.1f}% of total)
    - Gas: {sector_data['gas']} million tonnes CO2 ({sector_percentages['gas']:.1f}% of total)
    - Cement: {sector_data['cement']} million tonnes CO2 ({sector_percentages['cement']:.1f}% of total)
    - Trade: {sector_data['trade']} million tonnes CO2 ({sector_percentages['trade']:.1f}% of total)
    
    Based on scientific research and successful emissions reduction strategies from around the world, provide:
    
    1. A comprehensive overview of the challenge specific to {region}'s emissions profile
    2. Source-specific recommendations with realistic implementation timelines:
       - coal
       - oil
       - gas
       - cement
       - trade
    3. Policy mechanisms to achieve the targets (carbon pricing, regulations, incentives)
    4. Economic implications and just transition considerations
    5. Monitoring and verification approaches
    
    For each source, include 2-3 specific policy actions with quantified potential impact. 
    Provide concrete, actionable policies rather than general approaches. 
    Cite relevant examples from other regions where similar policies have succeeded.
    Format your response in clear, structured markdown with headers and bullet points.
    """
    
    # Call OpenAI API with the updated syntax for version 1.0+
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # Use appropriate model
            messages=[
                {"role": "system", "content": "You are a climate policy expert specializing in emission reduction strategies. Provide evidence-based recommendations grounded in real-world policy examples and scientific research."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the generated recommendations
        recommendations = response.choices[0].message.content
    except Exception as e:
        # Fallback if OpenAI call fails
        recommendations = f"""
        # Error Generating Policy Recommendations
        
        Unfortunately, an error occurred while generating policy recommendations: {str(e)}
        
        Please try again later or adjust your parameters.
        """
    
    # Generate trajectory data for visualization
    target_years = list(range(current_year, target_year + 1))
    current_trajectory = [latest_emission * (1 + metrics['avg_annual_pct_change']/100) ** i for i in range(len(target_years))]
    target_trajectory = [latest_emission * (1 - (i * annual_reduction_needed/100)) for i in range(len(target_years))]
    
    trajectories = {
        'years': target_years,
        'current': current_trajectory,
        'target': target_trajectory,
        'annual_reduction_needed': annual_reduction_needed
    }
    
    return recommendations, trajectories

def create_recommendation_chart(trajectories):
    """Create a chart comparing current and target emission trajectories"""
    fig = go.Figure()
    
    # Add current trajectory line
    fig.add_trace(go.Scatter(
        x=trajectories['years'],
        y=trajectories['current'],
        mode='lines',
        name='Current Trajectory',
        line=dict(color='#FF5252', width=2)
    ))
    
    # Add target trajectory line
    fig.add_trace(go.Scatter(
        x=trajectories['years'],
        y=trajectories['target'],
        mode='lines',
        name='Target Trajectory',
        line=dict(color='#4CAF50', width=2)
    ))
    
    # Calculate the gap in the final year
    final_year_gap = trajectories['current'][-1] - trajectories['target'][-1]
    final_year = trajectories['years'][-1]
    
    # Add annotation for the gap
    fig.add_annotation(
        x=final_year,
        y=(trajectories['current'][-1] + trajectories['target'][-1]) / 2,
        text=f"Gap: {final_year_gap:.1f} MT",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#880E4F",
        ax=70,
        ay=0
    )
    
    # Add annotation for required annual reduction
    fig.add_annotation(
        x=trajectories['years'][0] + (final_year - trajectories['years'][0]) / 2,
        y=trajectories['target'][0] - 100,
        text=f"Required Annual Reduction: {trajectories['annual_reduction_needed']:.1f}%",
        showarrow=False,
        font=dict(size=14, color="#880E4F")
    )
    
    # Update layout
    fig.update_layout(
        title="Emission Reduction Pathways",
        xaxis_title="Year",
        yaxis_title="CO₂ Emissions (million tonnes)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def generate_forecast_summary(forecast_df, metrics, region, emission_type, model_type):
    """Generate an AI summary of the forecast results"""
    client = get_openai_client()

    # Extract key forecast metrics
    latest_year = forecast_df['Year'].iloc[0]
    final_year = forecast_df['Year'].iloc[-1]
    latest_value = forecast_df['Forecasted_CO2'].iloc[0]
    final_value = forecast_df['Forecasted_CO2'].iloc[-1]
    total_change = metrics['total_pct_change']
    avg_annual_change = metrics['avg_annual_pct_change']
    forecasted_data = forecast_df[['Year', 'Forecasted_CO2']]
    
    # Create forecast context summary
    forecast_context = {
        'region': region,
        'emission_type': emission_type,
        'model_type': model_type,
        'forecast_period': f"{latest_year} to {final_year}",
        'start_emissions': latest_value,
        'end_emissions': final_value,
        'total_percent_change': total_change,
        'avg_annual_percent_change': avg_annual_change,
        'min_annual_change': metrics['min_annual_pct_change'],
        'max_annual_change': metrics['max_annual_pct_change'],
        'forecasted_data': forecasted_data.to_dict(orient='records')
    }
    
    # Generate detailed prompt for AI summary
    prompt = f"""
    As a climate scientist, analyze this emissions forecast for {region}'s {emission_type} emissions from {latest_year} to {final_year}.
    
    Key metrics:
    - Model used: {model_type}
    - Starting emissions: {latest_value:.2f} million tonnes
    - Ending emissions: {final_value:.2f} million tonnes
    - Total change over period: {total_change:.2f}%
    - Average annual change: {avg_annual_change:.2f}%
    - Range of annual changes: {metrics['min_annual_pct_change']:.2f}% to {metrics['max_annual_pct_change']:.2f}%
    
    Provide a concise scientific analysis of the forecast including:
    1. Don't talk about the model performance , only the forecasted data and the historical data.
    2. Main trend interpretation and its significance
    3. Key uncertainties and limitations of this forecast
    4. Anamolies where the Historical data forecast changes abruptly, suggest possible causes.
    
    
    Format your response in clear markdown with no more than 1-2 paragraphs total.
    """
    
    # Call OpenAI API for summary generation with updated syntax
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Use appropriate model
            messages=[
                {"role": "system", "content": "You are a climate scientist specializing in emissions forecasting and interpretation. Provide concise, scientifically accurate analysis of forecast data."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the generated summary
        summary = response.choices[0].message.content
    except Exception as e:
        # Fallback if OpenAI call fails
        summary = f"""
        # Forecast Summary
        
        This forecast projects {region}'s {emission_type} emissions from {latest_year} to {final_year}, using a {model_type} model. The model predicts a {total_change:.2f}% {'increase' if total_change > 0 else 'decrease'} over this period, with an average annual change of {avg_annual_change:.2f}%.
        
        Note: A more detailed AI analysis couldn't be generated due to a technical error: {str(e)}
        """
    
    return summary