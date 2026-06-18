import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

st.markdown("""
    <style>
    /* Main styling */
    h1 { 
        color: white;
        text-align: center; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2 { 
        color: #667eea;
        border-bottom: 3px solid #764ba2;
        padding-bottom: 10px;
    }
    
    h3 {
        color: #667eea;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f2f6 0%, #e8eaf6 100%);
    }
    
    /* Container styling */
    [data-testid="stContainer"] {
        padding: 10px;
    }
    
    /* Radio button styling */
    .stRadio > label {
        font-size: 1.1em;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Selectbox styling */
    .stSelectbox > label {
        font-size: 1.05em;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_excel("Dry_Bean_Dataset.xlsx")
        # Remove duplicate columns if any
        data = data.loc[:, ~data.columns.duplicated(keep='first')]
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_dry_bean_model.pkl")
        return model
    except Exception as e:
        st.warning(f"Model loading error: {str(e)}")
        return None

df = load_data()
model = load_model()

if df is None:
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;'>
        <h2 style='margin: 0; font-size: 1.8em;'>🫘</h2>
        <h3 style='margin: 5px 0 0 0; color: white;'>Dry Bean Classification</h3>
        <p style='margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.9;'>Interactive Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Select Section:",
        ["📊 Visualizations", "🔮 Prediction"],
        index=0,
        label_visibility="collapsed"
    )

# ==================== VISUALIZATIONS PAGE ====================
if page == "📊 Visualizations":
    st.markdown("<h1>📊 Interactive Bean Feature Analysis</h1>", unsafe_allow_html=True)
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("No numeric columns found!")
        st.stop()
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px;'>
        <p style='margin: 0;'><b>🔍 Select a visualization type and explore the Dry Bean dataset features</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["📈 Feature Distribution", "🔥 Correlation Matrix", "🎯 Feature Scatter", "📦 Feature Box Plot"],
            index=0
        )
    
    st.markdown("---")
    
    try:
        if viz_type == "📈 Feature Distribution":
            st.markdown("### 📈 Feature Distribution Analysis")
            col = st.selectbox("Select Feature:", numeric_cols, key="dist_col")
            
            fig = px.histogram(
                df, 
                x=col, 
                nbins=40,
                title=f"Distribution of {col}",
                color_discrete_sequence=['#667eea'],
                labels={col: col}
            )
            fig.update_layout(
                height=450,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title=col,
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "🔥 Correlation Matrix":
            st.markdown("### 🔥 Feature Correlation Heatmap")
            st.write("Explore relationships between bean features")
            
            corr_matrix = df[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Viridis',
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text:.2f}',
                textfont={"size": 9},
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(
                title="Feature Correlation Matrix",
                height=650,
                xaxis_title="Features",
                yaxis_title="Features"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "🎯 Feature Scatter":
            st.markdown("### 🎯 Feature Relationship Scatter Plot")
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis Feature:", numeric_cols, key="x_scatter", index=0)
            with col2:
                y_col = st.selectbox("Y-axis Feature:", numeric_cols, key="y_scatter", index=1)
            
            if x_col != y_col:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col,
                    title=f"{x_col} vs {y_col}",
                    color_discrete_sequence=['#667eea'],
                    opacity=0.6,
                    height=450
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please select different features for X and Y axes")
        
        elif viz_type == "📦 Feature Box Plot":
            st.markdown("### 📦 Feature Distribution Box Plot")
            col = st.selectbox("Select Feature:", numeric_cols, key="box_plot")
            
            fig = px.box(
                df,
                y=col,
                title=f"Box Plot of {col}",
                color_discrete_sequence=['#667eea'],
                height=450
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="",
                yaxis_title=col
            )
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Visualization Error: {str(e)}")
        st.info("Try selecting different features or visualization type")

# ==================== PREDICTION PAGE ====================
elif page == "🔮 Prediction":
    st.markdown("<h1>🔮 Bean Type Classification Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if model is None:
        st.error("❌ Model not loaded. Please check model files.")
        st.stop()
    
    # Get all numeric columns except the last one (which is the target)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    model_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(numeric_cols)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;'>
        <h3 style='margin: 0;'>📊 Enter Bean Measurements</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        input_values = {}
        
        # Create a more attractive layout with 2 columns - reduced padding
        cols = st.columns(2)
        col_idx = 0
        
        for feature in numeric_cols:
            with cols[col_idx % 2]:
                # Get min and max from dataset
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                # Handle edge case where min equals max
                if min_val == max_val:
                    min_val = min_val * 0.95 if min_val != 0 else 0
                    max_val = max_val * 1.05 if max_val != 0 else 1
                
                # Create a container for better styling - reduced padding
                with st.container(border=True):
                    st.write(f"**{feature}**")
                    input_values[feature] = st.number_input(
                        f"",
                        min_value=int(min_val) if min_val >= 0 else int(min_val) - 1,
                        max_value=int(max_val) + 1,
                        value=int(mean_val),
                        step=1,
                        key=f"input_{feature}",
                        label_visibility="collapsed"
                    )
            col_idx += 1
        
        st.markdown("---")
        
        # Create prediction section with better styling
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("")
        
        with col2:
            predict_button = st.button(
                "🚀 Classify Bean", 
                use_container_width=True, 
                key="predict_btn",
                type="primary"
            )
        
        if predict_button:
            try:
                # Verify all features are provided
                if len(input_values) != len(numeric_cols):
                    st.error(f"❌ Missing features. Got {len(input_values)} but need {len(numeric_cols)}")
                    st.stop()
                
                # Prepare input with all features in the correct order
                input_array = np.array([[input_values[feature] for feature in numeric_cols]])
                
                # Make prediction
                prediction = model.predict(input_array)[0]
                
                st.markdown("---")
                
                # Beautiful prediction result section
                st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin: 20px 0;'>
                    <h2 style='margin: 0; font-size: 2em;'>✅ Prediction Complete!</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Display prediction with fancy styling
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div style='background: #f0f2f6; padding: 25px; border-radius: 12px; text-align: center; border-left: 5px solid #667eea; border-right: 5px solid #764ba2;'>
                        <p style='color: #667eea; font-size: 0.85em; margin: 0 0 8px 0; font-weight: 600;'>🫘 PREDICTED BEAN TYPE</p>
                        <h1 style='margin: 0; color: #764ba2; font-size: 2.2em;'>{prediction}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence scores section
                if hasattr(model, 'predict_proba'):
                    st.markdown("---")
                    st.markdown("<h3 style='text-align: center; color: #667eea;'>📊 Classification Confidence</h3>", unsafe_allow_html=True)
                    
                    proba = model.predict_proba(input_array)[0]
                    classes = model.classes_
                    prob_df = pd.DataFrame({
                        'Bean Type': classes,
                        'Confidence (%)': (proba * 100).round(2)
                    }).sort_values('Confidence (%)', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        top_confidence = prob_df.iloc[0]['Confidence (%)']
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white; text-align: center;'>
                            <p style='margin: 0; font-size: 0.85em;'>TOP CONFIDENCE</p>
                            <h2 style='margin: 8px 0 0 0; font-size: 1.8em;'>{top_confidence}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); padding: 15px; border-radius: 8px; color: white; text-align: center;'>
                            <p style='margin: 0; font-size: 0.85em;'>BEAN CLASSES</p>
                            <h2 style='margin: 8px 0 0 0; font-size: 1.8em;'>{len(classes)}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                    # Beautiful confidence chart
                    fig = px.bar(
                        prob_df, 
                        x='Bean Type', 
                        y='Confidence (%)',
                        title='Classification Confidence Scores',
                        color='Confidence (%)',
                        color_continuous_scale='Viridis',
                        text='Confidence (%)',
                        height=420
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Bean Type",
                        yaxis_title="Confidence (%)",
                        font=dict(size=11),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display input summary - more compact
                st.markdown("---")
                st.markdown("<h3 style='color: #667eea;'>📋 Input Summary (All 16 Features)</h3>", unsafe_allow_html=True)
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown("**Features 1-8:**")
                    for i, feature in enumerate(numeric_cols[:8], 1):
                        st.write(f"{i}. {feature}: **{input_values[feature]}**")
                
                with summary_col2:
                    st.markdown("**Features 9-16:**")
                    for i, feature in enumerate(numeric_cols[8:], 9):
                        st.write(f"{i}. {feature}: **{input_values[feature]}**")
                
                st.success("✅ Classification completed successfully!")
            
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("Please check that all input values are valid integers.")
    
    except Exception as e:
        st.error(f"❌ Error in prediction interface: {str(e)}")
        st.write(f"Debug info: {str(e)}")

