import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from ultralytics import YOLO
import cv2
import plotly.graph_objects as go

# Load your model
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

def calculate_percentile_scores(sample_data, reference_percentiles):
    """Calculate scores based on percentile ranges for each metric"""
    scores = []
    metrics = ["num_plastics", "avg_size", "avg_aspect_ratio"]
    
    for i, metric in enumerate(metrics):
        value = sample_data[i]
        percentiles = reference_percentiles[metric]
        
        if value >= percentiles["90"]:
            scores.append(4)  # >90th percentile
        elif value >= percentiles["75"]:
            scores.append(3)  # 75-90th percentile
        elif value >= percentiles["50"]:
            scores.append(2)  # 50-75th percentile
        elif value >= percentiles["25"]:
            scores.append(1)  # 25-50th percentile
        else:
            scores.append(0)  # <25th percentile
            
    return scores

def classify_pollution(total_score):
    """Convert total score to pollution classification"""
    if total_score >= 10:
        return "Critically Polluted", "red", 4
    elif total_score >= 7:
        return "Highly Polluted", "orange", 3
    elif total_score >= 4:
        return "Moderately Polluted", "yellow", 2
    else:
        return "Less Polluted", "green", 1

def create_speedometer(level, title):
    """Create a speedometer gauge for pollution level"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = level,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': 'green'},
                {'range': [1, 2], 'color': 'yellow'},
                {'range': [2, 3], 'color': 'orange'},
                {'range': [3, 4], 'color': 'red'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': level}}))
    
    fig.update_layout(height=300)
    return fig

def create_metric_bar(value, max_value, percentiles, metric_name):
    """Create a horizontal bar for metric visualization with percentile lines"""
    fig = go.Figure()
    
    # Add the main bar
    fig.add_trace(go.Bar(
        x=[value],
        y=[metric_name],
        orientation='h',
        marker=dict(
            color='royalblue',
            line=dict(color='royalblue', width=1)
        ),
        text=f"{value:.1f}",
        textposition='outside',
        width=0.5
    ))
    
    # Add percentile lines
    percentile_colors = {'25': 'blue', '50': 'green', '75': 'orange', '90': 'red'}
    for percentile, color in percentile_colors.items():
        p_value = percentiles[percentile]
        fig.add_shape(
            type="line",
            x0=p_value,
            y0=-0.25,
            x1=p_value,
            y1=0.25,
            line=dict(color=color, width=2, dash="dash"),
        )
        fig.add_annotation(
            x=p_value,
            y=0.5,
            text=f"{percentile}th",
            showarrow=False,
            font=dict(size=10, color=color)
        )
    
    # Set layout
    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            range=[0, max(max_value*1.1, value*1.1)],
            showgrid=True,
            title=None
        ),
        yaxis=dict(
            showticklabels=False,
            title=None
        ),
        title=dict(
            text=metric_name,
            x=0.5,
            xanchor='center'
        ),
        showlegend=False
    )
    
    return fig

# Main app
def main():
    st.set_page_config(layout="wide")
    st.title("Microplastics Detection and Classification")
    
    # Load pre-calculated percentiles
    try:
        with open('percentile_values.json', 'r') as f:
            reference_percentiles = json.load(f)
    except FileNotFoundError:
        st.error("Percentile values file not found. Please run the dataset analysis script first.")
        return
    
    # Sidebar for file upload and options
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    # Load model
    model = load_model()
    
    if uploaded_file is not None:
        # Create two columns for original and processed images
        col1, col2 = st.columns(2)
        
        # Display original image
        image = Image.open(uploaded_file)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        if st.button("Detect Microplastics"):
            # Progress bar for processing
            progress_bar = st.progress(0)
            
            # Convert PIL Image to OpenCV format for model
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            progress_bar.progress(25)
            st.write("Processing image...")
            
            # Run detection
            results = model(img)
            boxes = results[0].boxes
            
            progress_bar.progress(50)
            
            # Draw detections on image for display
            img_with_boxes = results[0].plot()
            img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            
            progress_bar.progress(75)
            
            # Extract metrics
            if len(boxes) > 0:
                # Number of plastics
                num_plastics = len(boxes)
                
                # Calculate sizes
                if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    widths = xyxy[:, 2] - xyxy[:, 0]
                    heights = xyxy[:, 3] - xyxy[:, 1]
                    areas = widths * heights
                    avg_size = float(np.mean(areas))
                    
                    # Calculate aspect ratios
                    aspect_ratios = np.maximum(widths / heights, heights / widths)
                    avg_aspect_ratio = float(np.mean(aspect_ratios))
                else:
                    avg_size = 0
                    avg_aspect_ratio = 0
            else:
                num_plastics = 0
                avg_size = 0
                avg_aspect_ratio = 0
            
            # Calculate pollution scores
            sample_data = [num_plastics, avg_size, avg_aspect_ratio]
            scores = calculate_percentile_scores(sample_data, reference_percentiles)
            total_score = sum(scores)
            pollution_class, color, level = classify_pollution(total_score)
            
            progress_bar.progress(100)
            
            # Display image with detections
            with col2:
                st.subheader("Detection Results")
                st.image(img_with_boxes, use_column_width=True)
            
            # Results section with speedometer
            st.header("Pollution Analysis")
            
            # Display speedometer for pollution classification
            speedometer_col, metrics_col = st.columns([1, 2])
            
            with speedometer_col:
                st.subheader("Pollution Level")
                speedometer = create_speedometer(level, pollution_class)
                st.plotly_chart(speedometer, use_container_width=True)
                st.markdown(f"### Score: {total_score}/12 points")
            
            with metrics_col:
                st.subheader("Metrics Analysis")
                
                # Calculate max values for bars (from 90th percentile or actual value, whichever is higher)
                max_values = {
                    'num_plastics': max(reference_percentiles['num_plastics']['90'] * 1.5, num_plastics * 1.2),
                    'avg_size': max(reference_percentiles['avg_size']['90'] * 1.5, avg_size * 1.2),
                    'avg_aspect_ratio': max(reference_percentiles['avg_aspect_ratio']['90'] * 1.5, avg_aspect_ratio * 1.2)
                }
                
                # Create metric bars
                metric_names = ["Number of Microplastics", "Average Size (px²)", "Average Aspect Ratio"]
                for i, (metric, name) in enumerate(zip(['num_plastics', 'avg_size', 'avg_aspect_ratio'], metric_names)):
                    metric_fig = create_metric_bar(
                        sample_data[i], 
                        max_values[metric], 
                        reference_percentiles[metric],
                        name
                    )
                    st.plotly_chart(metric_fig, use_container_width=True)
                    
                    # Display score for each metric
                    st.caption(f"Score: {scores[i]}/4 points")
            
            # Try to load the dataset metrics for comparison
            try:
                metrics_df = pd.read_csv('dataset_metrics.csv')
                
                # Distribution charts comparing to dataset
                st.subheader("Comparison with Dataset Distribution")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                metrics = ['num_plastics', 'avg_size', 'avg_aspect_ratio']
                titles = ['Microplastic Count', 'Average Size (px²)', 'Average Aspect Ratio']
                
                for i, (metric, title) in enumerate(zip(metrics, titles)):
                    # Plot distribution
                    sns.histplot(metrics_df[metric], kde=True, ax=axes[i])
                    axes[i].set_title(title)
                    
                    # Add sample value line
                    axes[i].axvline(sample_data[i], color='red', linestyle='--', 
                                  label=f'Sample: {sample_data[i]:.2f}')
                    
                    # Add percentile lines
                    for p, color in zip([25, 50, 75, 90], ['blue', 'green', 'orange', 'purple']):
                        val = reference_percentiles[metric][str(p)]
                        axes[i].axvline(val, color=color, alpha=0.5, linestyle=':',
                                       label=f'{p}th: {val:.2f}')
                    
                    axes[i].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
            except FileNotFoundError:
                st.info("Dataset metrics file not found. Distribution comparison not available.")

if __name__ == "__main__":
    main()