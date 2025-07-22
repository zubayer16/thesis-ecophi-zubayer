from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
import logging
import time
import psutil
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Time Series Segmentation Visualization API",
    description="API for visualizing time series segmentation model predictions",
    version="1.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables to store model and data
model = None
training_data = None
ts_seg = None
device = None

class TimeSeriesVisualizationAPI:
    def __init__(self):
        self.model = None
        self.training_data = None
        self.ts_seg = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_cache = {}
        
    def get_hardware_info(self):
        """Get computational hardware information"""
        hardware_info = {
            "device": str(self.device),
            "cpu_model": platform.processor() or "Unknown",
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "cpu_frequency": f"{psutil.cpu_freq().current:.1f} MHz" if psutil.cpu_freq() else "Unknown",
            "total_memory": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "available_memory": f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
        }
        
        # Add GPU information if CUDA is available
        if torch.cuda.is_available():
            hardware_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / (1024**3):.3f} GB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / (1024**3):.3f} GB",
                "cuda_version": torch.version.cuda,
            })
        else:
            hardware_info.update({
                "gpu_name": "Not Available",
                "gpu_info": "CUDA not available - using CPU"
            })
            
        return hardware_info
        
    def load_model_and_data(self, model_path: str = "models/final_model.pth"):
        """Load the trained model and training data"""
        try:
            # Load your TimeSeriesSegmentation class and model
            # You'll need to import your actual classes here
            from pipeline import TimeSeriesSegmentation
            from model import CNN_GRU_Attention
            
            self.ts_seg = TimeSeriesSegmentation()
            
            # Load training data (you might want to save this during training)
            rms_df_with_labels, rms_df_with_labels2 = self.ts_seg.prepare_data()
            self.training_data = self.ts_seg.prepare_training_data(rms_df_with_labels, rms_df_with_labels2)
            
            # Load model
            input_shape = 2  # Adjust based on your model
            self.model = CNN_GRU_Attention(input_shape, hidden_dim=64, num_layers=3, dropout_rate=0.1)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model and data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model and data: {e}")
            return False
    
    def generate_predictions(self, phase_type: str, phase_num: int, num_samples: int):
        """Generate predictions for the specified phase type and number"""
        if self.model is None or self.training_data is None:
            raise Exception("Model or training data not loaded")
        
        # Get the appropriate test data based on phase_type and phase_num
        if phase_type == "voltage":
            X_test_key = f"X_test_v_phase{phase_num}_seq"
            time_test_key = f"time_test_phase{phase_num}_seq"
        else:  # current
            X_test_key = f"X_test_c_phase{phase_num}_seq"
            time_test_key = f"time_test_phase{phase_num}_seq"
        
        y_test_key = f"y_test_phase{phase_num}_seq"
        
        # Get test data
        X_test = self.training_data[X_test_key]
        y_test = self.training_data[y_test_key]
        time_test = self.training_data.get(time_test_key)
        
        # Limit number of samples
        num_samples = min(num_samples, len(X_test))
        X_test = X_test[:num_samples]
        y_test = y_test[:num_samples]
        if time_test is not None:
            time_test = time_test[:num_samples]
        
        # Generate predictions with timing
        self.model.eval()
        
        # Warm-up run (not counted in timing)
        with torch.no_grad():
            if len(X_test) > 0:
                X_warmup = torch.tensor(X_test[:1], dtype=torch.float32).to(self.device)
                _ = self.model(X_warmup)
        
        # Actual inference with timing
        start_time = time.time()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            predictions = self.model(X_test_tensor).cpu().numpy()
        end_time = time.time()
        
        # Calculate timing metrics
        total_inference_time = end_time - start_time
        avg_inference_time_per_sample = total_inference_time / num_samples if num_samples > 0 else 0
        
        return {
            "predictions": predictions.tolist(),
            "ground_truth": y_test.tolist(),
            "input_data": X_test.tolist(),
            "time_data": time_test.tolist() if time_test is not None else None,
            "phase_type": phase_type,
            "phase_num": phase_num,
            "total_inference_time": total_inference_time,
            "avg_inference_time_per_sample": avg_inference_time_per_sample,
            "samples_per_second": num_samples / total_inference_time if total_inference_time > 0 else 0
        }

    def create_visualization_plot(self, predictions_data, plot_type: str):
        """Create visualization plot and return as base64 string"""
        predictions = np.array(predictions_data["predictions"])
        ground_truth = np.array(predictions_data["ground_truth"])
        input_data = np.array(predictions_data["input_data"])
        time_data = predictions_data["time_data"]
        phase_type = predictions_data["phase_type"]
        phase_num = predictions_data["phase_num"]
        
        # Set a modern style with larger fonts
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Set global font sizes
        plt.rcParams.update({
            'font.size': 14,          # Base font size
            'axes.titlesize': 18,     # Title font size
            'axes.labelsize': 16,     # Axis label font size
            'xtick.labelsize': 14,    # X-tick label font size
            'ytick.labelsize': 14,    # Y-tick label font size
            'legend.fontsize': 14,    # Legend font size
            'figure.titlesize': 20    # Figure title font size
        })
        
        if plot_type == "comparison":
            # Create comparison dashboard with all samples
            num_samples = len(predictions)  # Show all samples
            
            # Calculate grid size for displaying all samples
            cols = 2  # 2 columns for better readability
            rows = (num_samples + cols - 1) // cols  # Calculate rows needed
            
            # Increased figure size significantly
            fig, axes = plt.subplots(rows, cols, figsize=(28, 6*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            if num_samples == 1:
                axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
            else:
                axes = axes.flatten()
            
            for i in range(num_samples):
                ax = axes[i]
                
                # Get sample data
                sample_input = input_data[i]
                sample_time = time_data[i] if time_data else np.arange(len(sample_input))
                
                # Plot only the first sensor/input data with thicker line
                ax.plot(sample_time, sample_input[:, 0], alpha=0.8, color="#5A37B9", linewidth=4)
                
                # Plot predictions vs ground truth with thicker lines
                pred_start, pred_end = predictions[i]
                true_start, true_end = ground_truth[i]
                
                ax.axvspan(pred_start, pred_end, color='#FFD700', alpha=0.4, label='Predicted')
                ax.axvline(true_start, color='#DC143C', linestyle='--', linewidth=4, label='True Start')
                ax.axvline(true_end, color='#228B22', linestyle='--', linewidth=4, label='True End')
                
                # Larger titles and labels
                ax.set_title(f'Sample {i+1}', fontsize=18, fontweight='bold', pad=15)
                ax.set_xlabel('Time', fontsize=16, fontweight='bold')
                ax.set_ylabel(phase_type.title(), fontsize=16, fontweight='bold')
                
                if i == 0:  # Only show legend on first plot to avoid clutter
                    ax.legend(fontsize=14, loc='upper right')
                
                # Enhanced grid
                ax.grid(True, alpha=0.4, linewidth=1)
                
                # Larger tick labels
                ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
            
            # Hide unused subplots
            for i in range(num_samples, len(axes)):
                axes[i].set_visible(False)
                
                        
        else:  # time_series view
            # Show single sample with detailed view - much larger figure
            sample_idx = 0
            fig, ax = plt.subplots(1, 1, figsize=(20, 12))
            
            sample_input = input_data[sample_idx]
            sample_time = time_data[sample_idx] if time_data else np.arange(len(sample_input))
            
            # Plot only the first input data with thicker line
            ax.plot(sample_time, sample_input[:, 0], 
                label='Input Signal', alpha=0.9, color='#5A37B9', linewidth=5)
            
            # Plot predictions vs ground truth with thicker lines
            pred_start, pred_end = predictions[sample_idx]
            true_start, true_end = ground_truth[sample_idx]
            
            ax.axvspan(pred_start, pred_end, color='#FFD700', alpha=0.4, label='Predicted Event')
            ax.axvline(true_start, color='#DC143C', linestyle='--', linewidth=5, label='True Start')
            ax.axvline(true_end, color='#228B22', linestyle='--', linewidth=5, label='True End')
            
            # Much larger titles and labels
            ax.set_title(f'{phase_type.title()} Phase {phase_num} - Time Series Detail View', 
                        fontsize=24, fontweight='bold', pad=25)
            ax.set_xlabel('Time', fontsize=20, fontweight='bold')
            ax.set_ylabel(f'{phase_type.title()} Values', fontsize=20, fontweight='bold')
            
            # Larger legend
            ax.legend(fontsize=16, loc='upper right', framealpha=0.9)
            
            # Enhanced grid
            ax.grid(True, alpha=0.4, linewidth=1.2)
            
            # Much larger tick labels
            ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=8)
        
        plt.tight_layout()
        
        # Convert plot to base64 with higher DPI for better quality
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Reset matplotlib rcParams to default to avoid affecting other plots
        plt.rcParams.update(plt.rcParamsDefault)
        
        return plot_base64

# Initialize the API class
viz_api = TimeSeriesVisualizationAPI()

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    logger.info("Starting up FastAPI application...")
    success = viz_api.load_model_and_data()
    if not success:
        logger.warning("Failed to load model and data on startup")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface"""
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: static/index.html not found</h1>")

@app.get("/visualize")
async def visualize_predictions(
    phase_type: str = "voltage",
    phase_num: int = 1,
    num_samples: int = 50,
    plot_type: str = "comparison"
):
    """Generate visualization for the specified parameters"""
    try:
        # Generate predictions (includes timing information)
        predictions_data = viz_api.generate_predictions(phase_type, phase_num, num_samples)
        
        # Get hardware information
        hardware_info = viz_api.get_hardware_info()
        
        # Create plot
        plot_base64 = viz_api.create_visualization_plot(predictions_data, plot_type)
        
        return {
            "plot": plot_base64,
            "predictions": predictions_data["predictions"],
            "ground_truth": predictions_data["ground_truth"],
            "phase_type": phase_type,
            "phase_num": phase_num,
            "num_samples": len(predictions_data["predictions"]),
            "total_inference_time": predictions_data["total_inference_time"],
            "avg_inference_time_per_sample": predictions_data["avg_inference_time_per_sample"],
            "samples_per_second": predictions_data["samples_per_second"],
            "hardware_info": hardware_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in visualization endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": viz_api.model is not None,
        "data_loaded": viz_api.training_data is not None,
        "device": str(viz_api.device),
        "hardware_info": viz_api.get_hardware_info(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if viz_api.model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    return {
        "model_type": type(viz_api.model).__name__,
        "device": str(viz_api.device),
        "training_data_keys": list(viz_api.training_data.keys()) if viz_api.training_data else [],
        "model_parameters": sum(p.numel() for p in viz_api.model.parameters()),
        "model_trainable_parameters": sum(p.numel() for p in viz_api.model.parameters() if p.requires_grad),
        "hardware_info": viz_api.get_hardware_info()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)