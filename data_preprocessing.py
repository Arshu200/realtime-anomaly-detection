"""
Data preprocessing and EDA module for CPU usage anomaly detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class CPUUsageProcessor:
    """Data preprocessing and EDA for CPU usage time series."""
    
    def __init__(self, data_path: str):
        """Initialize with dataset path."""
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        
    def load_data(self):
        """Load and parse the dataset."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Parse timestamps and set as index
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        logger.info(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        return self.df
    
    def resample_data(self, frequency='30T', method='interpolate'):
        """
        Resample data to uniform time intervals.
        
        Args:
            frequency: Target frequency (e.g., '30T' for 30 minutes, '1H' for 1 hour)
            method: 'interpolate', 'forward_fill', or 'backward_fill'
        """
        logger.info(f"Resampling data to {frequency} intervals using {method}")
        
        # Set timestamp as index for resampling
        df_indexed = self.df.set_index('timestamp')
        
        # Resample to target frequency
        if method == 'interpolate':
            resampled = df_indexed.resample(frequency).interpolate(method='linear')
        elif method == 'forward_fill':
            resampled = df_indexed.resample(frequency).ffill()
        elif method == 'backward_fill':
            resampled = df_indexed.resample(frequency).bfill()
        else:
            raise ValueError("Method must be 'interpolate', 'forward_fill', or 'backward_fill'")
        
        # Fill any remaining NaN values
        resampled = resampled.ffill().bfill()
        
        # Reset index to get timestamp as column
        self.processed_df = resampled.reset_index()
        
        logger.info(f"Resampled data shape: {self.processed_df.shape}")
        return self.processed_df
    
    def compute_statistics(self):
        """Compute and log basic statistical summaries."""
        if self.processed_df is None:
            df_to_use = self.df
        else:
            df_to_use = self.processed_df
            
        stats = df_to_use['cpu_usage'].describe()
        
        logger.info("=== CPU Usage Statistics ===")
        logger.info(f"Mean: {stats['mean']:.4f}")
        logger.info(f"Std: {stats['std']:.4f}")
        logger.info(f"Min: {stats['min']:.4f}")
        logger.info(f"Max: {stats['max']:.4f}")
        logger.info(f"25th percentile: {stats['25%']:.4f}")
        logger.info(f"50th percentile: {stats['50%']:.4f}")
        logger.info(f"75th percentile: {stats['75%']:.4f}")
        
        # Additional statistics
        skewness = df_to_use['cpu_usage'].skew()
        kurtosis = df_to_use['cpu_usage'].kurtosis()
        
        logger.info(f"Skewness: {skewness:.4f}")
        logger.info(f"Kurtosis: {kurtosis:.4f}")
        
        return stats
    
    def plot_time_series(self, save_path='cpu_usage_timeseries.png'):
        """Plot the CPU usage time series."""
        if self.processed_df is None:
            df_to_use = self.df
        else:
            df_to_use = self.processed_df
            
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Main time series plot
        axes[0].plot(df_to_use['timestamp'], df_to_use['cpu_usage'], 
                    linewidth=1, alpha=0.8, color='steelblue')
        axes[0].set_title('CPU Usage Over Time', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Timestamp')
        axes[0].set_ylabel('CPU Usage')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution plot
        axes[1].hist(df_to_use['cpu_usage'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_title('CPU Usage Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('CPU Usage')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Time series plot saved to {save_path}")
    
    def plot_moving_averages(self, windows=[24, 48, 168], save_path='moving_averages.png'):
        """
        Plot moving averages with different window sizes.
        
        Args:
            windows: List of window sizes (in number of periods)
        """
        if self.processed_df is None:
            df_to_use = self.df.copy()
        else:
            df_to_use = self.processed_df.copy()
            
        plt.figure(figsize=(15, 8))
        
        # Original data
        plt.plot(df_to_use['timestamp'], df_to_use['cpu_usage'], 
                linewidth=1, alpha=0.6, label='Original', color='lightgray')
        
        # Moving averages
        colors = ['red', 'blue', 'green']
        for i, window in enumerate(windows):
            if len(df_to_use) >= window:
                ma = df_to_use['cpu_usage'].rolling(window=window, center=True).mean()
                plt.plot(df_to_use['timestamp'], ma, 
                        linewidth=2, label=f'MA {window}', color=colors[i % len(colors)])
        
        plt.title('CPU Usage with Moving Averages', fontsize=14, fontweight='bold')
        plt.xlabel('Timestamp')
        plt.ylabel('CPU Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Moving averages plot saved to {save_path}")
    
    def seasonal_decomposition(self, period=None, save_path='seasonal_decomposition.png'):
        """
        Perform seasonal decomposition of the time series.
        
        Args:
            period: Seasonality period. If None, will try to detect automatically.
        """
        if self.processed_df is None:
            df_to_use = self.df.copy()
        else:
            df_to_use = self.processed_df.copy()
            
        # Set timestamp as index
        ts_data = df_to_use.set_index('timestamp')['cpu_usage']
        
        # Determine period if not provided
        if period is None:
            # For daily seasonality with 40-min intervals: 24*60/40 = 36 periods per day
            # For weekly seasonality: 36*7 = 252 periods per week
            period = 36  # Daily seasonality
            
        logger.info(f"Performing seasonal decomposition with period={period}")
        
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=period)
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0], title='Original')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            
            for ax in axes:
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Seasonal decomposition plot saved to {save_path}")
            
            return decomposition
            
        except Exception as e:
            logger.warning(f"Seasonal decomposition failed: {e}")
            return None
    
    def check_stationarity(self):
        """Check stationarity using Augmented Dickey-Fuller test."""
        if self.processed_df is None:
            df_to_use = self.df
        else:
            df_to_use = self.processed_df
            
        result = adfuller(df_to_use['cpu_usage'].dropna())
        
        logger.info("=== Stationarity Test (ADF) ===")
        logger.info(f"ADF Statistic: {result[0]:.6f}")
        logger.info(f"p-value: {result[1]:.6f}")
        logger.info("Critical Values:")
        for key, value in result[4].items():
            logger.info(f"\t{key}: {value:.6f}")
            
        if result[1] <= 0.05:
            logger.info("Series is stationary")
        else:
            logger.info("Series is non-stationary")
            
        return result
    
    def generate_interactive_plot(self, save_path='interactive_cpu_usage.html'):
        """Generate interactive plotly visualization."""
        if self.processed_df is None:
            df_to_use = self.df
        else:
            df_to_use = self.processed_df
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU Usage Time Series', 'CPU Usage Distribution'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Time series plot
        fig.add_trace(
            go.Scatter(
                x=df_to_use['timestamp'],
                y=df_to_use['cpu_usage'],
                mode='lines',
                name='CPU Usage',
                line=dict(color='steelblue', width=1)
            ),
            row=1, col=1
        )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=df_to_use['cpu_usage'],
                nbinsx=50,
                name='Distribution',
                marker_color='lightcoral'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Interactive CPU Usage Analysis',
            height=800,
            showlegend=True
        )
        
        fig.write_html(save_path)
        logger.info(f"Interactive plot saved to {save_path}")
    
    def get_processed_data(self):
        """Return the processed dataset."""
        if self.processed_df is None:
            return self.df
        return self.processed_df
    
    def run_full_eda(self):
        """Run complete EDA pipeline."""
        logger.info("Starting complete EDA pipeline...")
        
        # Load data
        self.load_data()
        
        # Resample to 30-minute intervals
        self.resample_data(frequency='30T')
        
        # Compute statistics
        self.compute_statistics()
        
        # Generate visualizations
        self.plot_time_series()
        self.plot_moving_averages()
        self.seasonal_decomposition()
        self.generate_interactive_plot()
        
        # Check stationarity
        self.check_stationarity()
        
        logger.info("EDA pipeline completed successfully!")
        
        return self.processed_df


if __name__ == "__main__":
    # Example usage
    processor = CPUUsageProcessor("dataset.csv")
    processed_data = processor.run_full_eda()
    print(f"Processed data shape: {processed_data.shape}")