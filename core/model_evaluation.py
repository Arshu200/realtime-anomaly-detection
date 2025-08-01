"""
Model evaluation module for CPU anomaly detection system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive evaluation for anomaly detection models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = {}
        
    def simulate_anomalies(self, df, anomaly_ratio=0.05, anomaly_types=['spike', 'dip', 'drift']):
        """
        Simulate anomalies in the dataset for evaluation purposes.
        
        Args:
            df: DataFrame with timestamp and cpu_usage columns
            anomaly_ratio: Percentage of data points to make anomalous
            anomaly_types: Types of anomalies to simulate
            
        Returns:
            DataFrame with simulated anomalies and labels
        """
        logger.info(f"Simulating anomalies with {anomaly_ratio*100}% anomaly ratio")
        
        df_sim = df.copy()
        df_sim['is_true_anomaly'] = False
        df_sim['anomaly_type'] = 'normal'
        
        num_anomalies = int(len(df_sim) * anomaly_ratio)
        anomaly_indices = np.random.choice(len(df_sim), num_anomalies, replace=False)
        
        # Calculate baseline statistics
        cpu_mean = df_sim['cpu_usage'].mean()
        cpu_std = df_sim['cpu_usage'].std()
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(anomaly_types)
            
            if anomaly_type == 'spike':
                # Create upward spike (2-4 standard deviations above mean)
                spike_magnitude = np.random.uniform(2, 4) * cpu_std
                df_sim.loc[idx, 'cpu_usage'] = min(1.0, cpu_mean + spike_magnitude)
                df_sim.loc[idx, 'anomaly_type'] = 'spike'
                
            elif anomaly_type == 'dip':
                # Create downward dip (2-4 standard deviations below mean)
                dip_magnitude = np.random.uniform(2, 4) * cpu_std
                df_sim.loc[idx, 'cpu_usage'] = max(0.0, cpu_mean - dip_magnitude)
                df_sim.loc[idx, 'anomaly_type'] = 'dip'
                
            elif anomaly_type == 'drift':
                # Create gradual drift over several points
                drift_length = min(10, len(df_sim) - idx)
                drift_magnitude = np.random.uniform(1.5, 3) * cpu_std
                drift_direction = np.random.choice([-1, 1])
                
                for i in range(drift_length):
                    if idx + i < len(df_sim):
                        drift_value = drift_magnitude * (i / drift_length) * drift_direction
                        new_value = df_sim.loc[idx + i, 'cpu_usage'] + drift_value
                        df_sim.loc[idx + i, 'cpu_usage'] = np.clip(new_value, 0.0, 1.0)
                        df_sim.loc[idx + i, 'anomaly_type'] = 'drift'
            
            df_sim.loc[idx, 'is_true_anomaly'] = True
        
        logger.info(f"Simulated {num_anomalies} anomalies:")
        logger.info(f"- Spikes: {sum(df_sim['anomaly_type'] == 'spike')}")
        logger.info(f"- Dips: {sum(df_sim['anomaly_type'] == 'dip')}")
        logger.info(f"- Drifts: {sum(df_sim['anomaly_type'] == 'drift')}")
        
        return df_sim
    
    def evaluate_detection_performance(self, y_true, y_pred, y_scores=None):
        """
        Evaluate detection performance with various metrics.
        
        Args:
            y_true: True anomaly labels (binary)
            y_pred: Predicted anomaly labels (binary)
            y_scores: Anomaly scores (continuous, optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating detection performance...")
        
        # Basic classification metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'confusion_matrix': cm.tolist()
        }
        
        # AUC metrics if scores are provided
        if y_scores is not None:
            try:
                auc_roc = roc_auc_score(y_true, y_scores)
                results['auc_roc'] = auc_roc
            except ValueError:
                logger.warning("Could not calculate AUC-ROC (might have only one class)")
                results['auc_roc'] = None
        
        # Log results
        logger.info("=== Detection Performance ===")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        logger.info(f"False Positive Rate: {false_positive_rate:.4f}")
        logger.info(f"True Positives: {tp}")
        logger.info(f"False Positives: {fp}")
        logger.info(f"True Negatives: {tn}")
        logger.info(f"False Negatives: {fn}")
        
        self.evaluation_results = results
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    def plot_roc_curve(self, y_true, y_scores, save_path='roc_curve.png'):
        """Plot ROC curve."""
        if y_scores is None:
            logger.warning("Cannot plot ROC curve without anomaly scores")
            return
            
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auc_score = roc_auc_score(y_true, y_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve plot saved to {save_path}")
            
        except ValueError as e:
            logger.warning(f"Could not plot ROC curve: {e}")
    
    def plot_precision_recall_curve(self, y_true, y_scores, save_path='precision_recall_curve.png'):
        """Plot Precision-Recall curve."""
        if y_scores is None:
            logger.warning("Cannot plot PR curve without anomaly scores")
            return
            
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Precision-Recall curve plot saved to {save_path}")
            
        except ValueError as e:
            logger.warning(f"Could not plot PR curve: {e}")
    
    def plot_detection_overview(self, df_with_detections, save_path='detection_overview.png'):
        """
        Plot comprehensive detection overview.
        
        Args:
            df_with_detections: DataFrame with true labels, predictions, and scores
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Time series with true and detected anomalies
        axes[0].plot(df_with_detections['timestamp'], df_with_detections['cpu_usage'], 
                    linewidth=1, alpha=0.7, color='blue', label='CPU Usage')
        
        # True anomalies
        true_anomalies = df_with_detections[df_with_detections['is_true_anomaly']]
        if len(true_anomalies) > 0:
            axes[0].scatter(true_anomalies['timestamp'], true_anomalies['cpu_usage'], 
                          color='red', s=50, label='True Anomalies', marker='x', alpha=0.8)
        
        # Detected anomalies
        detected_anomalies = df_with_detections[df_with_detections['is_detected_anomaly']]
        if len(detected_anomalies) > 0:
            axes[0].scatter(detected_anomalies['timestamp'], detected_anomalies['cpu_usage'], 
                          color='orange', s=30, label='Detected Anomalies', marker='o', alpha=0.6)
        
        axes[0].set_title('True vs Detected Anomalies', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('CPU Usage')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores over time
        if 'anomaly_score' in df_with_detections.columns:
            axes[1].plot(df_with_detections['timestamp'], df_with_detections['anomaly_score'], 
                        linewidth=1, alpha=0.8, color='purple')
            
            # Highlight true anomalies
            if len(true_anomalies) > 0:
                axes[1].scatter(true_anomalies['timestamp'], 
                              true_anomalies['anomaly_score'] if 'anomaly_score' in true_anomalies.columns else [0]*len(true_anomalies), 
                              color='red', s=30, alpha=0.8)
            
            axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Threshold')
            axes[1].set_title('Anomaly Scores Over Time', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Anomaly Score')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution of anomaly scores for normal vs anomalous points
        if 'anomaly_score' in df_with_detections.columns:
            normal_scores = df_with_detections[~df_with_detections['is_true_anomaly']]['anomaly_score']
            anomaly_scores = df_with_detections[df_with_detections['is_true_anomaly']]['anomaly_score']
            
            bins = np.linspace(0, df_with_detections['anomaly_score'].max(), 30)
            
            axes[2].hist(normal_scores, bins=bins, alpha=0.7, label='Normal Points', 
                        color='blue', density=True)
            if len(anomaly_scores) > 0:
                axes[2].hist(anomaly_scores, bins=bins, alpha=0.7, label='True Anomalies', 
                            color='red', density=True)
            
            axes[2].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Threshold')
            axes[2].set_title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Anomaly Score')
            axes[2].set_ylabel('Density')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detection overview plot saved to {save_path}")
    
    def analyze_detection_errors(self, df_with_detections):
        """Analyze false positives and false negatives."""
        logger.info("Analyzing detection errors...")
        
        # False Positives: Detected as anomaly but actually normal
        false_positives = df_with_detections[
            (df_with_detections['is_detected_anomaly']) & 
            (~df_with_detections['is_true_anomaly'])
        ]
        
        # False Negatives: Actually anomaly but not detected
        false_negatives = df_with_detections[
            (~df_with_detections['is_detected_anomaly']) & 
            (df_with_detections['is_true_anomaly'])
        ]
        
        # True Positives: Correctly detected anomalies
        true_positives = df_with_detections[
            (df_with_detections['is_detected_anomaly']) & 
            (df_with_detections['is_true_anomaly'])
        ]
        
        analysis = {
            'false_positives': {
                'count': len(false_positives),
                'percentage': len(false_positives) / len(df_with_detections) * 100,
                'avg_cpu_usage': false_positives['cpu_usage'].mean() if len(false_positives) > 0 else 0,
                'avg_anomaly_score': false_positives['anomaly_score'].mean() if len(false_positives) > 0 and 'anomaly_score' in false_positives.columns else 0
            },
            'false_negatives': {
                'count': len(false_negatives),
                'percentage': len(false_negatives) / len(df_with_detections) * 100,
                'avg_cpu_usage': false_negatives['cpu_usage'].mean() if len(false_negatives) > 0 else 0,
                'avg_anomaly_score': false_negatives['anomaly_score'].mean() if len(false_negatives) > 0 and 'anomaly_score' in false_negatives.columns else 0
            },
            'true_positives': {
                'count': len(true_positives),
                'percentage': len(true_positives) / len(df_with_detections) * 100,
                'avg_cpu_usage': true_positives['cpu_usage'].mean() if len(true_positives) > 0 else 0,
                'avg_anomaly_score': true_positives['anomaly_score'].mean() if len(true_positives) > 0 and 'anomaly_score' in true_positives.columns else 0
            }
        }
        
        logger.info("=== Error Analysis ===")
        logger.info(f"False Positives: {analysis['false_positives']['count']} ({analysis['false_positives']['percentage']:.2f}%)")
        logger.info(f"False Negatives: {analysis['false_negatives']['count']} ({analysis['false_negatives']['percentage']:.2f}%)")
        logger.info(f"True Positives: {analysis['true_positives']['count']} ({analysis['true_positives']['percentage']:.2f}%)")
        
        return analysis
    
    def evaluate_detection_rate_by_threshold(self, y_true, y_scores, thresholds=None):
        """Evaluate detection performance across different thresholds."""
        if y_scores is None:
            logger.warning("Cannot evaluate thresholds without anomaly scores")
            return None
            
        if thresholds is None:
            thresholds = np.linspace(0.1, 3.0, 30)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold (highest F1)
        optimal_idx = results_df['f1_score'].idxmax()
        optimal_threshold = results_df.loc[optimal_idx, 'threshold']
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f} (F1: {results_df.loc[optimal_idx, 'f1_score']:.3f})")
        
        return results_df, optimal_threshold
    
    def plot_threshold_analysis(self, threshold_results, save_path='threshold_analysis.png'):
        """Plot threshold analysis results."""
        if threshold_results is None:
            return
            
        plt.figure(figsize=(10, 6))
        
        plt.plot(threshold_results['threshold'], threshold_results['precision'], 
                linewidth=2, label='Precision', marker='o', markersize=4)
        plt.plot(threshold_results['threshold'], threshold_results['recall'], 
                linewidth=2, label='Recall', marker='s', markersize=4)
        plt.plot(threshold_results['threshold'], threshold_results['f1_score'], 
                linewidth=2, label='F1-Score', marker='^', markersize=4)
        
        # Mark optimal threshold
        optimal_idx = threshold_results['f1_score'].idxmax()
        optimal_threshold = threshold_results.loc[optimal_idx, 'threshold']
        optimal_f1 = threshold_results.loc[optimal_idx, 'f1_score']
        
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal Threshold ({optimal_threshold:.2f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Detection Performance vs Threshold', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Threshold analysis plot saved to {save_path}")
    
    def generate_evaluation_report(self, save_path='evaluation_report.txt'):
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_results:
            logger.warning("No evaluation results to report")
            return
            
        report = f"""
=== CPU Anomaly Detection Model Evaluation Report ===
Generated at: {pd.Timestamp.now()}

CLASSIFICATION METRICS:
- Precision: {self.evaluation_results['precision']:.4f}
- Recall: {self.evaluation_results['recall']:.4f}
- F1-Score: {self.evaluation_results['f1_score']:.4f}
- Specificity: {self.evaluation_results['specificity']:.4f}

ERROR RATES:
- False Positive Rate: {self.evaluation_results['false_positive_rate']:.4f}
- False Negative Rate: {self.evaluation_results['false_negative_rate']:.4f}

CONFUSION MATRIX:
- True Positives: {self.evaluation_results['true_positives']}
- False Positives: {self.evaluation_results['false_positives']}
- True Negatives: {self.evaluation_results['true_negatives']}
- False Negatives: {self.evaluation_results['false_negatives']}

ADDITIONAL METRICS:
"""
        
        if 'auc_roc' in self.evaluation_results and self.evaluation_results['auc_roc'] is not None:
            report += f"- AUC-ROC: {self.evaluation_results['auc_roc']:.4f}\n"
        
        report += f"""
RECOMMENDATIONS:
- Precision of {self.evaluation_results['precision']:.2f} means {self.evaluation_results['precision']*100:.1f}% of detected anomalies are true anomalies
- Recall of {self.evaluation_results['recall']:.2f} means {self.evaluation_results['recall']*100:.1f}% of true anomalies are detected
- F1-Score of {self.evaluation_results['f1_score']:.2f} represents the harmonic mean of precision and recall

"""
        
        # Add recommendations based on results
        if self.evaluation_results['precision'] < 0.5:
            report += "- Consider increasing the anomaly threshold to reduce false positives\n"
        if self.evaluation_results['recall'] < 0.5:
            report += "- Consider decreasing the anomaly threshold to catch more anomalies\n"
        if self.evaluation_results['f1_score'] > 0.8:
            report += "- Model shows excellent performance\n"
        elif self.evaluation_results['f1_score'] > 0.6:
            report += "- Model shows good performance\n"
        else:
            report += "- Model performance needs improvement - consider tuning parameters\n"
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # This would typically be used with real model outputs
    # See the main integration for complete usage example