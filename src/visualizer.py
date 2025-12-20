"""
Visualization functions for clustering results
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import streamlit as st


class Visualizer:
    """Class for creating visualizations"""
    
    @staticmethod
    def plot_clustering_result(X, labels, title="Clustering Result"):
        """
        Plot clustering result
        
        Parameters:
        -----------
        X : ndarray
            Data points
        labels : ndarray
            Cluster labels
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create scatter plot
        scatter = ax.scatter(
            X[:, 0], 
            X[:, 1],
            c=labels,
            cmap='viridis',
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.8
        )
        
        # Styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Feature 1', fontsize=13, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_dendrogram(X, method='ward', title="Dendrogram"):
        """
        Plot dendrogram
        
        Parameters:
        -----------
        X : ndarray
            Data points
        method : str
            Linkage method
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate linkage
        Z = linkage(X, method=method)
        
        # Plot dendrogram
        dendrogram(
            Z,
            ax=ax,
            color_threshold=0,
            above_threshold_color='gray'
        )
        
        # Styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Sample Index', fontsize=13, fontweight='bold')
        ax.set_ylabel('Distance', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_linkage_comparison(X, results, n_clusters):
        """
        Plot comparison of different linkage methods
        
        Parameters:
        -----------
        X : ndarray
            Data points
        results : dict
            Dictionary with linkage methods as keys and labels as values
        n_clusters : int
            Number of clusters
        """
        n_methods = len(results)
        fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 12))
        
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (method, labels) in enumerate(results.items()):
            # Plot clustering result
            scatter = axes[0, idx].scatter(
                X[:, 0], 
                X[:, 1],
                c=labels,
                cmap='viridis',
                s=60,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.6
            )
            
            axes[0, idx].set_title(
                f'{method.upper()} Linkage\nClusters', 
                fontsize=14, 
                fontweight='bold'
            )
            axes[0, idx].set_xlabel('Feature 1', fontsize=11)
            axes[0, idx].set_ylabel('Feature 2', fontsize=11)
            axes[0, idx].grid(True, alpha=0.3, linestyle='--')
            plt.colorbar(scatter, ax=axes[0, idx], label='Cluster')
            
            # Plot dendrogram
            Z = linkage(X, method=method)
            dendrogram(
                Z, 
                ax=axes[1, idx],
                color_threshold=0,
                above_threshold_color='gray'
            )
            
            axes[1, idx].set_title(
                f'{method.upper()} Linkage\nDendrogram',
                fontsize=14,
                fontweight='bold'
            )
            axes[1, idx].set_xlabel('Sample Index', fontsize=11)
            axes[1, idx].set_ylabel('Distance', fontsize=11)
            axes[1, idx].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_metrics_comparison(cluster_range, metrics_dict):
        """
        Plot metrics for different numbers of clusters
        
        Parameters:
        -----------
        cluster_range : range
            Range of cluster numbers
        metrics_dict : dict
            Dictionary with metric names as keys and lists as values
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Silhouette Score
        axes[0].plot(
            cluster_range, 
            metrics_dict['silhouette'],
            marker='o',
            linewidth=2.5,
            markersize=8,
            color='#2196f3'
        )
        axes[0].set_title('Silhouette Score\n(Higher is Better)', 
                         fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Davies-Bouldin Index
        axes[1].plot(
            cluster_range,
            metrics_dict['davies_bouldin'],
            marker='s',
            linewidth=2.5,
            markersize=8,
            color='#ff9800'
        )
        axes[1].set_title('Davies-Bouldin Index\n(Lower is Better)',
                         fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Number of Clusters', fontsize=12)
        axes[1].set_ylabel('Index', fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # Calinski-Harabasz Score
        axes[2].plot(
            cluster_range,
            metrics_dict['calinski_harabasz'],
            marker='^',
            linewidth=2.5,
            markersize=8,
            color='#4caf50'
        )
        axes[2].set_title('Calinski-Harabasz Score\n(Higher is Better)',
                         fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Number of Clusters', fontsize=12)
        axes[2].set_ylabel('Score', fontsize=12)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_customer_segmentation(df, labels):
        """
        Plot customer segmentation results
        
        Parameters:
        -----------
        df : DataFrame
            Customer data
        labels : ndarray
            Cluster labels
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        scatter = ax.scatter(
            df['Chi tiêu (USD)'],
            df['Tần suất mua (lần/tháng)'],
            c=labels,
            cmap='viridis',
            s=120,
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )
        
        ax.set_title('Phân khúc khách hàng', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Chi tiêu trung bình (USD)', 
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Tần suất mua hàng (lần/tháng)', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Phân khúc', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
