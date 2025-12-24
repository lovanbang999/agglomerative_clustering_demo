"""
Utility functions for the application
"""

import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles

def load_custom_css():
    """Load custom CSS styling"""
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .sub-header {
            font-size: 1.8rem;
            color: #2ca02c;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #2ca02c;
            padding-bottom: 0.5rem;
        }
        
        .stButton>button {
            width: 100%;
            border-radius: 0.5rem;
            height: 3rem;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)


def generate_dataset(dataset_type, n_samples=300, random_state=42):
    """
    Generate different types of datasets for clustering
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset ('blobs', 'moons', 'circles', 'anisotropic')
    n_samples : int
        Number of samples to generate
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    X : ndarray of shape (n_samples, 2)
        Generated samples
    y : ndarray of shape (n_samples,)
        True labels (for reference)
    """
    
    if dataset_type == 'blobs':
        X, y = make_blobs(
            n_samples=n_samples,
            centers=4,
            n_features=2,
            random_state=random_state,
            cluster_std=0.6
        )
    
    elif dataset_type == 'moons':
        X, y = make_moons(
            n_samples=n_samples,
            noise=0.1,
            random_state=random_state
        )
    
    elif dataset_type == 'circles':
        X, y = make_circles(
            n_samples=n_samples,
            noise=0.05,
            factor=0.5,
            random_state=random_state
        )
    
    elif dataset_type == 'anisotropic':
        X, y = make_blobs(
            n_samples=n_samples,
            centers=3,
            n_features=2,
            random_state=random_state
        )
        # Apply transformation
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return X, y


def format_code_example(code_str):
    """Format code example with syntax highlighting"""
    return f"""
    ```python
    {code_str}
    ```
    """
