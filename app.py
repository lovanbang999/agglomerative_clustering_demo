"""
Agglomerative Clustering Demo
Data Mining Course - Feature Extraction Topic

Main application for demonstrating AgglomerativeClustering from scikit-learn
"""

import streamlit as st
from src.clustering_demo import ClusteringDemo
from src.visualizer import Visualizer
from src.utils import load_custom_css

# Page configuration
st.set_page_config(
    page_title="Agglomerative Clustering Demo",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

def main():
    """Main application"""
    
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #1f77b4; margin-bottom: 0;'>🌳 Agglomerative Clustering</h1>
            <p style='font-size: 1.2rem; color: #666;'>Phân cụm phân cấp với scikit-learn</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Chọn phần demo:",
        [
            "📚 Lý thuyết",
            "🎯 Demo cơ bản", 
            "🔍 So sánh Linkage",
            "⚙️ Tham số",
            "🌍 Ứng dụng thực tế"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Hướng dẫn:**
    - Chọn phần demo ở trên
    - Điều chỉnh tham số
    - Nhấn nút để xem kết quả
    """)
    
    # Initialize demo
    demo = ClusteringDemo()
    
    # Route to appropriate page
    if page == "📚 Lý thuyết":
        demo.show_theory()
    elif page == "🎯 Demo cơ bản":
        demo.show_basic_demo()
    elif page == "🔍 So sánh Linkage":
        demo.show_linkage_comparison()
    elif page == "⚙️ Tham số":
        demo.show_parameter_analysis()
    elif page == "🌍 Ứng dụng thực tế":
        demo.show_real_world_application()

if __name__ == "__main__":
    main()
