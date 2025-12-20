"""
Main clustering demo module
Contains all demonstration functions
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

from src.utils import generate_dataset
from src.visualizer import Visualizer


class ClusteringDemo:
    """Main demo class for Agglomerative Clustering"""
    
    def __init__(self):
        self.visualizer = Visualizer()
    
    def show_theory(self):
        """Display theoretical content"""
        st.markdown("## 📚 Lý thuyết Agglomerative Clustering")
        
        # Introduction
        st.markdown("""
        <div class="info-box">
        <h3>1️⃣ Agglomerative Clustering là gì?</h3>
        <p><b>Agglomerative Clustering</b> (phân cụm kết tụ) là thuật toán phân cụm phân cấp 
        (hierarchical clustering) theo hướng <b>bottom-up</b>:</p>
        <ul>
            <li>🔹 Bắt đầu: Mỗi điểm dữ liệu là một cụm riêng</li>
            <li>🔹 Lặp lại: Gộp hai cụm gần nhất lại với nhau</li>
            <li>🔹 Kết thúc: Khi đạt số cụm mong muốn hoặc tất cả thành một cụm</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Linkage methods
        st.markdown("""
        <div class="info-box">
        <h3>2️⃣ Các phương pháp Linkage</h3>
        <p>Cách đo khoảng cách giữa các cụm:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔸 Ward Linkage**
            - Tối thiểu hóa phương sai trong cụm
            - Tạo cụm compact, cân bằng
            - Chỉ dùng với Euclidean distance
            - ✅ Thường cho kết quả tốt nhất
            
            **🔸 Complete Linkage**
            - Khoảng cách max giữa các điểm
            - Tạo cụm compact
            - Nhạy cảm với outliers
            """)
        
        with col2:
            st.markdown("""
            **🔸 Average Linkage**
            - Trung bình khoảng cách các cặp
            - Cân bằng giữa single và complete
            - Ít nhạy cảm với noise
            
            **🔸 Single Linkage**
            - Khoảng cách min giữa các điểm
            - Dễ bị "chain effect"
            - Tốt cho cụm non-convex
            """)
        
        # Parameters
        st.markdown("""
        <div class="info-box">
        <h3>3️⃣ Tham số chính trong sklearn</h3>
        </div>
        """, unsafe_allow_html=True)
        
        params_df = pd.DataFrame({
            'Tham số': [
                'n_clusters',
                'linkage',
                'metric',
                'distance_threshold',
                'connectivity'
            ],
            'Mô tả': [
                'Số cụm cần tìm',
                'Phương pháp liên kết (ward, complete, average, single)',
                'Độ đo khoảng cách (euclidean, manhattan, cosine...)',
                'Ngưỡng cắt dendrogram (nếu dùng thì n_clusters=None)',
                'Ma trận kết nối xác định láng giềng'
            ],
            'Mặc định': ['2', 'ward', 'euclidean', 'None', 'None']
        })
        
        st.dataframe(params_df, use_container_width=True)
        
        # Code example
        st.markdown("### 💻 Code cơ bản")
        
        st.code("""
from sklearn.cluster import AgglomerativeClustering

# Khởi tạo model
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward',
    metric='euclidean'
)

# Fit và predict
labels = model.fit_predict(X)

# Thông tin model
print(f"Số cụm: {model.n_clusters_}")
print(f"Số lá: {model.n_leaves_}")
        """, language='python')
        
        # Pros and cons
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Ưu điểm</h4>
            <ul>
                <li>Không cần chỉ định số cụm trước</li>
                <li>Tạo cấu trúc phân cấp (dendrogram)</li>
                <li>Phát hiện cụm hình dạng phức tạp</li>
                <li>Kết quả deterministic</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Nhược điểm</h4>
            <ul>
                <li>Độ phức tạp cao: O(n³) thời gian</li>
                <li>Không phù hợp dữ liệu lớn</li>
                <li>Quyết định gộp không thể hoàn tác</li>
                <li>Nhạy cảm với noise và outliers</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def show_basic_demo(self):
        """Basic demonstration"""
        st.markdown("## 🎯 Demo Cơ Bản")
        
        st.info("💡 Chạy Agglomerative Clustering với các tham số khác nhau")
        
        # Parameters
        st.markdown("### ⚙️ Cấu hình")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dataset_type = st.selectbox(
                "Loại dữ liệu:",
                ['blobs', 'moons', 'circles', 'anisotropic'],
                format_func=lambda x: {
                    'blobs': '⚪ Blobs (Cụm tròn)',
                    'moons': '🌙 Moons (Bán nguyệt)',
                    'circles': '⭕ Circles (Vòng tròn)',
                    'anisotropic': '📐 Anisotropic'
                }[x]
            )
        
        with col2:
            n_clusters = st.slider("Số cụm:", 2, 6, 3)
        
        with col3:
            linkage = st.selectbox(
                "Linkage:",
                ['ward', 'complete', 'average', 'single']
            )
        
        n_samples = st.slider("Số mẫu:", 100, 500, 300, 50)
        
        # Show code
        st.markdown("### 💻 Code")
        st.code(f"""
from sklearn.cluster import AgglomerativeClustering

# Khởi tạo model
model = AgglomerativeClustering(
    n_clusters={n_clusters},
    linkage='{linkage}'
)

# Fit và predict
labels = model.fit_predict(X)
        """, language='python')
        
        # Run button
        if st.button("🚀 Chạy Clustering", type="primary"):
            with st.spinner("Đang phân cụm..."):
                # Generate data
                X, y_true = generate_dataset(dataset_type, n_samples)
                
                # Clustering
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage
                )
                labels = model.fit_predict(X)
                
                # Metrics
                st.markdown("### 📊 Kết quả")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Số cụm", model.n_clusters_)
                
                with col2:
                    silhouette = silhouette_score(X, labels)
                    st.metric("Silhouette", f"{silhouette:.3f}")
                
                with col3:
                    davies_bouldin = davies_bouldin_score(X, labels)
                    st.metric("Davies-Bouldin", f"{davies_bouldin:.3f}")
                
                with col4:
                    calinski = calinski_harabasz_score(X, labels)
                    st.metric("Calinski-Harabasz", f"{calinski:.1f}")
                
                # Visualizations
                st.markdown("### 📈 Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = self.visualizer.plot_clustering_result(
                        X, labels,
                        f"Clustering Result ({linkage})"
                    )
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    fig = self.visualizer.plot_dendrogram(
                        X, linkage,
                        f"Dendrogram ({linkage})"
                    )
                    st.pyplot(fig)
                    plt.close()
                
                # Distribution
                st.markdown("### 📊 Phân bố cụm")
                cluster_dist = pd.Series(labels).value_counts().sort_index()
                st.bar_chart(cluster_dist)
    
    def show_linkage_comparison(self):
        """Compare different linkage methods"""
        st.markdown("## 🔍 So Sánh Linkage Methods")
        
        st.info("💡 So sánh các phương pháp linkage trên cùng dataset")
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_type = st.selectbox(
                "Dataset:",
                ['blobs', 'moons', 'circles', 'anisotropic'],
                key='comp_dataset'
            )
            n_samples = st.slider("Số mẫu:", 100, 500, 300, key='comp_samples')
        
        with col2:
            n_clusters = st.slider("Số cụm:", 2, 5, 3, key='comp_clusters')
            linkage_methods = st.multiselect(
                "Chọn linkage methods:",
                ['ward', 'complete', 'average', 'single'],
                default=['ward', 'complete', 'average']
            )
        
        if st.button("🔄 So sánh", type="primary"):
            if not linkage_methods:
                st.error("❌ Vui lòng chọn ít nhất 1 phương pháp!")
                return
            
            with st.spinner("Đang so sánh..."):
                # Generate data
                X, _ = generate_dataset(dataset_type, n_samples)
                
                # Cluster with each method
                results = {}
                metrics_data = []
                
                for method in linkage_methods:
                    model = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage=method
                    )
                    labels = model.fit_predict(X)
                    results[method] = labels
                    
                    # Calculate metrics
                    metrics_data.append({
                        'Linkage': method.upper(),
                        'Silhouette': silhouette_score(X, labels),
                        'Davies-Bouldin': davies_bouldin_score(X, labels),
                        'Calinski-Harabasz': calinski_harabasz_score(X, labels)
                    })
                
                # Visualize
                st.markdown("### 📈 Visualization")
                fig = self.visualizer.plot_linkage_comparison(X, results, n_clusters)
                st.pyplot(fig)
                plt.close()
                
                # Metrics comparison
                st.markdown("### 📊 So sánh Metrics")
                metrics_df = pd.DataFrame(metrics_data)
                
                # Format metrics
                metrics_df['Silhouette'] = metrics_df['Silhouette'].apply(lambda x: f"{x:.3f}")
                metrics_df['Davies-Bouldin'] = metrics_df['Davies-Bouldin'].apply(lambda x: f"{x:.3f}")
                metrics_df['Calinski-Harabasz'] = metrics_df['Calinski-Harabasz'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(metrics_df, use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                <h4>📌 Cách đọc metrics:</h4>
                <ul>
                    <li><b>Silhouette Score</b>: Càng cao càng tốt (từ -1 đến 1)</li>
                    <li><b>Davies-Bouldin</b>: Càng thấp càng tốt (≥ 0)</li>
                    <li><b>Calinski-Harabasz</b>: Càng cao càng tốt (≥ 0)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    def show_parameter_analysis(self):
        """Analyze effect of parameters"""
        st.markdown("## ⚙️ Phân Tích Tham Số")
        
        tab1, tab2 = st.tabs(["📊 Số cụm", "🔗 Connectivity"])
        
        with tab1:
            self._analyze_n_clusters()
        
        with tab2:
            self._analyze_connectivity()
    
    def _analyze_n_clusters(self):
        """Analyze optimal number of clusters"""
        st.markdown("### 📊 Tìm số cụm tối ưu")
        
        st.info("💡 Sử dụng metrics để tìm số cụm phù hợp")
        
        if st.button("📈 Phân tích", type="primary", key='analyze_k'):
            with st.spinner("Đang phân tích..."):
                # Generate data
                X, _ = generate_dataset('blobs', 300)
                
                # Test range
                cluster_range = range(2, 9)
                metrics_dict = {
                    'silhouette': [],
                    'davies_bouldin': [],
                    'calinski_harabasz': []
                }
                
                # Progress bar
                progress_bar = st.progress(0)
                
                for idx, k in enumerate(cluster_range):
                    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                    labels = model.fit_predict(X)
                    
                    metrics_dict['silhouette'].append(silhouette_score(X, labels))
                    metrics_dict['davies_bouldin'].append(davies_bouldin_score(X, labels))
                    metrics_dict['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
                    
                    progress_bar.progress((idx + 1) / len(cluster_range))
                
                # Plot
                fig = self.visualizer.plot_metrics_comparison(cluster_range, metrics_dict)
                st.pyplot(fig)
                plt.close()
                
                # Table
                st.markdown("### 📋 Bảng kết quả")
                results_df = pd.DataFrame({
                    'k': list(cluster_range),
                    'Silhouette': [f"{x:.3f}" for x in metrics_dict['silhouette']],
                    'Davies-Bouldin': [f"{x:.3f}" for x in metrics_dict['davies_bouldin']],
                    'Calinski-Harabasz': [f"{x:.1f}" for x in metrics_dict['calinski_harabasz']]
                })
                st.dataframe(results_df, use_container_width=True)
    
    def _analyze_connectivity(self):
        """Analyze connectivity constraint effect"""
        st.markdown("### 🔗 Ảnh hưởng của Connectivity")
        
        st.markdown("""
        <div class="info-box">
        <p><b>Connectivity matrix</b> xác định các điểm nào có thể được gộp với nhau dựa trên cấu trúc không gian.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_neighbors = st.slider("Số láng giềng (k):", 2, 20, 10)
        with col2:
            n_clusters = st.slider("Số cụm:", 2, 5, 2, key='conn_k')
        
        if st.button("🔗 So sánh", type="primary", key='compare_conn'):
            with st.spinner("Đang so sánh..."):
                # Generate moon data
                X, _ = generate_dataset('moons', 300)
                
                # Create connectivity
                connectivity = kneighbors_graph(
                    X, n_neighbors=n_neighbors, include_self=False
                )
                
                # Compare
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Without connectivity
                model1 = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
                labels1 = model1.fit_predict(X)
                
                scatter1 = axes[0].scatter(
                    X[:, 0], X[:, 1], c=labels1, cmap='viridis',
                    s=60, alpha=0.7, edgecolors='black', linewidth=0.8
                )
                axes[0].set_title('Không có Connectivity', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Feature 1', fontsize=12)
                axes[0].set_ylabel('Feature 2', fontsize=12)
                axes[0].grid(True, alpha=0.3)
                plt.colorbar(scatter1, ax=axes[0])
                
                # With connectivity
                model2 = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='average',
                    connectivity=connectivity
                )
                labels2 = model2.fit_predict(X)
                
                scatter2 = axes[1].scatter(
                    X[:, 0], X[:, 1], c=labels2, cmap='viridis',
                    s=60, alpha=0.7, edgecolors='black', linewidth=0.8
                )
                axes[1].set_title(f'Với Connectivity (k={n_neighbors})', 
                                fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Feature 1', fontsize=12)
                axes[1].set_ylabel('Feature 2', fontsize=12)
                axes[1].grid(True, alpha=0.3)
                plt.colorbar(scatter2, ax=axes[1])
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.success("✅ Connectivity giúp thuật toán tôn trọng cấu trúc không gian!")
    
    def show_real_world_application(self):
        """Show real-world application"""
        st.markdown("## 🌍 Ứng Dụng Thực Tế: Phân Khúc Khách Hàng")
        
        st.info("💡 Ví dụ: Phân nhóm khách hàng theo hành vi mua sắm")
        
        # Generate customer data
        st.markdown("### 📋 Dữ liệu khách hàng")
        
        np.random.seed(42)
        
        # Create segments
        seg1 = np.random.randn(60, 2) * [10, 5] + [30, 80]   # Low spend, high freq
        seg2 = np.random.randn(70, 2) * [8, 8] + [60, 40]    # Medium
        seg3 = np.random.randn(70, 2) * [12, 6] + [90, 20]   # High spend, low freq
        
        X = np.vstack([seg1, seg2, seg3])
        
        df = pd.DataFrame(X, columns=['Chi tiêu (USD)', 'Tần suất mua (lần/tháng)'])
        df['Khách hàng ID'] = [f'KH{i:04d}' for i in range(200)]
        
        st.dataframe(df.head(10), use_container_width=True)
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_segments = st.slider("Số phân khúc:", 2, 5, 3)
        with col2:
            linkage = st.selectbox("Linkage:", ['ward', 'complete', 'average'], key='app_linkage')
        
        if st.button("🎯 Phân khúc", type="primary"):
            with st.spinner("Đang phân tích..."):
                # Clustering
                model = AgglomerativeClustering(n_clusters=n_segments, linkage=linkage)
                labels = model.fit_predict(X)
                
                df['Phân khúc'] = labels
                
                # Results
                st.markdown("### 📊 Kết quả")
                
                # Visualization
                fig = self.visualizer.plot_customer_segmentation(df, labels)
                st.pyplot(fig)
                plt.close()
                
                # Segment analysis
                st.markdown("### 📈 Phân tích từng phân khúc")
                
                for seg in range(n_segments):
                    seg_data = df[df['Phân khúc'] == seg]
                    avg_spend = seg_data['Chi tiêu (USD)'].mean()
                    avg_freq = seg_data['Tần suất mua (lần/tháng)'].mean()
                    count = len(seg_data)
                    
                    with st.expander(f"🏷️ Phân khúc {seg} ({count} khách hàng)", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Số KH", count)
                        with col2:
                            st.metric("Chi tiêu TB", f"${avg_spend:.1f}")
                        with col3:
                            st.metric("Tần suất TB", f"{avg_freq:.1f}")
                        
                        # Recommendation
                        if avg_spend > 70 and avg_freq > 60:
                            st.success("🌟 **VIP**: Ưu đãi đặc biệt, chương trình loyalty")
                        elif avg_spend > 50:
                            st.info("💰 **High Value**: Cross-selling, upselling")
                        else:
                            st.warning("📢 **Potential**: Khuyến mãi, engagement campaigns")
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Tải kết quả (CSV)",
                    csv,
                    "customer_segments.csv",
                    "text/csv"
                )
