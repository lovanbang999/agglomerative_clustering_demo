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
        st.markdown("### 1️⃣ Agglomerative Clustering là gì?")
        st.info("""
**Agglomerative Clustering** (phân cụm kết tụ) là thuật toán phân cụm phân cấp (hierarchical clustering) theo hướng **bottom-up**:

- 🔹 **Bắt đầu**: Mỗi điểm dữ liệu là một cụm riêng
- 🔹 **Lặp lại**: Gộp hai cụm gần nhất lại với nhau
- 🔹 **Kết thúc**: Khi đạt số cụm mong muốn hoặc tất cả thành một cụm
        """)
        
        # Linkage methods
        st.markdown("### 2️⃣ Các phương pháp Linkage")
        st.info("Cách đo khoảng cách giữa các cụm:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
**🔸 Ward Linkage**
- Tối thiểu hóa phương sai trong cụm
- Tạo cụm compact, cân bằng
- Chỉ dùng với Euclidean distance
- Thường cho kết quả tốt nhất

**🔸 Complete Linkage**
- Khoảng cách **lớn nhất** giữa các điểm thuộc 2 cụm
- Chỉ gộp khi tất cả các điểm đều tương đối gần nhau
- Tạo cụm chặt, đồng đều kích thước
- Nhạy cảm với outliers
            """)
        
        with col2:
            st.markdown("""
**🔸 Average Linkage**
- Lấy **trung bình** cộng tất cả khoảng cách giữa các điểm
- Cân bằng giữa Single và Complete
- Ít nhạy cảm với noise hơn

**🔸 Single Linkage**
- Khoảng cách **nhỏ nhất** giữa các điểm thuộc 2 cụm
- Chỉ cần một cặp điểm gần là hai cụm được gộp
- Tạo cụm dài, dễ bị "chain effect" (ảnh hưởng bởi nhiễu)
- Tốt cho cụm hình dạng phức tạp, không lồi (non-convex)
            """)
        
        # Parameters
        st.markdown("### 3️⃣ Tham số chính trong sklearn")
        
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
        st.markdown("### 4️⃣ Ưu điểm và Nhược điểm")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
**✅ Ưu điểm:**
- Không cần chỉ định số cụm trước
- Tạo cấu trúc phân cấp (dendrogram)
- Phát hiện cụm hình dạng phức tạp
- Kết quả deterministic (không ngẫu nhiên)
            """)
        
        with col2:
            st.warning("""
**⚠️ Nhược điểm:**
- Độ phức tạp cao: O(n³) thời gian
- Không phù hợp với dữ liệu lớn
- Quyết định gộp không thể hoàn tác
- Nhạy cảm với noise và outliers
            """)
    
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
                ['ward', 'average', 'complete', 'single']
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
                print('===> Label: ', labels)
                
                # Metrics
                st.markdown("### 📊 Kết quả")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Số cụm", model.n_clusters_)
                
                with col2:
                    silhouette = silhouette_score(X, labels)
                    st.metric("Silhouette (Độ tách biệt)", f"{silhouette:.3f}")
                
                with col3:
                    davies_bouldin = davies_bouldin_score(X, labels)
                    st.metric("Davies-Bouldin (Độ chồng lấn)", f"{davies_bouldin:.3f}")
                
                with col4:
                    calinski = calinski_harabasz_score(X, labels)
                    st.metric("Calinski-Harabasz (giữa-cluster / trong-cluster)", f"{calinski:.1f}")
                
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
                ['ward', 'average', 'complete', 'single'],
                default=['ward', 'average', 'complete']
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
                
                st.info("""
**📌 Cách đọc metrics:**

- **Silhouette Score**: Càng cao càng tốt (từ -1 đến 1)
- **Davies-Bouldin**: Càng thấp càng tốt (≥ 0)
- **Calinski-Harabasz**: Càng cao càng tốt (≥ 0)
                """)
    
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
        
        st.dataframe(df.head(200), use_container_width=True)
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_segments = st.slider("Số phân khúc:", 2, 5, 3)
        with col2:
            linkage = st.selectbox("Linkage:", ['ward', 'complete', 'average', 'single'], key='app_linkage')
        
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
