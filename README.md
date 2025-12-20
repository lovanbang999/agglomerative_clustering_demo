# Agglomerative Clustering Demo
### Data Mining Course - Feature Extraction Topic

## 📁 Cấu trúc Project

```
agglomerative_clustering_demo/
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── src/                        # Source code
│   ├── __init__.py
│   ├── clustering_demo.py      # Demo functions
│   ├── visualizer.py           # Visualization utilities
│   └── utils.py                # Helper functions
├── data/                       # Data folder (for custom datasets)
├── outputs/                    # Output folder (for exports)
└── docs/                       # Documentation folder
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt

```bash
# Di chuyển vào thư mục project
cd agglomerative_clustering_demo

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: **http://localhost:8501**

## 📚 Nội dung Demo

### 1. Lý thuyết (📚)
- Giới thiệu Agglomerative Clustering
- Các phương pháp Linkage: Ward, Complete, Average, Single
- Tham số trong sklearn
- Ưu điểm và nhược điểm
- Code examples

### 2. Demo Cơ Bản (🎯)
- Chọn loại dataset: Blobs, Moons, Circles, Anisotropic
- Điều chỉnh số cụm và linkage method
- Xem kết quả clustering
- Dendrogram visualization
- Evaluation metrics

### 3. So Sánh Linkage (🔍)
- So sánh trực quan các linkage methods
- Metrics comparison
- Dendrogram của từng phương pháp

### 4. Phân Tích Tham Số (⚙️)
- **Số cụm**: Tìm số cụm tối ưu với metrics
- **Connectivity**: Ảnh hưởng của connectivity constraint

### 5. Ứng Dụng Thực Tế (🌍)
- Phân khúc khách hàng
- Phân tích từng segment
- Chiến lược marketing
- Export kết quả CSV

## 💻 Ví dụ Code

### Basic Usage
```python
from sklearn.cluster import AgglomerativeClustering

# Khởi tạo model
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

# Fit và predict
labels = model.fit_predict(X)
```

### With Connectivity
```python
from sklearn.neighbors import kneighbors_graph

# Tạo connectivity matrix
connectivity = kneighbors_graph(X, n_neighbors=10)

# Clustering với connectivity
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='average',
    connectivity=connectivity
)
labels = model.fit_predict(X)
```

## 📊 Metrics Đánh Giá

- **Silhouette Score**: Cao hơn là tốt hơn (từ -1 đến 1)
- **Davies-Bouldin Index**: Thấp hơn là tốt hơn (≥ 0)
- **Calinski-Harabasz Score**: Cao hơn là tốt hơn (≥ 0)

## 🎓 Sử Dụng Cho Thuyết Trình

### Gợi ý flow:
1. **Lý thuyết** (3-4 phút): Giới thiệu concept và các linkage methods
2. **Demo cơ bản** (4-5 phút): Chạy trực tiếp với các dataset khác nhau
3. **So sánh linkage** (2-3 phút): Cho thấy sự khác biệt giữa các phương pháp
4. **Tham số** (2-3 phút): Tìm số cụm tối ưu, connectivity
5. **Ứng dụng** (3-4 phút): Case study phân khúc khách hàng
6. **Q&A** (3-5 phút): Trả lời câu hỏi

### Tips:
- ✅ Test ứng dụng trước khi thuyết trình
- ✅ Chuẩn bị backup (screenshot) phòng lỗi
- ✅ Giải thích code khi demo
- ✅ Kết nối lý thuyết với thực hành

## 📖 Tài Liệu Tham Khảo

- [Sklearn AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [Hierarchical Clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [Scipy Dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html)

## ❓ Troubleshooting

### Port đã bị chiếm?
```bash
streamlit run app.py --server.port 8502
```

### Module not found?
```bash
pip install -r requirements.txt --upgrade
```

### Clear cache?
```bash
streamlit cache clear
```

## 📝 License

Code cho mục đích học tập - Data Mining Course

---
