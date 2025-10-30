# 🌿 Bell Pepper Disease Detection using CNN  

> **Published Research Project:**  
> “CNN-Based Approach for Efficient Bell Pepper Leaf Disease Recognition”  
> *Author:* Yalla Venkata Suresh | *Year:* 2024  

---

## 🧠 Overview  

This research introduces a **Convolutional Neural Network (CNN)** architecture for the **automated detection of bacterial spot disease** in *Capsicum annuum (Bell Pepper)* leaves.  
The model demonstrates how deep learning can revolutionize **agricultural disease diagnosis**, achieving exceptional accuracy and supporting early disease intervention for farmers.  

---

## 🎯 Research Objectives  

- Develop a CNN model to classify **healthy vs diseased bell pepper leaves**.  
- Minimize training epochs while retaining maximum accuracy.  
- Provide a foundation for **AI-driven agricultural automation** systems.  
- Support precision agriculture through early disease identification and yield protection.  

---

## 🧾 Abstract  

Plant diseases drastically reduce agricultural productivity and profit margins.  
This study presents a **deep-learning-based classification model** using CNNs to recognize bacterial spot disease in bell pepper leaves.  
The model achieved **99.49 % accuracy** on a curated dataset of 2,475 images, outperforming prior studies while reducing computational complexity.  
The system efficiently classifies unseen images within seconds, demonstrating real-world applicability for mobile or IoT-based crop-health monitoring.  

---

## 🧬 Dataset  

- **Dataset Name:** Pepperbell  
- **Total Images:** 2,475  
- **Classes:** `Pepper_bell_healthy` | `Pepper_bell_diseased`  
- **Split:** 80 % Training | 10 % Validation | 10 % Testing  
- **Source:** Kaggle and custom field collection  
- **Download:** [📦 Pepperbell Dataset (Google Drive)](https://drive.google.com/drive/folders/1G3TX8nkK1ndUL2hyKcMlFl39BtHqOQzG?usp=sharing)

---

## 🧪 Methodology  

1. **Data Preprocessing & Augmentation**  
   - Image resizing and rescaling  
   - Rotation, flipping, and zoom augmentation  
   - Normalization to enhance generalization  

2. **Model Architecture (CNN)**  
   - **Convolution Layers:** Feature extraction  
   - **ReLU Activation:** Non-linearity introduction  
   - **Max Pooling:** Dimensionality reduction  
   - **Flatten + Fully Connected Layers:** Classification  
   - **Softmax Output:** Binary classification  

3. **Training Setup**  
   - **Frameworks:** TensorFlow | Keras  
   - **Optimizer:** Adam  
   - **Loss Function:** Categorical Cross-Entropy  
   - **Epochs:** 15 (Optimal convergence achieved early)  

---

## 📊 Results & Performance  

| Metric | Value |
|--------|--------|
| **Accuracy** | **99.49 %** |
| **Loss** | 0.015 |
| **Validation Accuracy** | 99.4 % |
| **Training Time** | ~45 sec / epoch |
| **Inference Time** | < 2 sec per image |

#### 🔹 Observations  
- Validation and training curves converge smoothly → minimal overfitting.  
- Robust performance with reduced epochs and lightweight architecture.  
- Effective for **edge deployment** in agricultural automation systems.  

---

## 📈 Visualization  

*(Include these plots in your notebook or repo images folder)*  

- **Training vs Validation Accuracy**  
- **Training vs Validation Loss**  
- **Sample Predictions** (Healthy | Diseased)  

---

## 📚 Literature Background  

Previous approaches in plant disease detection achieved 90–97 % accuracy using shallow CNN architectures.  
This work surpasses earlier benchmarks by combining optimized data augmentation, layer reduction, and accelerated training.  
It builds upon the works of Rehan Mahmood et al. (2020), Bagde et al. (2015), and Picon et al. (2019).  

---

## 🔬 Conclusion  

The proposed CNN model efficiently identifies bell pepper leaf diseases with 99.49 % accuracy, outperforming existing methods in speed and precision.  
Its implementation can significantly assist farmers in early disease detection, protecting crop yield and ensuring sustainable agriculture.  

---

## 🧩 Repository Structure  

