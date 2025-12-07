# Deep Learning for LP m-Height Prediction (MHA + MoE)

This repository provides a full deep-learning pipeline for predicting the **log₂(m-height)** of a parity-check matrix (P) given parameters ((n, k, m)). The model combines a **Transformer Encoder** backbone with a **Soft-Gated Mixture-of-Experts (MoE)** output layer to capture the highly non-linear structure of Linear Programming (LP) m-height estimation.

---

## 📂 Repository Structure

### **1. Core Notebook**

**`Best_model_v2.ipynb`** — the main end-to-end workflow.

**Highlights:**

* **Model Architecture**
  A 2.46M-parameter network with **6 Transformer layers** and a **7-Expert MoE head**.

* **Training Pipeline**
  Trains with `tf.distribute.MirroredStrategy` on A100 GPUs and includes:

  * **Hard Example Mining** — focuses on the top 10% error samples.
  * **Cooldown Phase** — stabilizes the model with a reduced learning rate and a mixed dataset of normal + hard samples.

* **Inference + Test-Time Augmentation (TTA)**
  Implements the full evaluation script with 16-way augmentation for robust predictions.

---

### **2. Data Generation**

**`data_gen.py`**
Creates synthetic parity matrices and corresponding LP-computed m-heights.
Used to generate >100k supplemental training samples (45k, 54k, 90k, 108k). All synthetic datasets were produced on the TAMU HPRC using 64-core CPU nodes for fast LP solving.

---

## **3. Data Directories**

### **`tf_dataset/` — TFRecord Training Data**

A high-performance dataset used by the TensorFlow training pipeline.

Includes:

* `train.tfrecord`, `val.tfrecord`, `test.tfrecord`
* `hard_booster_legal.tfrecord` — the mined hard subset for booster training
* `dataset_info.json` — mean/variance statistics, matrix dimensions ((p_{\text{rows}} , p_{\text{cols}} , m_{\max}))

---

### **Why `.tfrecord`?**

`.tfrecord` format is preferred for training because:

1. **High-throughput I/O** optimized for GPU pipelines.
2. **Streaming efficiency**, avoiding full `.pkl` loads into memory.
3. **Seamless integration with `tf.data`**, enabling parallel decoding, prefetching, and fast batching.

---

### **`train_data/` — Raw Training Sources**

These are the original pickle files before TFRecord conversion:

```python
DATA_SOURCES = [
    ("train_data/input_data_56k.pkl",  "train_data/output_data_56k.pkl",  "Instructor_56k", True),
    ("train_data/X_generated_45k.pkl", "train_data/y_generated_45k.pkl",  "Generated_45k",  False),
    ("train_data/X_generated_54k.pkl", "train_data/y_generated_54k.pkl",  "Generated_54k",  False),
    ("train_data/X_generated_90k.pkl", "train_data/y_generated_90k.pkl",  "Generated_90k",  False),
    ("train_data/X_generated_108k.pkl","train_data/y_generated_108k.pkl", "Generated_108k", False),
]
```

* The **56k dataset** was provided by the instructor.
* All other datasets were **generated on TAMU HPRC** via `data_gen.py` using 64-core CPU nodes.

These files are merged and transformed into TFRecords for efficient training.

---

### **`test_data/` — Evaluation Data**

Used exclusively for inference and leaderboard scoring:

```
INPUT_PATH  = "test_data/x_full_dataset_4.5k_samples.pkl"
OUTPUT_PATH = "test_data/y_full_dataset_4.5k_samples.pkl"
```

---

## 🚀 Key Techniques & Innovations

### **1. Transformer + MoE Architecture**

A multi-head attention encoder extracts structural features from the parity matrix, while the Soft-Gated MoE head models distinct behavior across different ((n, k, m)) regimes.

---

### **2. Hard Example Mining**

After the initial training cycle:

* The **top 10% worst-performing samples** are identified.
* A booster phase trains the model exclusively on these hard cases.
* Result: significantly improved performance on tail and high-complexity matrices.

---

### **3. Cooldown Fine-Tuning**

A stabilization stage with:

* **90% regular samples**
* **10% hard samples**
* A reduced learning rate

This reduces prediction variance and prevents overfitting introduced during booster training.

---

### **4. Test-Time Augmentation (TTA)**

During inference:

* Each matrix is augmented **16 times** using valid column permutations and sign flips.
* Predictions are averaged **in log-space**, improving consistency and eliminating variance from augmentation randomness.

---

## 🛠️ Requirements

* Python 3.10+
* TensorFlow 2.x
* NumPy
* Pandas
* Matplotlib / Seaborn
* CUDA 12 (for GPU acceleration)

---

## ▶️ Usage Guide

### **1. (Optional) Generate Synthetic Data**

```bash
python data_gen.py
```

Generates new parity matrices and computes their LP m-heights.

---

### **2. Train the Model**

Open `Best_model_v2.ipynb` and run:

* **Dataset Conversion Cells** → builds `.tfrecord` files from raw pickle datasets
* **Training Cells** → runs full training, hard mining, and cooldown

---

### **3. Evaluate the Model (with TTA)**

In the notebook:

* Load the 4.5k test set from `test_data/`
* Run the inference section to compute TTA-averaged predictions


