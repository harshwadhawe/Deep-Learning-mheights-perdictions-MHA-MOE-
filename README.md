# Deep Learning for LP m-Height Prediction (MHA + MoE)

This repository contains the code and datasets for predicting the **log2(m-height)** of a given parity matrix $P$ and its associated parameters $(n, k, m)$. The model utilizes a **Transformer Encoder** backbone with a **Soft-Gated Mixture-of-Experts (MoE)** head to handle the highly non-linear and regime-dependent nature of the problem.

## 📂 Repository Structure

### **1. Core Notebook**
* **`Best_model_v2.ipynb`**: The main Jupyter Notebook containing the end-to-end pipeline.
    * **Architecture**: Defines the 2.46M parameter model using Multi-Head Attention (6 layers) and 7-expert MoE.
    * **Training**: Implements the training loop with `tf.distribute.MirroredStrategy` for A100 GPUs, including the "Hard Example Mining" and "Cooldown" phases.
    * **Inference**: Contains the evaluation script and the **Test-Time Augmentation (TTA)** logic for final predictions.

### **2. Data Generation**
* **`data_gen.py`**: The Python script used to generate the synthetic parity matrices and their corresponding ground-truth m-heights using Linear Programming (LP). This script was used to create the 102k supplemental samples used to boost the dataset size.

### **3. Data Directories**
* **`tf_dataset/`**: Contains the processed datasets in **TFRecord format** for high-performance training.
    * `train.tfrecord`, `val.tfrecord`, `test.tfrecord`: Standard splits.
    * `hard_booster_legal.tfrecord`: The mined "hard" examples used for the booster training phase.
    * `dataset_info.json`: Metadata including normalization statistics (mean/variance) and matrix dimensions.

* **`train_data/`**: The raw pickle files (`.pkl`) containing the training inputs (matrices $P$ and parameters $n, k, m$) and labels (m-heights). Includes both the instructor-provided data and custom-generated "Giant" matrices.

* **`test_data/`**: Contains the validation and test sets used for final scoring, such as `x_full_dataset_4.5k_samples.pkl`.

## 🚀 Key Techniques
* **Architecture**: Context-aware Transformer Encoders + Soft-Gated Mixture of Experts.
* **Hard Mining**: A secondary training phase focusing exclusively on the top 10% highest-error samples.
* **Cooldown**: A final fine-tuning phase with a low learning rate and a mixed dataset (90% normal / 10% hard) to stabilize the model.
* **Test-Time Augmentation (TTA)**: 

[Image of Test Time Augmentation]
 During inference, the input matrix is augmented 16 times (column permutations/sign flips) and predictions are averaged in log-space to reduce variance.

## 🛠️ Requirements
* Python 3.10+
* TensorFlow 2.x
* NumPy
* Pandas
* Matplotlib / Seaborn (for visualization)

## Usage
1.  **Generate Data**: Run `data_gen.py` if you need to create fresh samples.
2.  **Train**: Open `Best_model_v2.ipynb` and execute the training cells. Ensure `tf_dataset/` is populated first by running the conversion cells in the notebook.
3.  **Evaluate**: Use the inference section in the notebook to run TTA on `test_data/`.