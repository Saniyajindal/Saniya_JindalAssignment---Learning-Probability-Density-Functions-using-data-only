# 🚀 GAN-based Density Estimation of Air Quality Data
**Course:** UCS761: Deep Networks & Gradient Flow  
**Student:** Saniya Jindal  
**University:** Thapar Institute of Engineering and Technology  
**Roll Number:** 102303183  

---

## 📌 Project Objective
The goal of this assignment is to learn an unknown **Probability Density Function (PDF)** of a transformed random variable using a **Generative Adversarial Network (GAN)**. Instead of assuming a parametric form (like Gaussian), the GAN implicitly models the distribution directly from data samples.

## 🛠️ Mathematical Transformation
Based on my university roll number ($r = 102303183$), the transformation parameters were calculated as:
- **$a_r$**: $0.5 \times (r \pmod 7) = 1.5$
- **$b_r$**: $0.3 \times (r \pmod 5 + 1) = 1.2$

**Transformation Function:**
$$z = x + 1.5 \sin(1.2x)$$

## 🏗️ GAN Architecture
The model consists of two competing neural networks:

### 1. The Generator ($G$)
- **Input**: Latent noise vector (Size: 10) sampled from $N(0, 1)$.
- **Structure**: 3 Fully Connected layers with ReLU activation.
- **Goal**: Produce fake samples $z_f$ that mimic the distribution of $z$.

### 2. The Discriminator ($D$)
- **Input**: Single value (Real $z$ or Fake $z_f$).
- **Structure**: 3 Fully Connected layers with LeakyReLU activation and a Sigmoid output.
- **Goal**: Distinguish between real and generated samples.

## 📊 Results & Observations
![PDF Plot](pdf_plot.png)

### Key Findings:
* **Mode Coverage**: The GAN successfully identifies the primary peaks of the transformed $NO_2$ distribution, showing high fidelity in learning the underlying PDF.
* **Training Stability**: Using the **Adam optimizer** with a learning rate of $0.0002$ ensured a stable Nash equilibrium between the Generator and Discriminator.
* **Non-Parametric Learning**: The model successfully learned the distribution without any prior assumptions about the data's shape (Gaussian, Exponential, etc.).

## 🚀 How to Run
1. Ensure `india-air-quality-data.csv` is in the root directory.
2. Install dependencies:
   ```bash
   pip install torch pandas matplotlib seaborn scikit-learn
