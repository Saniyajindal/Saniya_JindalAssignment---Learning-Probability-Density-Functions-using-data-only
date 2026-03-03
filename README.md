# Saniya_JindalAssignment---Learning-Probability-Density-Functions-using-data-only
# Learning PDF using GANs
**Name:** Saniya Jindal  
**Roll No:** 102303183  

## Objective
To learn the unknown probability density function (PDF) of a transformed random variable (NO2 concentration) using a Generative Adversarial Network (GAN).

## Parameters & Transformation
- **a_r:** 1.5
- **b_r:** 1.2
- **Function:** $z = x + 1.5 \sin(1.2x)$

## Files
- `gan_assignment.py`: Main implementation script.
- `pdf_plot.png`: Visual comparison of real vs generated distribution.

## Observations
- **Mode Coverage:** The GAN successfully captures the distribution peaks.
- **Training:** Adam optimizer provided stable results over 1000 epochs.
