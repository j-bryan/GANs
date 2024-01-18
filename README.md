This project implements a number of generative adversarial networks (GAN) models.
While the target usage is scenario generation for variable renewable energy (VRE) resources (e.g. wind and solar power), regional electrical load, electricity price, and other time series quantities relevant to studying energy systems and electrical grids, these models could easily be adapted to other applications.

All of this work is part of my dissertation research about integrating energy storage capacity with nuclear power plants, how these systems will interact with evolving electricity markets, and how they can be an economical low-carbon source of electricity that is complementary to wind and solar power.

At present, models are implemented using the following architectures:
- feedforward neural networks (FNN)
- convolutional neural networks (CNN)
- [WIP] doppelGANger (DGAN), which use long short-term memory (LSTM) models and a meta-data modeling approach (see [Lin et al., 2020](https://arxiv.org/abs/1909.13403))
- [WIP] stochastic differential equation GANs (SDE-GAN) (see [Kidger et al., 2021](https://arxiv.org/abs/2102.03657))

A number of other helpful tools and features have been implemented.
- Data loaders for time series data
- Training classes for WGAN, WGAN with weight clipping, and WGAN with gradient penalty
- Plotting and evaluation functions for comparing generated time series with the true data
- [WIP] Preprocessing pipelines compatible with torch.Tensor objects, inspired by scikit-learn's Pipeline class
