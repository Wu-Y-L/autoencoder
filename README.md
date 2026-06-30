# Variational Autoencoder

- This VAE architecture has U-net like skip connections between encoder and decoder for maintaining high frequency information. Self-attention ( MultiheadAttention) increase receptive field. Uses GroupNorm, and SiLU/GELU.
- Targeted for microscopy, but should work for anything. Maybe for something else add image quality metrics in, but I personally don't like them for microscopy. Training generative adversarial style. Loss has reconstruction loss ( l1 ), KL divergence, and discriminator loss.

