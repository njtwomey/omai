# Generative models (part 6, neural architectures)

Deep generative models trained and sampled: autoencoders and variational autoencoders, autoregressive models (PixelCNN-style on tiny images, sequence models), normalising flows (coupling layers, planar and radial flows), generative adversarial networks (the minimax game, mode collapse, Wasserstein losses), energy-based models (contrastive divergence, Langevin sampling), diffusion models (forward noising, the denoiser, samplers), and flow matching with fast sampling, as sessions over `nn`, `losses`, `dists`, `sampling`, `stochastic` and the trainer base, on 2-D data and tiny images.

## Capabilities

- Every model on 2-D toy densities (moons, rings, a mixture) with the learned density or samples as a probe at the viewed step, and the loss on the tape; the model's own slots (architecture, objective variant, optimiser, schedule).
- VAE: the latent space and reconstructions, the KL weight (β) as a knob, posterior collapse.
- Flows: the transformation drawn layer by layer, the log-determinant on the tape.
- GAN: the generator and discriminator losses, the discriminator's surface over the plane, mode collapse and the fix.
- Diffusion: the forward process on the data, the denoiser's vector field at each noise level, the reverse sampler step by step (DDPM, DDIM), the SDE and ODE views (with `probability`).
- Flow matching: the straight-line paths and the learned velocity field.

## Examples to make

- A VAE on two moons: samples and the latent space, β as the knob.
- A flow's layers transforming a Gaussian into rings, one layer per step.
- A GAN's discriminator surface and the generator's samples, oscillating then settling.
- Diffusion on 2-D: the reverse process from noise to data, with the score field drawn at the viewed noise level.
- Flow matching against diffusion on the same data: the sampling paths side by side.

## Rendering

- Samples in the plane with the true density (`decisionSurface` as a density), vector fields (`fields`), image grids for tiny images, latent-space embeddings, loss curves.
- Missing: a vector-field-at-step panel; the comparison panel.

## Shared

- `shared/`: the toy density slot with truth (shared with `probability` and `unsupervised-learning`), the sample probes; the trainer base.

## Problems

- `generative/` moves here. The diffusion sampler as a session of its own (not training) is the cheap, high-value page.
