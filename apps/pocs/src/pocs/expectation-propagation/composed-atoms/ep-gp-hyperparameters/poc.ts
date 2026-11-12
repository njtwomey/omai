import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Gaussian process with a learned length scale, variance and noise',
  summary:
    'A dense Gaussian prior on eight function values whose kernel has a random length scale and variance and whose observations have a random noise sd (three scalar latents read by custom functions): EP runs inside a grid over the three, walking the region of the lattice where the posterior has mass, each node weighted by its EP evidence and the prior.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composed',
    'dense prior (gaussian-mv) at each grid node',
    'gaussian row (exact)',
    'the EP evidence as the weight of a node (ep-grid, a walk over three axes)',
  ],
  added: '2026-11-12',
}
