import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Implicit neural representation',
  summary:
    'A small network fitted to the colour of every pixel of a 32 by 32 picture, as a function of the pixel coordinates. A plain ReLU network on raw coordinates stays blurry on edges and texture (spectral bias); encode the coordinates or switch to sine activations to sharpen.',
  tags: [
    'neural field',
    'implicit',
    'inr',
    'coordinate network',
    'siren',
    'fourier features',
    'hash grid',
    'spectral bias',
  ],
  added: '2026-11-02',
}
