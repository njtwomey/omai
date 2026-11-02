import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Implicit neural representation',
  summary:
    'A small network fitted to the colour of every pixel of a picture, as a function of the pixel coordinates; compare encodings and sine activations against a plain ReLU network.',
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
