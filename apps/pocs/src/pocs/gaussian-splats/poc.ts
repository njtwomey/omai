import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: '3-D Gaussian splatting',
  summary:
    'A cloud of 3-D Gaussians fitted to a few views of a toy scene and drawn by projecting them to the screen. Play the training, watch the ellipses move onto the object, and switch adaptive density control on to see Gaussians split and clone.',
  tags: [
    'gaussian splatting',
    '3dgs',
    'splats',
    'novel view',
    'multi-view',
    'densification',
    'rasterisation',
    'neural field',
  ],
  added: '2026-11-02',
}
