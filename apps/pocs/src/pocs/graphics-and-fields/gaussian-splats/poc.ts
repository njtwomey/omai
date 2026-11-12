import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: '3-D Gaussian splatting',
  summary:
    'A cloud of 3-D Gaussians fitted to a few views of a toy scene (spheres and a cube). Play the training, then drag the novel view to fly a camera round the fitted scene, and drag the 3-D box to see the Gaussians themselves; download them as a .ply.',
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
