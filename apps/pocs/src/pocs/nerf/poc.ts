import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Neural radiance field (NeRF)',
  summary:
    'A tiny NeRF fitted to a few views of a toy scene (spheres and a cube). Play the training, then drag the novel view to fly a camera round the learned scene, and drag the 3-D box to see the density the network learned as a point cloud.',
  tags: [
    'nerf',
    'neural radiance field',
    'volume rendering',
    'novel view',
    'neural field',
    'multi-view',
    'positional encoding',
  ],
  added: '2026-11-02',
  group: 'graphics-and-fields',
}
