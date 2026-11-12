import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Tracking a position and a velocity (vector-state scan)',
  summary:
    'A target with a hidden velocity, read through noisy positions: a scan over a two-dimensional state, EP on the chain of dense Gaussian transitions, against the exact Kalman smoother. Change the noise and the push.',
  tags: [
    'expectation propagation',
    'ep',
    'program',
    'kalman',
    'linear dynamical system',
    'state space',
    'scan',
    'tracking',
  ],
  added: '2026-11-11',
}
