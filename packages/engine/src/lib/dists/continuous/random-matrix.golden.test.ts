import { describeDistributions } from '../../../testing/distributions.ts'
import type { Tensor } from '../../tensor/index.ts'
import { marchenkoPastur } from './random-matrix.ts'

// The reference is the textbook density integrated numerically by scipy (scipy.stats has no Marchenko-Pastur law).
describeDistributions('dists/continuous/random_matrix', {
  marchenkoPastur: (p) => marchenkoPastur(p[0] as Tensor, p[1] as Tensor),
})
