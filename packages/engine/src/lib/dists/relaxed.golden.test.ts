import { describeDistributions, type Family } from '../../testing/distributions.ts'
import type { Tensor } from '../tensor/index.ts'
import * as D from './index.ts'

const t = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor

const FAMILIES: Readonly<Record<string, Family>> = {
  relaxedBernoulli: (p) => D.relaxedBernoulli(t(p, 0), { logits: t(p, 1) }),
  relaxedOneHotCategorical: (p) => D.relaxedOneHotCategorical(t(p, 0), { logits: t(p, 1) }),
}

describeDistributions('dists/relaxed', FAMILIES)
