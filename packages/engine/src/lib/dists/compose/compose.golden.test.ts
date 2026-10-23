import { describeDistributions, type Family } from '../../../testing/distributions.ts'
import type { Tensor } from '../../tensor/index.ts'
import * as D from '../index.ts'
import * as C from './index.ts'

const t = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor
const B = D.bijectors

const FAMILIES: Readonly<Record<string, Family>> = {
  truncatedGamma: (p) => C.truncated(D.gamma(t(p, 0), t(p, 1)), { low: t(p, 2), high: t(p, 3) }),
  truncatedNormalAbove: (p) => C.truncated(D.normal(t(p, 0), t(p, 1)), { low: t(p, 2) }),
  truncatedNormalTail: (p) => C.truncated(D.normal(t(p, 0), t(p, 1)), { low: t(p, 2), high: t(p, 3) }),
  zeroTruncatedPoisson: (p) => C.truncated(D.poisson(t(p, 0)), { low: 1 }),
  expAffineNormal: (p) => C.transformed(D.normal(t(p, 0), t(p, 1)), [B.exp(), B.affine(1, 2)]),
  expNormal: (p) => C.transformed(D.normal(t(p, 0), t(p, 1)), B.exp()),
  empirical: (p) => C.empirical(t(p, 0), t(p, 1)),
}

for (const file of ['structure']) describeDistributions(`dists/compose/${file}`, FAMILIES)
