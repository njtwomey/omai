import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { dmd, sindy } from './identification.ts'

const CASES: Readonly<Record<string, GoldenFn>> = {
  sindy: ([x, xdot], p) =>
    sindy(x as Tensor, { derivatives: xdot as Tensor, degree: 2, threshold: p['threshold'] as number }).coefficients,
  dmd: ([snaps], p) => {
    const e = dmd(snaps as Tensor, { rank: p['rank'] as number }).eigenvalues
    const order = e.re
      .map((_, i) => i)
      .sort((a, b) => (e.re[a] ?? 0) - (e.re[b] ?? 0) || (e.im[a] ?? 0) - (e.im[b] ?? 0))
    return tensor([order.map((i) => e.re[i] ?? 0), order.map((i) => e.im[i] ?? 0)])
  },
}

describe('dynamics/identification against pysindy and numpy', () => runFixture('dynamics/identification', CASES, it))
