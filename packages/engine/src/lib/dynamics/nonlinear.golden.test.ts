import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { correlationDimension, largestLyapunovExponent } from './nonlinear.ts'

const FNS: Readonly<Record<string, GoldenFn>> = {
  corr_dim: ([s], p) => correlationDimension(toFlat(s as Tensor), p['emb'] as number).dimension,
  lyap_r: ([s]) =>
    largestLyapunovExponent(toFlat(s as Tensor), { dimension: 3, lag: 1, minTsep: 5, trajectoryLen: 6 }).exponent,
}

describe('dynamics/nonlinear against nolds', () => {
  runFixture('dynamics/nonlinear', FNS, it)
})
