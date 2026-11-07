import { describe, expect, it } from 'vitest'
import { item } from '../../tensor/index.ts'
import * as S from './index.ts'

const INF = Number.POSITIVE_INFINITY

// Values at the edges of the domain: numpy / scipy (scipy 1.x, one-off run), and ndtri of denormal p against mpmath at
// 40 digits (the root of erfc(-x / sqrt 2) / 2 = p for the double p).
const edges: readonly (readonly [string, () => unknown, number])[] = [
  ['erf(inf)', () => S.erf(INF), 1],
  ['erf(-inf)', () => S.erf(-INF), -1],
  ['erfcx(inf)', () => S.erfcx(INF), 0],
  ['lgamma(inf)', () => S.lgamma(INF), INF],
  ['lgamma(-inf)', () => S.lgamma(-INF), -INF],
  ['logaddexp(-inf, -inf)', () => S.logaddexp(-INF, -INF), -INF],
  ['logaddexp(inf, inf)', () => S.logaddexp(INF, INF), INF],
  ['logaddexp(-inf, 3)', () => S.logaddexp(-INF, 3), 3],
  ['gammainc(0, 1)', () => S.gammainc(0, 1), 1],
  ['gammaincc(0, 1)', () => S.gammaincc(0, 1), 0],
  ['gamma(0)', () => S.gamma(0), INF],
  ['gamma(-0)', () => S.gamma(-0), -INF],
  ['digamma(0)', () => S.digamma(0), -INF],
  ['digamma(-0)', () => S.digamma(-0), INF],
  ['polygamma(1, 0)', () => S.polygamma(1, 0), INF],
  ['polygamma(2, 0)', () => S.polygamma(2, 0), -INF],
  ['zeta(inf)', () => S.zeta(INF), 1],
  ['hurwitzZeta(inf, 1)', () => S.hurwitzZeta(INF, 1), 1],
  ['hurwitzZeta(inf, 2)', () => S.hurwitzZeta(INF, 2), 0],
  ['hurwitzZeta(inf, 0.5)', () => S.hurwitzZeta(INF, 0.5), INF],
  ['hurwitzZeta(2, 0)', () => S.hurwitzZeta(2, 0), INF],
  ['hurwitzZeta(2, -1)', () => S.hurwitzZeta(2, -1), INF],
  ['lambertW(inf)', () => S.lambertW(INF), INF],
  ['expn(0, 0)', () => S.expn(0, 0), INF],
  ['expn(2, 0)', () => S.expn(2, 0), 1],
  ['ndtri(5e-324)', () => S.ndtri(5e-324), -38.467405617144344],
  ['ndtri(1e-323)', () => S.ndtri(1e-323), -38.44939448087599],
  ['ndtri(2e-322)', () => S.ndtri(2e-322), -38.37145454875547],
  ['ndtri(1e-320)', () => S.ndtri(1e-320), -38.26912534303265],
  ['ndtri(1e-318)', () => S.ndtri(1e-318), -38.14868137015514],
  ['ndtri(1e-315)', () => S.ndtri(1e-315), -37.96730035106736],
  ['ndtri(1e-310)', () => S.ndtri(1e-310), -37.663060331949524],
  ['ndtri(1e-300)', () => S.ndtri(1e-300), -37.0470962993612],
]

describe('special functions at infinity, at poles and in the denormal tails', () => {
  it.each(edges)('%s', (_name, call, expected) => {
    const v = item(call() as never)
    if (Number.isFinite(expected) && expected !== 0) expect(Math.abs(v / expected - 1)).toBeLessThan(1e-14)
    else expect(v).toBe(expected)
  })

  it.each([
    ['gammainc(0, 0)', () => S.gammainc(0, 0)],
    ['gamma(-1)', () => S.gamma(-1)],
    ['digamma(-1)', () => S.digamma(-1)],
    ['gamma(-inf)', () => S.gamma(-INF)],
    ['polygamma(2, -1)', () => S.polygamma(2, -1)],
    ['chdtr(2, nan)', () => S.chdtr(2, Number.NaN)],
    ['chdtrc(2, nan)', () => S.chdtrc(2, Number.NaN)],
    ['fdtr(2, 3, nan)', () => S.fdtr(2, 3, Number.NaN)],
    ['fdtrc(2, 3, nan)', () => S.fdtrc(2, 3, Number.NaN)],
  ])('%s stays NaN (undefined, as in scipy)', (_name, call) => {
    expect(item(call())).toBeNaN()
  })
})
