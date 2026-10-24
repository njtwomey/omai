import { goldenSuite } from '../../../testing/suite.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { ols } from '../_linear.ts'
import * as B from './index.ts'

// The largest change in fitted values when projecting each column of `b` onto the span of `a` and back: zero when the
// two matrices have the same column space.
function spanDistance(a: Tensor, b: Tensor): number {
  const n = a.shape[0] ?? 0
  const p = a.shape[1] ?? 0
  const q = b.shape[1] ?? 0
  const ad = toFlat(a)
  const bd = toFlat(b)
  let worst = 0
  for (let c = 0; c < q; c++) {
    const col = Array.from({ length: n }, (_, i) => bd[i * q + c] ?? 0)
    const r = ols(ad, n, p, col).residuals
    for (const v of r) worst = Math.max(worst, Math.abs(v))
  }
  return worst
}

goldenSuite('smoothing bases against patsy and scipy', 'stats/smoothing/bases', {
  bs: ([x], p) => B.bs(x, { df: Number(p['df']), intercept: p['intercept'] as boolean }).X,
  bsKnots: ([x, knots]) => B.bs(x, { knots: toFlat(knots), degree: 2 }).X,
  cr: ([x], p) => B.cr(x, { k: Number(p['k']) }).X,
  cc: ([x], p) => B.cc(x, { k: Number(p['k']) }).X,
  te: ([x, z]) => B.te([B.cr(x, { k: 3 }), B.cc(z, { k: 5 })]).X,
  nsSpan: ([x, crx]) => spanDistance(B.ns(x, { df: 5, intercept: true }).X, crx),
  ps: ([x], p) => B.ps(x, { k: Number(p['k']) }).X,
})
