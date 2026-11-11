// gaussian-transition: the messages against the exact marginals of the joint Gaussian of (x, w) (the cavities times the factor),
// for a square invertible matrix, a non-square one, an offset, and cavities that are flat or only partly informative.
import { describe, expect, it } from 'vitest'
import { dgemm } from '../../numerics/linalg/index.ts'
import { spd } from '../_dense.ts'
import { gaussianMv } from '../families-vector.ts'
import { messageArg } from './args.ts'
import { gaussianTransition } from './transition.ts'

/** The marginal natural parameters of x and of w in the joint of the two cavities and the factor, by inverting the joint precision. */
function tilted(
  cx: readonly number[],
  cw: readonly number[],
  F: readonly number[],
  c: readonly number[],
  Q: readonly number[],
  rows: number,
  cols: number,
) {
  const n = rows + cols
  const QInv = spd(Q, rows)?.inverse() as Float64Array
  const J = new Float64Array(n * n)
  const h = new Float64Array(n)
  for (let i = 0; i < rows; i++) {
    h[i] = (cx[i] as number) + QInv.slice(i * rows, (i + 1) * rows).reduce((s, v, k) => s + v * (c[k] as number), 0)
    for (let j = 0; j < rows; j++) J[i * n + j] = (cx[rows + i * rows + j] as number) + (QInv[i * rows + j] as number)
  }
  const QF = dgemm(QInv, F, rows, rows, cols)
  const FQF = dgemm(F, QF, cols, rows, cols, { transposeA: true })
  for (let i = 0; i < cols; i++) {
    let hv = cw[i] as number
    for (let k = 0; k < rows; k++) {
      let qc = 0
      for (let l = 0; l < rows; l++) qc += (QInv[k * rows + l] as number) * (c[l] as number)
      hv -= (F[k * cols + i] as number) * qc
    }
    h[rows + i] = hv
    for (let j = 0; j < cols; j++)
      J[(rows + i) * n + rows + j] = (cw[cols + i * cols + j] as number) + (FQF[i * cols + j] as number)
    for (let k = 0; k < rows; k++) {
      J[k * n + rows + i] = -(QF[k * cols + i] as number)
      J[(rows + i) * n + k] = -(QF[k * cols + i] as number)
    }
  }
  const cov = spd(J, n)?.inverse() as Float64Array
  const mean = spd(J, n)?.solve(h) as Float64Array
  const factor = spd(J, n) as NonNullable<ReturnType<typeof spd>>
  // log of the integral of the cavities times the factor: the Gaussian integral of the joint, with the factor's own constants.
  let hJh = 0
  for (let i = 0; i < n; i++) hJh += (h[i] as number) * (mean[i] as number)
  const QF0 = spd(Q, rows) as NonNullable<ReturnType<typeof spd>>
  let cQc = 0
  const Qc = QF0.solve(c)
  for (let i = 0; i < rows; i++) cQc += (c[i] as number) * (Qc[i] as number)
  const logZ =
    0.5 * hJh -
    0.5 * factor.logDet +
    0.5 * n * Math.log(2 * Math.PI) -
    0.5 * rows * Math.log(2 * Math.PI) -
    0.5 * QF0.logDet -
    0.5 * cQc
  const marginal = (from: number, d: number): number[] => {
    const sub = new Float64Array(d * d)
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) sub[i * d + j] = cov[(from + i) * n + from + j] as number
    const f = spd(sub, d) as NonNullable<ReturnType<typeof spd>>
    const P = f.inverse()
    const m = Array.from(mean.slice(from, from + d))
    return [
      ...Array.from(P).slice(0, 0),
      ...Array.from({ length: d }, (_, i) =>
        P.slice(i * d, (i + 1) * d).reduce((s, v, k) => s + v * (m[k] as number), 0),
      ),
      ...P,
    ]
  }
  return { x: marginal(0, rows), w: marginal(rows, cols), logZ }
}

describe.each([
  {
    name: 'a constant-velocity step (square, with an offset)',
    rows: 2,
    cols: 2,
    F: [1, 0.5, 0, 1],
    c: [0.1, -0.2],
    Q: [0.2, 0.05, 0.05, 0.1],
  },
  {
    name: 'a non-square matrix',
    rows: 3,
    cols: 2,
    F: [1, 0, 0, 1, 1, 1],
    c: [0, 0, 0],
    Q: [1, 0.2, 0, 0.2, 0.8, 0.1, 0, 0.1, 0.5],
  },
])('gaussian-transition: $name', ({ rows, cols, F, c, Q }) => {
  const fx = gaussianMv(rows)
  const fw = gaussianMv(cols)
  const spdOf = (n: number, scale: number): number[] =>
    Array.from({ length: n * n }, (_, k) =>
      Math.floor(k / n) === k % n ? scale + 0.3 * Math.floor(k / n) : 0.1 * scale,
    )
  const cavityX = [...Array.from({ length: rows }, (_, i) => 0.3 * (i + 1)), ...spdOf(rows, 1.5)]
  const cavityW = [...Array.from({ length: cols }, (_, i) => -0.4 * (i + 1)), ...spdOf(cols, 2)]
  const params = { matrix: F, offset: c, covariance: Q }
  const args = [messageArg(fx, cavityX), messageArg(fw, cavityW)]
  const ref = tilted(cavityX, cavityW, F, c, Q, rows, cols)
  const run = (to: number, a = args) =>
    gaussianTransition.ep?.({ mode: 'ep', to, args: a, params, power: 1 }).message ?? []

  it('the message to x times its cavity is the marginal of x in the joint', () => {
    const m = run(0)
    for (const [i, v] of m.entries()) expect(v + (cavityX[i] as number)).toBeCloseTo(ref.x[i] as number, 8)
  })

  it('the message to w times its cavity is the marginal of w in the joint', () => {
    const m = run(1)
    for (const [i, v] of m.entries()) expect(v + (cavityW[i] as number)).toBeCloseTo(ref.w[i] as number, 8)
  })

  it('sends nothing to w from a flat cavity of x, and a message from a partly informed one (one coordinate read)', () => {
    const flat = run(1, [messageArg(fx, new Array<number>(rows + rows * rows).fill(0)), messageArg(fw, cavityW)])
    expect(flat.every((v) => Math.abs(v) < 1e-12)).toBe(true)
    const partial = new Array<number>(rows + rows * rows).fill(0)
    partial[0] = 2
    partial[rows] = 4
    const m = run(1, [messageArg(fx, partial), messageArg(fw, cavityW)])
    expect(m.every(Number.isFinite)).toBe(true)
    // Its precision on w is positive semi-definite and not zero.
    expect(m.slice(cols).some((v) => v > 1e-6)).toBe(true)
  })

  it('has the log evidence of the integral: the cavities as unnormalised densities times the factor', () => {
    const own = gaussianTransition.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) ?? Number.NaN
    expect(own).toBeCloseTo(ref.logZ, 8)
  })
})
