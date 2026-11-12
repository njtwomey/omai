// The reference of the Wishart-group site (`mvn-wishart-group`) in any dimension: importance sampling of the precision matrix against the dense joint Gaussian of
// the readings, none of the site's algebra. Node only; not published.
import { rawSum } from '../lib/message-passing/core/args.ts'
import { wishartParams } from '../lib/message-passing/core/families-matrix.ts'
import type { Eta } from '../lib/message-passing/core/types.ts'
import { spd } from '../lib/message-passing/numerics/_dense.ts'
import { foldIn, gamma, key, normal } from '../lib/random/index.ts'
import { toFlat } from '../lib/tensor/index.ts'

/** A Gaussian cavity as its mean `m` and covariance `v` (row-major). */
export interface Reading {
  readonly m: readonly number[]
  readonly v: readonly number[]
}

/**
 * The tilted matrix and the mean's posterior of the factor in any dimension, by importance sampling: matrices from the Wishart of the cavity with the
 * scatter of the vectors' cavity means added, weighted by the density of the readings given the matrix (the joint Gaussian of the readings
 * `m_j = mean + e_j`, `e_j ~ N(0, L^-1 + V_j)`, a dense `J d` covariance, none of the operator's algebra), self-normalised.
 */
export function importanceOfGroup(
  d: number,
  vs: readonly Reading[],
  mean: Reading,
  cavity: Eta,
  samples: number,
): { readonly lambda: number[]; readonly meanCov: number[]; readonly meanOfMean: number[]; readonly ess: number } {
  const count = vs.length
  const centre = Array.from({ length: d }, (_, k) => vs.reduce((acc, r) => acc + (r.m[k] as number), 0) / count)
  const scatter = new Array<number>(d * d).fill(0)
  for (const r of vs)
    for (let a = 0; a < d; a++)
      for (let b = 0; b < d; b++)
        scatter[a * d + b] =
          (scatter[a * d + b] as number) +
          ((r.m[a] as number) - (centre[a] as number)) * ((r.m[b] as number) - (centre[b] as number))
  const proposal = rawSum(cavity, [0.5 * (count - 1), ...scatter.map((v) => -0.5 * v)])
  const p = wishartParams(proposal, d) as NonNullable<ReturnType<typeof wishartParams>>
  // L = C A A' C' with C the Cholesky factor of the scale and A the Bartlett matrix
  const chol = Float64Array.from(p.scale)
  for (let j = 0; j < d; j++) {
    let acc = chol[j * d + j] as number
    for (let k = 0; k < j; k++) acc -= (chol[j * d + k] as number) ** 2
    chol[j * d + j] = Math.sqrt(acc)
    for (let i = j + 1; i < d; i++) {
      let t = chol[i * d + j] as number
      for (let k = 0; k < j; k++) t -= (chol[i * d + k] as number) * (chol[j * d + k] as number)
      chol[i * d + j] = t / (chol[j * d + j] as number)
    }
  }
  const root = key(11)
  const diag = Array.from({ length: d }, (_, i) => toFlat(gamma(foldIn(root, i), [samples], (p.nu - i) / 2)))
  const off = toFlat(normal(foldIn(root, 99), [samples, (d * (d - 1)) / 2]))
  const size = count * d
  const sInput = new Float64Array(size * size)
  const resid = vs.flatMap((r) => Array.from({ length: d }, (_, k) => (r.m[k] as number) - (mean.m[k] as number)))
  const logw: number[] = []
  const lambdas: number[][] = []
  const means: number[][] = []
  const covs: number[][] = []
  for (let s = 0; s < samples; s++) {
    const a = new Float64Array(d * d)
    let o = 0
    for (let i = 0; i < d; i++) {
      a[i * d + i] = Math.sqrt(2 * ((diag[i] as ArrayLike<number>)[s] as number))
      for (let j = 0; j < i; j++) a[i * d + j] = off[s * ((d * (d - 1)) / 2) + o++] as number
    }
    const t = new Float64Array(d * d)
    for (let i = 0; i < d; i++)
      for (let j = 0; j <= i; j++) {
        let acc = 0
        for (let q = j; q <= i; q++) acc += (chol[i * d + q] as number) * (a[q * d + j] as number)
        t[i * d + j] = acc
      }
    const lam = new Array<number>(d * d).fill(0)
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++)
        for (let q = 0; q <= Math.min(i, j); q++)
          lam[i * d + j] = (lam[i * d + j] as number) + (t[i * d + q] as number) * (t[j * d + q] as number)
    const factor = spd(lam, d) as NonNullable<ReturnType<typeof spd>>
    const inv = factor.inverse()
    for (let g = 0; g < count; g++)
      for (let h = 0; h < count; h++)
        for (let i = 0; i < d; i++)
          for (let j = 0; j < d; j++)
            sInput[(g * d + i) * size + h * d + j] =
              (mean.v[i * d + j] as number) +
              (g === h ? (inv[i * d + j] as number) + ((vs[g] as Reading).v[i * d + j] as number) : 0)
    const f = spd(sInput, size) as NonNullable<ReturnType<typeof spd>>
    const alpha = f.solve(resid)
    let quad = 0
    for (let e = 0; e < size; e++) quad += (resid[e] as number) * (alpha[e] as number)
    // prior (the cavity) over proposal: |L|^(-(J - 1) / 2) exp(tr(scatter L) / 2)
    let tr = 0
    for (let e = 0; e < d * d; e++) tr += (scatter[e] as number) * (lam[e] as number)
    logw.push(-0.5 * (f.logDet + quad) - 0.5 * (count - 1) * factor.logDet + 0.5 * tr)
    lambdas.push(lam)
    // the mean's posterior given the matrix: N(m0 + V0 H' S^-1 r, V0 - V0 H' S^-1 H V0), H = [I ... I]
    const sInv = f.inverse()
    // V0 H' S^-1: rows are the d mean coordinates against the J d readings
    const vh = Array.from({ length: d }, (_, i) =>
      Array.from({ length: size }, (_, e) => mean.v[i * d + (e % d)] as number),
    )
    const w = vh.map((row) =>
      Array.from({ length: size }, (_, e) => row.reduce((acc, v, q) => acc + v * (sInv[q * size + e] as number), 0)),
    )
    means.push(
      Array.from(
        { length: d },
        (_, i) => (mean.m[i] as number) + (w[i] as number[]).reduce((acc, v, e) => acc + v * (resid[e] as number), 0),
      ),
    )
    covs.push(
      Array.from({ length: d * d }, (_, e) => {
        const i = Math.floor(e / d)
        const j = e % d
        let acc = mean.v[i * d + j] as number
        for (let q = 0; q < size; q++) acc -= ((w[i] as number[])[q] as number) * ((vh[j] as number[])[q] as number)
        return acc
      }),
    )
  }
  const top = Math.max(...logw)
  const weights = logw.map((v) => Math.exp(v - top))
  const total = weights.reduce((acc, v) => acc + v, 0)
  const sq = weights.reduce((acc, v) => acc + (v / total) ** 2, 0)
  const average = (rows: readonly number[][], e: number): number =>
    rows.reduce((acc, r, i) => acc + ((weights[i] as number) / total) * (r[e] as number), 0)
  const meanOfMean = Array.from({ length: d }, (_, i) => average(means, i))
  return {
    lambda: Array.from({ length: d * d }, (_, e) => average(lambdas, e)),
    meanOfMean,
    meanCov: Array.from({ length: d * d }, (_, e) => {
      const i = Math.floor(e / d)
      const j = e % d
      return (
        average(covs, e) +
        average(
          means.map((m) => [(m[i] as number) * (m[j] as number)]),
          0,
        ) -
        (meanOfMean[i] as number) * (meanOfMean[j] as number)
      )
    }),
    ess: 1 / sq,
  }
}
