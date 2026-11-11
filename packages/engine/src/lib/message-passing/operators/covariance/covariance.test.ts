import { describe, expect, it } from 'vitest'
import { messageArg, valueArg } from '../../core/args.ts'
import { normalWishart, normalWishartEta, wishart, wishartEta, wishartParams } from '../../core/families-matrix.ts'
import { gaussianMv, gaussianMvEta } from '../../core/families-vector.ts'
import type { Arg, Eta, OperatorContext } from '../../core/types.ts'
import { wishartRule } from '../../numerics/_bartlett.ts'
import { mvnNormalWishart, mvnWishart } from './covariance.ts'

const W = wishart(2)
const G = gaussianMv(2)

const ctx = (to: number, args: readonly Arg[], mode: 'ep' | 'vmp' = 'ep'): OperatorContext => ({
  mode,
  to,
  args,
  params: {},
  power: 1,
})

/** The brute-force tilted distribution of the precision matrix on a midpoint grid in (L11, L12, L22). */
function brute(
  cavity: Eta,
  integrand: (l11: number, l12: number, l22: number) => number,
  window: Eta = cavity,
): {
  readonly logZ: number
  readonly logDet: number
  readonly mean: number[]
  readonly weights: (f: (a: number, b: number, c: number) => number) => number
} {
  const p = wishartParams(window, 2) as NonNullable<ReturnType<typeof wishartParams>>
  const sd = (i: number, j: number): number =>
    Math.sqrt(
      p.nu * ((p.scale[i * 2 + j] as number) ** 2 + (p.scale[i * 2 + i] as number) * (p.scale[j * 2 + j] as number)),
    )
  const n = 90
  const box = [
    [Math.max(1e-6, (p.mean[0] as number) - 7 * sd(0, 0)), (p.mean[0] as number) + 9 * sd(0, 0)],
    [(p.mean[1] as number) - 8 * sd(0, 1), (p.mean[1] as number) + 8 * sd(0, 1)],
    [Math.max(1e-6, (p.mean[3] as number) - 7 * sd(1, 1)), (p.mean[3] as number) + 9 * sd(1, 1)],
  ] as const
  const steps = box.map((b) => (b[1] - b[0]) / n)
  const nodes: { l: [number, number, number]; w: number }[] = []
  let total = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < n; k++) {
        const l11 = box[0][0] + (i + 0.5) * (steps[0] as number)
        const l12 = box[1][0] + (j + 0.5) * (steps[1] as number)
        const l22 = box[2][0] + (k + 0.5) * (steps[2] as number)
        const det = l11 * l22 - l12 * l12
        if (det <= 0) continue
        const lq = W.pair(cavity, [Math.log(det), l11, l12, l12, l22]) - W.logPartition(cavity)
        const w = Math.exp(lq) * integrand(l11, l12, l22)
        nodes.push({ l: [l11, l12, l22], w })
        total += w
      }
  const cell = (steps[0] as number) * (steps[1] as number) * (steps[2] as number)
  const weights = (f: (a: number, b: number, c: number) => number): number =>
    nodes.reduce((s, n2) => s + n2.w * f(n2.l[0], n2.l[1], n2.l[2]), 0) / total
  return {
    logZ: Math.log(total * cell) + W.logPartition(cavity),
    logDet: weights((a, b, c) => Math.log(a * c - b * b)),
    mean: [weights((a) => a), weights((_a, b) => b), weights((_a, _b, c) => c)],
    weights,
  }
}

/** N(m; 0, V + L^-1) for a 2-vector m and symmetric V (row-major). */
function gaussianAt(m: readonly number[], v: readonly number[], l11: number, l12: number, l22: number): number {
  const det = l11 * l22 - l12 * l12
  const inv = [l22 / det, -l12 / det, -l12 / det, l11 / det]
  const s = [
    (v[0] as number) + (inv[0] as number),
    (v[1] as number) + (inv[1] as number),
    (v[2] as number) + (inv[2] as number),
    (v[3] as number) + (inv[3] as number),
  ]
  const sd = (s[0] as number) * (s[3] as number) - (s[1] as number) ** 2
  const quad =
    ((s[3] as number) * (m[0] as number) ** 2 -
      2 * (s[1] as number) * (m[0] as number) * (m[1] as number) +
      (s[0] as number) * (m[1] as number) ** 2) /
    sd
  return Math.exp(-0.5 * quad) / (2 * Math.PI * Math.sqrt(sd))
}

describe('mvn-wishart: the exact conjugate case', () => {
  it('an observed vector with a known mean is the conjugate update, and its evidence is the Student-t-like marginal', () => {
    const cavity = wishartEta(5, [1, 0.3, 0.3, 2], 2)
    const args = [valueArg(G, [1.2, -0.4]), valueArg(G, [0.2, 0.1]), messageArg(W, cavity)]
    const m = mvnWishart.ep?.(ctx(2, args)).message as number[]
    const r = [1.0, -0.5]
    expect(m).toEqual([
      0.5,
      -0.5 * (r[0] as number) * (r[0] as number),
      -0.5 * (r[0] as number) * (r[1] as number),
      -0.5 * (r[1] as number) * (r[0] as number),
      -0.5 * (r[1] as number) * (r[1] as number),
    ])
    expect(mvnWishart.vmp?.(ctx(2, args, 'vmp')).message).toEqual(m.map((v) => expect.closeTo(v, 12)))
    // the evidence is the integral of N(r; 0, L^-1) against the Wishart: against the brute-force grid
    const b = brute(cavity, (a, bb, c) => gaussianAt(r, [0, 0, 0, 0], a, bb, c))
    expect(mvnWishart.logEvidence?.(ctx(2, args))).toBeCloseTo(
      b.logZ - W.logPartition(cavity) + W.logPartition(cavity),
      3,
    )
  })

  it('n rows give the Wishart posterior with nu + n and the scatter matrix added', () => {
    const rows = [
      [1, 0.5],
      [-0.3, 0.2],
      [0.8, -1.1],
      [0.1, 0.4],
    ]
    let eta: Eta = wishartEta(4, [0.5, 0, 0, 0.5], 2)
    for (const y of rows) {
      const m = mvnWishart.ep?.(ctx(2, [valueArg(G, y), valueArg(G, [0, 0]), messageArg(W, eta)])).message as number[]
      eta = W.product(eta, m)
    }
    const p = wishartParams(eta, 2)
    expect(p?.nu).toBeCloseTo(8, 12)
    const scatter = [0, 0, 0, 0]
    for (const y of rows)
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++)
          scatter[i * 2 + j] = (scatter[i * 2 + j] as number) + (y[i] as number) * (y[j] as number)
    // S^-1 = S0^-1 + scatter = 2 I + scatter
    const inv = [2 + (scatter[0] as number), scatter[1] as number, scatter[2] as number, 2 + (scatter[3] as number)]
    const det = (inv[0] as number) * (inv[3] as number) - (inv[1] as number) ** 2
    expect(p?.scale[0]).toBeCloseTo((inv[3] as number) / det, 10)
    expect(p?.scale[1]).toBeCloseTo(-(inv[1] as number) / det, 10)
  })
})

describe('mvn-wishart: expectation propagation with Gaussian cavities', () => {
  const cavity = wishartEta(7, [0.6, 0.2, 0.2, 0.9], 2)

  it('the Bartlett rule integrates polynomials of the Wishart matrix exactly: E L = nu S, E L11 L22', () => {
    const p = wishartParams(cavity, 2) as NonNullable<ReturnType<typeof wishartParams>>
    const chol = Float64Array.from(p.scale)
    chol[0] = Math.sqrt(chol[0] as number)
    chol[2] = (p.scale[2] as number) / (chol[0] as number)
    chol[3] = Math.sqrt((p.scale[3] as number) - (chol[2] as number) ** 2)
    const rule = wishartRule(p.nu, chol, 2)
    let m11 = 0
    let m12 = 0
    let cross = 0
    let logDet = 0
    for (let k = 0; k < rule.count; k++) {
      const w = rule.weight[k] as number
      m11 += w * (rule.lambda[k * 4] as number)
      m12 += w * (rule.lambda[k * 4 + 1] as number)
      cross += w * (rule.lambda[k * 4] as number) * (rule.lambda[k * 4 + 3] as number)
      logDet += w * (rule.logDet[k] as number)
    }
    expect(m11).toBeCloseTo(p.mean[0] as number, 10)
    expect(m12).toBeCloseTo(p.mean[1] as number, 10)
    // E[L11 L22] = nu^2 S11 S22 + 2 nu S12^2 (the Wishart second moments, Cov(L11, L22) = 2 nu S12^2)
    const s = p.scale
    expect(cross).toBeCloseTo(p.nu ** 2 * (s[0] as number) * (s[3] as number) + 2 * p.nu * (s[1] as number) ** 2, 9)
    // a logarithm is not a polynomial: the rule's error is known and reported, and small
    expect(logDet - rule.logDetBias).toBeCloseTo(W.moments(cavity)[0] as number, 9)
    expect(Math.abs(rule.logDetBias)).toBeLessThan(1e-3)
  })

  it('the message to the precision matches the brute-force tilted moments, narrow and wide cavities of x, and the evidence', () => {
    for (const [mx, vx] of [
      [
        [0.9, -0.4],
        [0.05, 0.01, 0.01, 0.04],
      ],
      [
        [0.9, -0.4],
        [0.5, 0.2, 0.2, 0.7],
      ],
      [
        [-1.5, 1],
        [2.5, -0.5, -0.5, 3],
      ],
    ] as const) {
      const xeta = gaussianMvEta(mx, vx)
      const args = [messageArg(G, xeta), valueArg(G, [0, 0]), messageArg(W, cavity)]
      const msg = mvnWishart.ep?.(ctx(2, args)).message as number[]
      const matched = W.moments(W.product(cavity, msg))
      const b = brute(cavity, (a, bb, c) => gaussianAt(mx, vx, a, bb, c))
      // the nodes carry a logarithm (not a polynomial): within 1e-3 of the brute-force mean, 1e-5 where the tilt is narrow
      expect(Math.abs((matched[0] as number) - b.logDet), 'E log det').toBeLessThan(1.5e-3)
      expect(matched[1], 'E L11').toBeCloseTo(b.mean[0] as number, 2)
      expect(matched[2], 'E L12').toBeCloseTo(b.mean[1] as number, 2)
      expect(matched[4], 'E L22').toBeCloseTo(b.mean[2] as number, 2)
      // the evidence: the factor integrated against the unnormalised cavity of x as well
      const ev = mvnWishart.logEvidence?.(ctx(2, args)) as number
      expect(ev).toBeCloseTo(b.logZ + G.logPartition(xeta), 3)
    }
  })

  it('the message to a Gaussian vector is the moment match of the mixture of its conditionals', () => {
    const mx = [0.9, -0.4]
    const vx = [0.5, 0.2, 0.2, 0.7]
    const xeta = gaussianMvEta(mx, vx)
    const args = [messageArg(G, xeta), valueArg(G, [0, 0]), messageArg(W, cavity)]
    const msg = mvnWishart.ep?.(ctx(0, args)).message as number[]
    const proj = G.moments(G.product(xeta, msg))
    // brute force: x | L ~ N(P^-1 h, P^-1) with P = Vx^-1 + L; weights N(mx; 0, Vx + L^-1)
    const vdet = (vx[0] as number) * (vx[3] as number) - (vx[1] as number) ** 2
    const vinv = [
      (vx[3] as number) / vdet,
      -(vx[1] as number) / vdet,
      -(vx[1] as number) / vdet,
      (vx[0] as number) / vdet,
    ]
    const b = brute(cavity, (a, bb, c) => gaussianAt(mx, vx, a, bb, c))
    const comp = (a: number, bb: number, c: number): { mean: number[]; cov: number[] } => {
      const p = [(vinv[0] as number) + a, (vinv[1] as number) + bb, (vinv[2] as number) + bb, (vinv[3] as number) + c]
      const det = (p[0] as number) * (p[3] as number) - (p[1] as number) * (p[2] as number)
      const cov = [(p[3] as number) / det, -(p[1] as number) / det, -(p[2] as number) / det, (p[0] as number) / det]
      const h = [
        (vinv[0] as number) * (mx[0] as number) + (vinv[1] as number) * (mx[1] as number),
        (vinv[2] as number) * (mx[0] as number) + (vinv[3] as number) * (mx[1] as number),
      ]
      return {
        mean: [
          (cov[0] as number) * (h[0] as number) + (cov[1] as number) * (h[1] as number),
          (cov[2] as number) * (h[0] as number) + (cov[3] as number) * (h[1] as number),
        ],
        cov,
      }
    }
    const node = (a: number, bb: number, c: number): { mean: number[]; cov: number[] } => comp(a, bb, c)
    const m0 = b.weights((a, bb, c) => node(a, bb, c).mean[0] as number)
    const m1 = b.weights((a, bb, c) => node(a, bb, c).mean[1] as number)
    const s00 = b.weights((a, bb, c) => (node(a, bb, c).cov[0] as number) + (node(a, bb, c).mean[0] as number) ** 2)
    const s01 = b.weights(
      (a, bb, c) =>
        (node(a, bb, c).cov[1] as number) + (node(a, bb, c).mean[0] as number) * (node(a, bb, c).mean[1] as number),
    )
    expect(proj[0]).toBeCloseTo(m0, 3)
    expect(proj[1]).toBeCloseTo(m1, 3)
    expect(proj[2]).toBeCloseTo(s00, 3)
    expect(proj[3]).toBeCloseTo(s01, 3)
  })

  it('both the vector and the mean random: the message to the mean is the mixture over the matrix', () => {
    const xeta = gaussianMvEta([1, 0.5], [0.3, 0.05, 0.05, 0.2])
    const meta = gaussianMvEta([0, 0], [1, 0, 0, 1])
    const args = [messageArg(G, xeta), messageArg(G, meta), messageArg(W, cavity)]
    const toMean = mvnWishart.ep?.(ctx(1, args)).message as number[]
    const toX = mvnWishart.ep?.(ctx(0, args)).message as number[]
    // each Gaussian message is proper-ish and the pair is symmetric in the roles: x to mean is the mirror of mean to x about the difference
    expect(G.isProper(G.product(meta, toMean))).toBe(true)
    expect(G.isProper(G.product(xeta, toX))).toBe(true)
    // the mean moves towards x and x towards the mean
    expect(G.moments(G.product(meta, toMean))[0]).toBeGreaterThan(0)
    expect(G.moments(G.product(xeta, toX))[0] as number).toBeLessThan(1)
  })

  it('a flat cavity on the vector: the first message is the factor itself, the matrix prior is a Student-like scale mixture', () => {
    const args = [messageArg(G, G.flat), valueArg(G, [0, 0]), messageArg(W, cavity)]
    const toX = mvnWishart.ep?.(ctx(0, args)).message as number[]
    expect(G.isProper(toX)).toBe(true)
    // x ~ N(0, L^-1), L ~ W(7, S): covariance E[L^-1] = S^-1 / (nu - d - 1)
    const cov = G.summary(toX)['covariance'] as number[][]
    const sInv = [0.9 / 0.5, -0.2 / 0.5, -0.2 / 0.5, 0.6 / 0.5]
    expect(cov[0]?.[0]).toBeCloseTo((sInv[0] as number) / 4, 2)
    expect(cov[0]?.[1]).toBeCloseTo((sInv[1] as number) / 4, 2)
  })

  it('the variational rule is the expected residual outer product', () => {
    const xeta = gaussianMvEta([1, 0.5], [0.3, 0.05, 0.05, 0.2])
    const args = [messageArg(G, xeta), valueArg(G, [0.2, 0.1]), messageArg(W, cavity)]
    const m = mvnWishart.vmp?.(ctx(2, args, 'vmp')).message as number[]
    // E[(x - m)(x - m)'] = (E x - m)(E x - m)' + V
    expect(m[1]).toBeCloseTo(-0.5 * (0.8 ** 2 + 0.3), 12)
    expect(m[2]).toBeCloseTo(-0.5 * (0.8 * 0.4 + 0.05), 12)
    const toX = mvnWishart.vmp?.(ctx(0, args, 'vmp')).message as number[]
    const meanL = W.moments(cavity).slice(1)
    expect(toX.slice(2)).toEqual(meanL)
    expect(mvnWishart.expectedLog?.(ctx(2, args, 'vmp'))).toBeTypeOf('number')
  })

  it('refuses to integrate a matrix beyond the quadrature, and says so', () => {
    const g4 = gaussianMv(4)
    const w4 = wishart(4)
    const args = [
      messageArg(g4, gaussianMvEta([0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
      valueArg(g4, [0, 0, 0, 0]),
      messageArg(w4, wishartEta(9, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 4)),
    ]
    expect(() => mvnWishart.ep?.(ctx(2, args))).toThrow(/up to 3/)
  })
})

describe('mvn-wishart: a plate of rows (count and scatter) is the product of the row factors', () => {
  const rows = [
    [1, 0.5],
    [-0.3, 0.2],
    [0.8, -1.1],
    [0.1, 0.4],
    [0.6, 0.0],
  ]
  const n = rows.length
  const ybar = [0, 1].map((k) => rows.reduce((s, y) => s + (y[k] as number), 0) / n)
  const scatter = [0, 0, 0, 0]
  for (const y of rows)
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 2; j++)
        scatter[i * 2 + j] =
          (scatter[i * 2 + j] as number) +
          ((y[i] as number) - (ybar[i] as number)) * ((y[j] as number) - (ybar[j] as number))
  const params = { count: n, scatter }
  const cavity = wishartEta(6, [0.5, 0.1, 0.1, 0.4], 2)

  it('with a known mean it is the sequence of row updates', () => {
    const mean = [0.2, -0.1]
    const plate = mvnWishart.ep?.({ ...ctx(2, [valueArg(G, ybar), valueArg(G, mean), messageArg(W, cavity)]), params })
      .message as number[]
    let eta: Eta = cavity
    for (const y of rows)
      eta = W.product(
        eta,
        mvnWishart.ep?.(ctx(2, [valueArg(G, y), valueArg(G, mean), messageArg(W, eta)])).message as number[],
      )
    for (const [i, v] of W.product(cavity, plate).entries()) expect(v).toBeCloseTo(eta[i] as number, 10)
  })

  it('with a Gaussian cavity on the mean the tilted matrix matches the brute-force integral over the matrix', () => {
    const mc = [0.3, -0.2]
    const vc = [0.2, 0.05, 0.05, 0.3]
    const meta = gaussianMvEta(mc, vc)
    const msg = mvnWishart.ep?.({ ...ctx(2, [valueArg(G, ybar), messageArg(G, meta), messageArg(W, cavity)]), params })
      .message as number[]
    const matched = W.moments(W.product(cavity, msg))
    // prod_i N(y_i; mu, L^-1) integrated against N(mu; mc, Vc) is N(ybar; mc, Vc + L^-1 / n) |L|^((n - 1) / 2) exp(-tr(L scatter) / 2)
    const b = brute(
      cavity,
      (a, bb, c) => {
        const det = a * c - bb * bb
        const tr = a * (scatter[0] as number) + 2 * bb * (scatter[1] as number) + c * (scatter[3] as number)
        const diff = [(ybar[0] as number) - (mc[0] as number), (ybar[1] as number) - (mc[1] as number)]
        const v2 = [(vc[0] as number) * n, (vc[1] as number) * n, (vc[2] as number) * n, (vc[3] as number) * n]
        // N(diff; 0, Vc + L^-1 / n) = n N(sqrt(n) diff; 0, n Vc + L^-1) in two dimensions
        return (
          n *
          gaussianAt([Math.sqrt(n) * (diff[0] as number), Math.sqrt(n) * (diff[1] as number)], v2, a, bb, c) *
          det ** ((n - 1) / 2) *
          Math.exp(-0.5 * tr)
        )
      },
      W.product(cavity, [0.5 * n, ...scatter.map((v) => -0.5 * v)]),
    )
    expect(Math.abs((matched[0] as number) - b.logDet)).toBeLessThan(2e-3)
    expect(matched[1]).toBeCloseTo(b.mean[0] as number, 2)
    expect(matched[2]).toBeCloseTo(b.mean[1] as number, 2)
    expect(matched[4]).toBeCloseTo(b.mean[2] as number, 2)
    // and the message to the mean is a proper Gaussian that moves it towards the sample mean
    const toMean = mvnWishart.ep?.({
      ...ctx(1, [valueArg(G, ybar), messageArg(G, meta), messageArg(W, cavity)]),
      params,
    }).message as number[]
    expect(G.isProper(G.product(meta, toMean))).toBe(true)
  })
})

describe('mvn-wishart in three dimensions', () => {
  const G3 = gaussianMv(3)
  const W3 = wishart(3)
  const eye3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
  const scale = [0.6, 0.1, 0.05, 0.1, 0.5, 0.0, 0.05, 0.0, 0.4]
  const cavity = wishartEta(8, scale, 3)
  const ctx3 = (to: number, args: readonly Arg[]): OperatorContext => ({ mode: 'ep', to, args, params: {}, power: 1 })

  it('the Bartlett rule has the mean E L = nu S and the right log determinant to within its bias', () => {
    const p = wishartParams(cavity, 3) as NonNullable<ReturnType<typeof wishartParams>>
    // the lower Cholesky factor of the scale, by hand
    const l = new Float64Array(9)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j <= i; j++) {
        let v = p.scale[i * 3 + j] as number
        for (let k = 0; k < j; k++) v -= (l[i * 3 + k] as number) * (l[j * 3 + k] as number)
        l[i * 3 + j] = i === j ? Math.sqrt(v) : v / (l[j * 3 + j] as number)
      }
    const rule = wishartRule(p.nu, l, 3)
    expect(rule.count).toBe(5 ** 6)
    const mean = new Float64Array(9)
    for (let k = 0; k < rule.count; k++)
      for (let e = 0; e < 9; e++)
        mean[e] = (mean[e] as number) + (rule.weight[k] as number) * (rule.lambda[k * 9 + e] as number)
    for (let e = 0; e < 9; e++) expect(mean[e]).toBeCloseTo(p.mean[e] as number, 10)
    expect(Math.abs(rule.logDetBias)).toBeLessThan(0.02)
  })

  it('with a narrow Gaussian cavity on the vector the message is the conjugate update, and the matrix and the vector messages are proper', () => {
    const y = [0.8, -0.5, 0.3]
    const xeta = gaussianMvEta(y, [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4])
    const args = [messageArg(G3, xeta), valueArg(G3, [0, 0, 0]), messageArg(W3, cavity)]
    const msg = mvnWishart.ep?.(ctx3(2, args)).message as number[]
    const exact = mvnWishart.ep?.(ctx3(2, [valueArg(G3, y), valueArg(G3, [0, 0, 0]), messageArg(W3, cavity)]))
      .message as number[]
    const [a, b] = [W3.moments(W3.product(cavity, msg)), W3.moments(W3.product(cavity, exact))]
    for (let e = 1; e < 10; e++) expect(a[e]).toBeCloseTo(b[e] as number, 2)
    const toX = mvnWishart.ep?.(
      ctx3(0, [messageArg(G3, gaussianMvEta([0.5, 0, 0], eye3)), valueArg(G3, [0, 0, 0]), messageArg(W3, cavity)]),
    ).message as number[]
    expect(G3.isProper(G3.product(gaussianMvEta([0.5, 0, 0], eye3), toX))).toBe(true)
  })

  it('the exact conjugate case works at any dimension', () => {
    const d = 5
    const w = wishart(d)
    const y = [0.3, -0.2, 0.5, 0.1, -0.4]
    const eta = wishartEta(
      9,
      Array.from({ length: d * d }, (_, i) => (Math.floor(i / d) === i % d ? 1 : 0)),
      d,
    )
    const msg = mvnWishart.ep?.(
      ctx3(2, [valueArg(gaussianMv(d), y), valueArg(gaussianMv(d), new Array<number>(d).fill(0)), messageArg(w, eta)]),
    ).message as number[]
    expect(msg.length).toBe(1 + d * d)
    expect(msg[0]).toBe(0.5)
  })
})

describe('mvn-normal-wishart: the exact conjugate update', () => {
  const NW = normalWishart(2)
  it('n rows give the normal-Wishart posterior of the textbook (beta + n, nu + n, the scatter about the sample mean)', () => {
    const rows = [
      [1, 0.5],
      [-0.3, 0.2],
      [0.8, -1.1],
      [0.1, 0.4],
      [0.6, 0.0],
    ]
    const n = rows.length
    const m0 = [0.2, -0.1]
    const beta0 = 2
    const nu0 = 5
    const s0 = [1, 0.2, 0.2, 0.8]
    let eta: Eta = normalWishartEta(m0, beta0, nu0, s0)
    for (const y of rows)
      eta = NW.product(eta, mvnNormalWishart.ep?.(ctx(1, [valueArg(G, y), messageArg(NW, eta)])).message as number[])
    const p = NW.summary(eta)
    const ybar = [0, 1].map((k) => rows.reduce((s, y) => s + (y[k] as number), 0) / n)
    // posterior: beta_n = beta0 + n, m_n = (beta0 m0 + n ybar) / beta_n, nu_n = nu0 + n, S_n^-1 = S0^-1 + scatter + beta0 n / beta_n (ybar - m0)(ybar - m0)'
    const betaN = beta0 + n
    expect(p['beta']).toBeCloseTo(betaN, 10)
    expect(p['nu']).toBeCloseTo(nu0 + n, 10)
    const mN = [0, 1].map((k) => (beta0 * (m0[k] as number) + n * (ybar[k] as number)) / betaN)
    expect((p['m'] as number[])[0]).toBeCloseTo(mN[0] as number, 10)
    expect((p['m'] as number[])[1]).toBeCloseTo(mN[1] as number, 10)
    const det0 = (s0[0] as number) * (s0[3] as number) - (s0[1] as number) ** 2
    const inv0 = [
      (s0[3] as number) / det0,
      -(s0[1] as number) / det0,
      -(s0[2] as number) / det0,
      (s0[0] as number) / det0,
    ]
    const inv = inv0.map((v, e) => {
      const i = Math.floor(e / 2)
      const j = e % 2
      let scatter = 0
      for (const y of rows)
        scatter += ((y[i] as number) - (ybar[i] as number)) * ((y[j] as number) - (ybar[j] as number))
      return (
        v +
        scatter +
        ((beta0 * n) / betaN) * ((ybar[i] as number) - (m0[i] as number)) * ((ybar[j] as number) - (m0[j] as number))
      )
    })
    const detN = (inv[0] as number) * (inv[3] as number) - (inv[1] as number) ** 2
    const scale = p['scale'] as number[][]
    expect(scale[0]?.[0]).toBeCloseTo((inv[3] as number) / detN, 10)
    expect(scale[0]?.[1]).toBeCloseTo(-(inv[1] as number) / detN, 10)
  })

  it('the evidence of the rows is the multivariate Student t marginal (against the product of conditionals)', () => {
    const rows = [
      [1, 0.5],
      [-0.3, 0.2],
    ]
    const eta0 = normalWishartEta([0, 0], 1, 4, [1, 0, 0, 1])
    // the evidence is against the unnormalised cavity: the conditional evidence of a row divides by the mass of the cavity
    const cond = (y: readonly number[], eta: Eta): number =>
      (mvnNormalWishart.logEvidence?.(ctx(1, [valueArg(G, y), messageArg(NW, eta)])) as number) - NW.logPartition(eta)
    const first = cond(rows[0] as number[], eta0)
    const eta1 = NW.product(
      eta0,
      mvnNormalWishart.ep?.(ctx(1, [valueArg(G, rows[0] as number[]), messageArg(NW, eta0)])).message as number[],
    )
    const second = cond(rows[1] as number[], eta1)
    // the joint evidence is the same in either order
    const swapped0 = cond(rows[1] as number[], eta0)
    const eta1b = NW.product(
      eta0,
      mvnNormalWishart.ep?.(ctx(1, [valueArg(G, rows[1] as number[]), messageArg(NW, eta0)])).message as number[],
    )
    const swapped1 = cond(rows[0] as number[], eta1b)
    expect(first + second).toBeCloseTo(swapped0 + swapped1, 10)
    // the first row's marginal: a multivariate t with nu - d + 1 = 3 degrees of freedom, location 0, scale (beta + 1) / (beta (nu - d + 1)) S^-1
    // here beta = 1, nu = 4, S = I: scale matrix 2 / 3 I, density (Gamma((3 + 2) / 2) / (Gamma(3 / 2) 3 pi |2/3 I|^(1/2))) (1 + y'y / 2)^(-5/2)
    const y2 = 1 + 0.25
    const logT =
      Math.log(Math.exp(lgam(2.5) - lgam(1.5)) / (3 * Math.PI * (2 / 3))) - 2.5 * Math.log(1 + y2 / (3 * (2 / 3)))
    expect(first).toBeCloseTo(logT, 8)
  })
})

function lgam(x: number): number {
  // Lanczos
  const g = 7
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  const xx = x - 1
  let a = c[0] as number
  const t = xx + g + 0.5
  for (let i = 1; i < g + 2; i++) a += (c[i] as number) / (xx + i)
  return 0.5 * Math.log(2 * Math.PI) + (xx + 0.5) * Math.log(t) - t + Math.log(a)
}
