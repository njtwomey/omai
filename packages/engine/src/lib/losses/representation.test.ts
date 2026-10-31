import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { grad } from '../autodiff/index.ts'
import { key, normal } from '../random/index.ts'
import { add, item, matmul, mul, sub, sum, tensor, toArray, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const a = tensor([
  [0.3, -1.2, 2],
  [0.5, 0.15, -0.7],
  [1.1, 0.4, 0.2],
  [-0.6, 0.9, 0.3],
  [0.2, -0.4, 1.1],
])
const b = tensor([
  [0.2, -1, 1.5],
  [0.9, 0.1, -0.2],
  [1, -0.3, 0.6],
  [-0.2, 1.4, -0.5],
  [0.1, -0.5, 0.7],
])

describe('Barlow Twins, VICReg, alignment and uniformity', () => {
  for (const [name, f] of [
    ['Barlow Twins', (x: Value, y: Value) => L.barlowTwins(x, y, { lambda: 0.1 })],
    ['VICReg', (x: Value, y: Value) => L.vicreg(x, y)],
    ['alignment', (x: Value, y: Value) => L.alignment(x, y, { alpha: 3 })],
    ['uniformity', (x: Value) => L.uniformity(x, { t: 0.7 })],
  ] as const) {
    it(`${name} passes the conformance check`, () =>
      expectConformant(f as (...xs: Value[]) => Value, name === 'uniformity' ? [a] : [a, b]))
  }

  it('gives Barlow Twins zero on identical views with orthogonal standardised dimensions and penalises redundancy', () => {
    const orthogonal = [
      [1, 1],
      [-1, 1],
      [1, -1],
      [-1, -1],
    ]
    expect(item(L.barlowTwins(orthogonal, orthogonal, { eps: 0 }))).toBeCloseTo(0, 14)
    const redundant = [
      [1, 1],
      [-1, -1],
      [1, 1],
      [-1, -1],
    ]
    // The cross-correlation is all ones: the diagonal is right, the two off-diagonal entries cost lambda each.
    expect(item(L.barlowTwins(redundant, redundant, { eps: 0, lambda: 0.5 }))).toBeCloseTo(1, 14)
  })

  it('gives VICReg the variance hinge of a collapsed batch and nothing of a spread, decorrelated one', () => {
    const collapsed = [
      [2, 2],
      [2, 2],
      [2, 2],
    ]
    expect(item(L.vicreg(collapsed, collapsed, { eps: 1e-4 }))).toBeCloseTo(25 * (1 - Math.sqrt(1e-4)), 12)
    const spread = [
      [1, 1],
      [-1, 1],
      [1, -1],
      [-1, -1],
    ]
    // Unit variance (ddof 1 gives 4/3, above gamma), identical views, and diagonal covariance with zero off-diagonal.
    expect(item(L.vicreg(spread, spread, { eps: 0 }))).toBeCloseTo(0, 14)
  })

  it('measures alignment and uniformity on the sphere', () => {
    expect(item(L.alignment([[1, 0]], [[0, 1]]))).toBe(2)
    expect(
      item(
        L.uniformity(
          [
            [1, 0],
            [-1, 0],
          ],
          { t: 1 },
        ),
      ),
    ).toBeCloseTo(-4, 12)
    const clustered = [
      [1, 0],
      [1, 0],
      [0.99, 0.141],
      [0.99, -0.141],
    ]
    const spread = [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
    ]
    expect(item(L.uniformity(spread))).toBeLessThan(item(L.uniformity(clustered)))
  })
})

describe('noise-contrastive estimation', () => {
  it('learns the log-normaliser of an unnormalised Gaussian from data and noise samples', () => {
    // Data N(0, 1) and the unnormalised model s(x) = -x^2 / 2, whose log partition is log sqrt(2 pi). The noise is
    // N(0, 2^2), log q(x) = -x^2 / 8 - log(2 sqrt(2 pi)).
    const n = 4000
    const xd = normal(key(1), [n])
    const xn = normal(key(2), [n], { scale: 2 })
    const score = (x: Value): Value => mul(mul(x, x), -0.5)
    const logQ = (x: Value): Value => add(mul(mul(x, x), -1 / 8), -Math.log(2 * Math.sqrt(2 * Math.PI)))
    const loss = (c: Value): Value => L.noiseContrastive(score(xd), logQ(xd), score(xn), logQ(xn), { logNormaliser: c })
    let c: Value = tensor(0)
    for (let i = 0; i < 200; i++) c = sub(c, mul(grad(loss)(c), 2))
    expect(item(c)).toBeCloseTo(Math.log(Math.sqrt(2 * Math.PI)), 1)
  })

  it('is 2 log 2 for a model that is the noise itself', () => {
    expect(item(L.noiseContrastive([0, 0], [0, 0], [0, 0], [0, 0]))).toBeCloseTo(2 * Math.LN2, 14)
  })

  it('passes the conformance check', () => {
    expectConformant(
      (sd, sn, c) =>
        L.noiseContrastive(sd, [0.1, -0.2, 0.3], sn, [0.4, 0.5, -0.1, 0.2, 0.3, 0.1], { logNormaliser: c }),
      [tensor([0.3, -0.7, 1.1]), tensor([0.2, 0.5, -0.4, 0.9, -1.2, 0.1]), tensor(0.4)],
    )
  })
})

describe('score matching', () => {
  const x = tensor([[1.2], [-0.7], [0.3], [2.1], [-1.6]])
  const m2 = 0.2 * (1.2 ** 2 + 0.7 ** 2 + 0.3 ** 2 + 2.1 ** 2 + 1.6 ** 2)

  it('has the exact objective -theta + theta^2 m2 / 2 for the linear score -theta x, minimal at theta = 1 / m2', () => {
    const at = (theta: number): number => item(L.scoreMatching((v: Value) => mul(v, -theta), x))
    for (const theta of [0.4, 1, 1.7]) expect(at(theta)).toBeCloseTo(-theta + (theta * theta * m2) / 2, 13)
    expect(at(1 / m2)).toBeLessThan(at(1 / m2 + 0.05))
    expect(at(1 / m2)).toBeLessThan(at(1 / m2 - 0.05))
  })

  it('has the sliced objective equal the exact one in one dimension with Rademacher directions', () => {
    const exact = item(L.scoreMatching((v: Value) => mul(v, -0.8), x))
    const sliced = item(L.slicedScoreMatching((v: Value) => mul(v, -0.8), x, key(0), { kind: 'rademacher' }))
    expect(sliced).toBeCloseTo(exact, 13)
  })

  it('estimates the exact objective of a 2-D linear score with many random projections', () => {
    const data = normal(key(3), [64, 2])
    const A = tensor([
      [0.9, 0.2],
      [-0.1, 1.3],
    ])
    const score = (v: Value): Value => mul(matmul(v, A), -1)
    const exact = item(L.scoreMatching(score, data))
    const sliced = item(L.slicedScoreMatching(score, data, key(4), { projections: 400 }))
    const reduced = item(L.slicedScoreMatching(score, data, key(5), { projections: 400, varianceReduced: true }))
    expect(Math.abs(sliced - exact)).toBeLessThan(0.1)
    expect(Math.abs(reduced - exact)).toBeLessThan(0.05)
  })

  it('is differentiable in the parameters of the score model, through the forward-mode derivative', () => {
    const f = (theta: Value): Value =>
      L.slicedScoreMatching((v: Value) => mul(v, mul(theta, -1)), x, key(0), { kind: 'rademacher' })
    // d/dtheta of (-theta + theta^2 m2 / 2) = -1 + theta m2.
    expect(item(grad(f)(tensor(0.8)))).toBeCloseTo(-1 + 0.8 * m2, 12)
  })
})

describe('vector quantisation', () => {
  const book = tensor([
    [1, 0],
    [0, 1],
    [-1, 0],
  ])
  const z = tensor([
    [0.9, 0.1],
    [0.2, 1.3],
    [0.7, -0.2],
    [-1.2, 0.1],
  ])

  it('picks the nearest codes and splits the loss into the codebook and commitment terms', () => {
    const q = L.vectorQuantise(z, book, { beta: 0.25 })
    expect(toArray(q.indices)).toEqual([0, 1, 0, 2])
    for (const [row, code] of [
      [0, 0],
      [1, 1],
      [2, 0],
      [3, 2],
    ] as const)
      expect((toArray(q.quantised) as number[][])[row]?.map((v) => Math.round(v * 1e12) / 1e12)).toEqual(
        (toArray(book) as number[][])[code],
      ) // z + (e - z) is e up to rounding
    const sq = (0.1 ** 2 + 0.1 ** 2 + 0.2 ** 2 + 0.3 ** 2 + 0.3 ** 2 + 0.2 ** 2 + 0.2 ** 2 + 0.1 ** 2) / 8
    expect(item(q.codebookLoss)).toBeCloseTo(sq, 14)
    expect(item(q.commitmentLoss)).toBeCloseTo(sq, 14)
    expect(item(q.loss)).toBeCloseTo(1.25 * sq, 14)
    expect(toArray(q.usage)).toEqual([0.5, 0.25, 0.25])
    expect(item(q.perplexity)).toBeCloseTo(Math.exp(-(0.5 * Math.log(0.5) + 2 * 0.25 * Math.log(0.25))), 8)
  })

  it('sends the decoder gradient straight to the latents, the commitment gradient to the latents and the codebook gradient to the codes', () => {
    const through = grad((v: Value) => sum(L.vectorQuantise(v, book).quantised))(z)
    expect(toArray(through)).toEqual(
      toArray(
        tensor([
          [1, 1],
          [1, 1],
          [1, 1],
          [1, 1],
        ]),
      ),
    )
    const toCodes = grad((c: Value) => L.vectorQuantise(z, c, { beta: 0 }).loss)(book)
    // Code 1 is picked once, by z[1] = (0.2, 1.3): d/dc mean ||z - c||^2 = -2 (z - c) / 8.
    expect((toArray(toCodes) as number[][])[1]?.[1]).toBeCloseTo((-2 * 0.3) / 8, 14)
    const toLatents = grad((v: Value) => L.vectorQuantise(v, book, { beta: 1, ema: true }).loss)(z)
    expect((toArray(toLatents) as number[][])[0]?.[0]).toBeCloseTo((2 * -0.1) / 8, 14)
  })

  it('moves the EMA codebook to the means of the assigned latents', () => {
    let state: L.VqEmaState = { counts: tensor([1, 1, 1]), sums: book, codebook: book }
    const batch = tensor([
      [2, 0],
      [4, 0],
    ])
    for (let i = 0; i < 400; i++) state = L.vqEmaUpdate(state, batch, [0, 0], { decay: 0.9, eps: 1e-5 })
    expect((toArray(state.codebook) as number[][])[0]?.[0]).toBeCloseTo(3, 3)
  })
})
