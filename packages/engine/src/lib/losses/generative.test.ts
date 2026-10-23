import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { key } from '../random/index.ts'
import { item, matmul, tanh, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const real = tensor([0.3, -1.2, 2, 0.7])
const fake = tensor([0.5, 0.15, -0.7, 1.9])

describe('generative-model losses', () => {
  for (const objective of ['minimax', 'non-saturating', 'wasserstein', 'hinge', 'least-squares'] as const) {
    it(`differentiates the ${objective} objectives`, () => {
      expectConformant((r: Value, f: Value) => L.discriminatorLoss(r, f, { objective }), [real, fake])
      expectConformant((f: Value) => L.generatorLoss(f, { objective }), [fake])
    })
  }

  it('differentiates the gradient penalties in the critic parameters (second order)', () => {
    const xr = tensor([
      [0.2, -0.5],
      [1, 0.3],
      [-0.4, 0.8],
    ])
    const xf = tensor([
      [0.9, 0.1],
      [-1, 0.2],
      [0.3, -0.6],
    ])
    const w = tensor([
      [0.5, -1, 0.3],
      [0.2, 0.8, -0.4],
    ])
    const v = tensor([1, -0.5, 0.7])
    const critic = (ww: Value) => (x: Value) => matmul(tanh(matmul(x, ww)), v)
    expectConformant((ww: Value) => L.gradientPenalty(critic(ww), xr, xf, { alpha: [0.2, 0.5, 0.9] }), [w])
    expectConformant((ww: Value) => L.r1Penalty(critic(ww), xr), [w])
    // With a key, the interpolation is reproducible.
    const a = item(L.gradientPenalty(critic(w), xr, xf, { key: key(3) }))
    expect(item(L.gradientPenalty(critic(w), xr, xf, { key: key(3) }))).toBe(a)
    expect(() => L.gradientPenalty(critic(w), xr, xf)).toThrow()
  })

  it('differentiates the ELBO terms', () => {
    const mu = tensor([
      [0.3, -1],
      [0.5, 0.2],
    ])
    const s = tensor([
      [0.1, -0.4],
      [0.6, 0],
    ])
    expectConformant((m: Value, ls: Value) => L.gaussianKl(m, ls), [mu, s])
    expectConformant(
      (l: Value) =>
        L.bernoulliLogLikelihood(l, [
          [1, 0],
          [0.3, 1],
        ]),
      [mu],
    )
    expectConformant(
      (m: Value, ls: Value) =>
        L.gaussianLogLikelihood(
          m,
          [
            [1, 0],
            [0.3, 1],
          ],
          { logVariance: ls },
        ),
      [mu, s],
    )
    expectConformant((w: Value) => L.importanceWeightedBound(w), [mu])
    // One importance sample is the ELBO.
    expect(item(L.importanceWeightedBound([[-2.5], [-1]]))).toBeCloseTo(1.75, 15)
  })

  it('differentiates the score, diffusion and flow objectives', () => {
    const pred = tensor([
      [0.3, -1],
      [0.5, 0.2],
    ])
    expectConformant(
      (p: Value) =>
        L.denoisingScoreMatching(
          p,
          [
            [1, -1],
            [0.5, 2],
          ],
          [0.5, 2],
        ),
      [pred],
    )
    expectConformant(
      (p: Value) =>
        L.diffusionLoss(
          p,
          [
            [1, 0],
            [0, 1],
          ],
          [
            [0.2, 0.3],
            [1, -1],
          ],
          0.8,
          0.6,
          { parameterisation: 'v' },
        ),
      [pred],
    )
    expectConformant(
      (p: Value) =>
        L.flowMatching(
          p,
          [
            [1, 0],
            [0, 1],
          ],
          [
            [0.2, 0.3],
            [1, -1],
          ],
        ),
      [pred],
    )
    // The velocity of the straight path is constant along it.
    const path = L.flowMatchingPath([[1, 2]], [[3, -1]], [0.7])
    expect(item(L.flowMatching(path.velocity, [[1, 2]], [[3, -1]]))).toBe(0)
  })
})
