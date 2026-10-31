import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { grad } from '../autodiff/index.ts'
import { key, normal, split } from '../random/index.ts'
import { add, item, matmul, mul, sub, sum, tensor, toArray, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

describe('IRM and V-REx penalties', () => {
  const s = tensor([0.4, -1.1, 2.0, 0.7, -0.3, 1.2])
  const y = tensor([1, 0, 1, 1, 0, 0])

  it('has the IRMv1 penalty equal to the squared derivative at scale one (closed form for squared error)', () => {
    const x = [3, 1, 0.5, -2]
    const t = [1, 1, 0, 0.5]
    const g = (2 * x.reduce((acc, v, i) => acc + (v - (t[i] ?? 0)) * v, 0)) / x.length
    expect(item(L.irmPenalty(x, t, { risk: L.mse }))).toBeCloseTo(g * g, 12)
  })

  it('has the unbiased estimate multiply the gradients of the two halves', () => {
    const x = [3, 1, 0.5, -2]
    const t = [1, 1, 0, 0.5]
    const half = (a: number[], b: number[]): number =>
      (2 * a.reduce((acc, v, i) => acc + (v - (b[i] ?? 0)) * v, 0)) / a.length
    expect(item(L.irmPenalty(x, t, { risk: L.mse, unbiased: true }))).toBeCloseTo(
      half(x.slice(0, 2), t.slice(0, 2)) * half(x.slice(2), t.slice(2)),
      12,
    )
  })

  it('is zero for scores that are already the optimal classifier of the environment', () => {
    // Squared error with y = s: the derivative at scale one is zero.
    expect(item(L.irmPenalty([1, 2, 3], [1, 2, 3], { risk: L.mse }))).toBe(0)
  })

  it('passes the conformance check, a gradient of a gradient', () => {
    expectConformant((a) => L.irmPenalty(a, y), [s])
    expectConformant(
      (a) =>
        L.irmObjective(
          [
            { scores: a, target: y },
            { scores: mul(a, 0.5), target: y },
          ],
          { penaltyWeight: 3 },
        ).total,
      [s],
    )
    expectConformant(
      (a) =>
        L.vrexObjective(
          [
            { scores: a, target: y },
            { scores: mul(a, 0.5), target: y },
          ],
          { penaltyWeight: 3 },
        ).total,
      [s],
    )
  })

  it('has V-REx the variance of the risks', () => {
    expect(item(L.vrexPenalty([1, 3]))).toBe(1)
    expect(
      item(
        L.vrexObjective(
          [
            { scores: [0], target: [1] },
            { scores: [0], target: [3] },
          ],
          { risk: L.mse, penaltyWeight: 2 },
        ).total,
      ),
    ).toBe(5 + 2 * 16)
  })

  it('drops a spurious feature that a pooled fit relies on (planted environments)', () => {
    // y = x1 + 0.5 noise (stable); x2 = y + sigma_e noise, with sigma 0.1 in one environment and 1 in the other: x2 is
    // informative in both but its relation to y changes. The invariant solution is w = (1, 0).
    const sigmas = [0.1, 1.0]
    const keys = split(key(7), 6)
    const envs = sigmas.map((sigma, e) => {
      const n = 500
      const x1 = normal(keys[3 * e] as ReturnType<typeof key>, [n])
      const eta = normal(keys[3 * e + 1] as ReturnType<typeof key>, [n])
      const zeta = normal(keys[3 * e + 2] as ReturnType<typeof key>, [n])
      const target = add(x1, mul(eta, 0.5))
      const x2 = add(target, mul(zeta, sigma))
      const features = tensor((toArray(x1) as number[]).map((v, i) => [v, (toArray(x2) as number[])[i] ?? 0]))
      return { features, target }
    })
    const fit = (objective: (w: Value) => Value, rate: number, steps: number): number[] => {
      let w: Value = tensor([0, 0])
      for (let i = 0; i < steps; i++) w = sub(w, mul(grad(objective)(w), rate))
      return toArray(w) as number[]
    }
    const batches = (w: Value): { scores: Value; target: Value }[] =>
      envs.map((e) => ({ scores: matmul(e.features, w), target: e.target }))
    const erm = fit((w) => L.vrexObjective(batches(w), { risk: L.mse, penaltyWeight: 0 }).total, 0.1, 400)
    const vrex = fit((w) => L.vrexObjective(batches(w), { risk: L.mse, penaltyWeight: 300 }).total, 0.01, 1500)
    expect(erm[1]).toBeGreaterThan(0.25)
    expect(Math.abs(vrex[1] ?? 1)).toBeLessThan(0.5 * (erm[1] ?? 0))
    // The IRM penalty is near zero in every environment at the invariant solution (1, 0) and large at the pooled fit.
    const penaltyAt = (w: number[]): number => item(L.irmObjective(batches(tensor(w)), { risk: L.mse }).penalty)
    expect(penaltyAt([1, 0])).toBeLessThan(0.02)
    expect(penaltyAt(erm)).toBeGreaterThan(5 * penaltyAt([1, 0]))
    expect(vrex[0]).toBeGreaterThan(erm[0] ?? 0)
  })
})

describe('feature and gradient alignment', () => {
  it('has CORAL the closed form for covariances that differ by a scale', () => {
    // Source variance 1 and target variance 0.25 in one dimension: |1 - 0.25|^2 / 4 = 0.140625 (population 1 over n-1 = 1... computed from the data).
    const source = [[1], [-1]]
    const target = [[0.5], [-0.5]]
    expect(item(L.coral(source, target))).toBeCloseTo((2 - 0.5) ** 2 / 4, 14)
    expect(item(L.coral(source, source))).toBe(0)
    expect(item(L.coral(source, [[2], [0]], { matchMean: true }))).toBeCloseTo((2 - 2) ** 2 / 4 + 1, 14)
  })

  it('has the MMD penalty zero for equal feature distributions and positive for shifted ones', () => {
    const a = [[0], [1], [2]]
    expect(item(L.mmdPenalty([a, a], { bandwidths: [1] }))).toBeCloseTo(0, 14)
    expect(item(L.mmdPenalty([a, [[3], [4], [5]]], { bandwidths: [1] }))).toBeGreaterThan(0.1)
  })

  it('aligns gradients: cosine zero for parallel, two for opposite, and the Fish inner product', () => {
    expect(
      item(
        L.gradientAlignment([
          [1, 2],
          [2, 4],
        ]),
      ),
    ).toBeCloseTo(0, 12)
    expect(
      item(
        L.gradientAlignment([
          [1, 2],
          [-1, -2],
        ]),
      ),
    ).toBeCloseTo(2, 12)
    expect(
      item(
        L.gradientAlignment(
          [
            [1, 2],
            [3, 4],
          ],
          { kind: 'inner' },
        ),
      ),
    ).toBe(-11)
    expect(
      item(
        L.gradientAlignment([
          [1, 0],
          [0, 1],
        ]),
      ),
    ).toBe(1)
  })

  it('takes per-sample gradients and the Fishr penalty of their variances', () => {
    const risk = (w: Value, x: Value, t: Value): Value => sum(mul(sub(mul(x, w), t), sub(mul(x, w), t)))
    expect(toArray(L.perSampleGradients(risk, [1], [[2], [3]], [[1], [1]]))).toEqual([[4], [12]])
    expect(
      item(
        L.fishrPenalty([
          [[1], [-1]],
          [[2], [-2]],
        ]),
      ),
    ).toBeCloseTo(2.25, 14)
    expect(
      item(
        L.fishrPenalty([
          [[1], [-1]],
          [[-1], [1]],
        ]),
      ),
    ).toBe(0)
    expectConformant(
      (w) =>
        L.fishrPenalty([
          L.perSampleGradients(risk, w, [[2], [3], [1]], [[1], [1], [0]]),
          L.perSampleGradients(risk, w, [[1], [-2]], [[0], [1]]),
        ]),
      [tensor([0.7])],
    )
    expectConformant(
      (w) =>
        L.gradientAlignment([
          L.environmentGradient((p) => risk(p, tensor([[2]]), tensor([[1]])), w),
          L.environmentGradient((p) => risk(p, tensor([[1]]), tensor([[3]])), w),
        ]),
      [tensor([0.4])],
    )
  })
})
