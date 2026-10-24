import { describe, expect, it } from 'vitest'
import { expectMoments } from '../../../testing/sampling.ts'
import { key } from '../../random/index.ts'
import { exp, mul, neg, reshape, split, square, stack, sum, tensor, type Value, zeros } from '../../tensor/index.ts'
import { sampleChains } from '../chains.ts'
import { banana, nealsFunnel, numericTarget, target } from '../target.ts'
import { hmcKernel } from './hmc.ts'
import { riemannianHmcKernel, riemannianTrajectory } from './riemannian.ts'

/** The expected Fisher information of the 2-D funnel: diag(1/9 + 1/2, e^-v). */
const funnelMetric = (x: Value): Value => {
  const [v] = split(x, 2).map((t) => sum(t)) as [Value, Value]
  const a = tensor(1 / 9 + 1 / 2)
  const b = exp(neg(v))
  const z = zeros([])
  return reshape(stack([a, z, z, b]), [2, 2])
}

function energyErrors(
  t: ReturnType<typeof target>,
  opts: { readonly metric?: (x: Value) => Value },
  eps: number,
): number {
  const path = riemannianTrajectory(t, [0.4, 0.3], [0.5, -0.7], {
    stepSize: eps,
    numIntegrationSteps: Math.round(0.8 / eps),
    tolerance: 1e-13,
    fixedPointIterations: 40,
    ...opts,
  })
  const h0 = path[0]?.energy ?? 0
  let worst = 0
  for (const p of path) worst = Math.max(worst, Math.abs(p.energy - h0))
  return worst
}

describe('riemannianHmcKernel', () => {
  it('reduces to HMC with mass matrix G when the metric is constant (same key, same transition)', () => {
    const g = [
      [2, 0.6],
      [0.6, 1],
    ]
    const t = target((x) => mul(-0.5, sum(mul(square(x), [2, 1]))), 2)
    const r = riemannianHmcKernel(t, {
      stepSize: 0.2,
      numIntegrationSteps: 7,
      metric: () => tensor(g),
      tolerance: 1e-14,
    })
    const det = 2 - 0.36
    const h = hmcKernel(t, {
      stepSize: 0.2,
      numIntegrationSteps: 7,
      inverseMassMatrix: [
        [1 / det, -0.6 / det],
        [-0.6 / det, 2 / det],
      ],
    })
    const a = r.step(key(1), r.init([0.3, -0.4]))
    const b = h.step(key(1), h.init([0.3, -0.4]))
    expect(a.info.accepted).toBe(b.info.accepted)
    expect(a.info.acceptanceRate).toBeCloseTo(b.info.acceptanceRate, 10)
    for (let i = 0; i < 2; i++) expect(a.state.position[i]).toBeCloseTo(b.state.position[i] ?? 0, 10)
  })

  it('conserves energy to second order with the SoftAbs metric of a curved target', () => {
    const b = banana({ curvature: 0.4, scale: 1 })
    const ratio = energyErrors(b, {}, 0.1) / energyErrors(b, {}, 0.05)
    expect(ratio).toBeGreaterThan(3)
    expect(ratio).toBeLessThan(5)
  })

  it('conserves energy to second order with a given position-dependent metric', () => {
    const f = nealsFunnel(2)
    const ratio = energyErrors(f, { metric: funnelMetric }, 0.1) / energyErrors(f, { metric: funnelMetric }, 0.05)
    expect(ratio).toBeGreaterThan(3)
    expect(ratio).toBeLessThan(5)
  })

  it("samples Neal's funnel with the Fisher metric, neck included", () => {
    const f = nealsFunnel(2)
    const k = riemannianHmcKernel(f, {
      stepSize: 0.6,
      numIntegrationSteps: 4,
      metric: funnelMetric,
      fixedPointIterations: 6,
    })
    const c = sampleChains(k, {
      key: key(2),
      numChains: 2,
      numWarmup: 50,
      numDraws: 250,
      init: [
        [0, 0],
        [1, 0.5],
      ],
    })
    expectMoments(c, { mean: [0, 0], sd: [3] })
  })

  it('rejects as divergent where the density or the metric breaks, without throwing', () => {
    const broken = numericTarget(() => Number.NaN, 2, { grad: () => [Number.NaN, Number.NaN] })
    const k = riemannianHmcKernel(broken, {
      stepSize: 0.1,
      numIntegrationSteps: 3,
      metric: () =>
        tensor([
          [1, 0],
          [0, 1],
        ]),
    })
    const t = k.step(key(3), k.init([0, 0]))
    expect(t.info.divergent).toBe(true)
    expect(t.info.accepted).toBe(false)
    const indefinite = riemannianHmcKernel(
      target((x) => neg(sum(square(x))), 2),
      {
        stepSize: 0.1,
        numIntegrationSteps: 3,
        metric: () =>
          tensor([
            [1, 2],
            [2, 1],
          ]),
      },
    )
    expect(indefinite.step(key(4), indefinite.init([0.1, 0.1])).info.divergent).toBe(true)
  })

  it('validates its settings', () => {
    const t = target((x) => neg(sum(square(x))), 1)
    expect(() => riemannianHmcKernel(t, { stepSize: 0, numIntegrationSteps: 1 })).toThrow()
    expect(() => riemannianHmcKernel(t, { stepSize: 0.1, numIntegrationSteps: 0 })).toThrow()
    expect(() =>
      riemannianHmcKernel(
        numericTarget(() => 0, 1),
        { stepSize: 0.1, numIntegrationSteps: 1 },
      ),
    ).toThrow()
    expect(() => riemannianTrajectory(t, [0, 0], [0], { stepSize: 0.1, numIntegrationSteps: 1 })).toThrow()
  })
})
