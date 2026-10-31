import { describe, expect, it } from 'vitest'
import { fromStorage, item, toFlat } from '../tensor/index.ts'
import { iterate, runAlgorithm } from '../trace/index.ts'
import {
  agglomerativeBottleneckAlgorithm,
  type BottleneckState,
  bottleneckCriticalBeta,
  bottleneckCurveAlgorithm,
  informationBottleneckAlgorithm,
  mutualInformation,
  plantedBottleneck,
} from './index.ts'

const planted = plantedBottleneck({ groups: 3, perGroup: 4, signal: 0.8 })

/** I(X; T) and I(T; Y) in bits recomputed from the encoder with the library's mutual information of joint tables. */
function informations(
  joint: ArrayLike<number>,
  nx: number,
  ny: number,
  encoder: ArrayLike<number>,
  nt: number,
): [number, number] {
  const px = Array.from({ length: nx }, (_, x) =>
    Array.from({ length: ny }, (_, y) => joint[x * ny + y] ?? 0).reduce((a, b) => a + b, 0),
  )
  const xt = new Float64Array(nx * nt)
  const ty = new Float64Array(nt * ny)
  for (let x = 0; x < nx; x++) {
    for (let t = 0; t < nt; t++) {
      xt[x * nt + t] = (px[x] ?? 0) * (encoder[x * nt + t] ?? 0)
      for (let y = 0; y < ny; y++)
        ty[t * ny + y] = (ty[t * ny + y] ?? 0) + (encoder[x * nt + t] ?? 0) * (joint[x * ny + y] ?? 0)
    }
  }
  return [
    item(mutualInformation(fromStorage(xt, [nx, nt], 'float64'), { base: 2 })),
    item(mutualInformation(fromStorage(ty, [nt, ny], 'float64'), { base: 2 })),
  ]
}

describe('the planted joint', () => {
  it('has the truth it claims', () => {
    expect(planted.groups).toEqual([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    expect(planted.groupEntropy).toBeCloseTo(Math.log2(3), 14)
    expect(item(mutualInformation(planted.joint, { base: 2 }))).toBeCloseTo(planted.mutualInformation, 12)
    expect(toFlat(planted.joint).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14)
    const exact = plantedBottleneck({ groups: 2, perGroup: 3, signal: 1 })
    expect(exact.mutualInformation).toBeCloseTo(1, 14)
    expect(() => plantedBottleneck({ groups: 1 })).toThrow(/two groups/)
    expect(() => plantedBottleneck({ signal: 2 })).toThrow(/signal/)
  })
})

describe('the critical beta', () => {
  it('has the closed form 1 / (1 - 2e)^2 for a symmetric binary channel', () => {
    const e = 0.1
    const joint = [
      [0.5 * (1 - e), 0.5 * e],
      [0.5 * e, 0.5 * (1 - e)],
    ]
    expect(bottleneckCriticalBeta(joint)).toBeCloseTo(1 / (1 - 2 * e) ** 2, 10)
    expect(
      bottleneckCriticalBeta([
        [0.25, 0.25],
        [0.25, 0.25],
      ]),
    ).toBe(Number.POSITIVE_INFINITY)
    expect(
      bottleneckCriticalBeta([
        [0.5, 0],
        [0, 0.5],
      ]),
    ).toBeCloseTo(1, 10)
  })

  it('splits the solution: below it the representation is empty, above it it is not', () => {
    const e = 0.1
    const joint = [
      [0.5 * (1 - e), 0.5 * e],
      [0.5 * e, 0.5 * (1 - e)],
    ]
    const bc = bottleneckCriticalBeta(joint)
    const run = (beta: number): BottleneckState =>
      runAlgorithm(informationBottleneckAlgorithm(joint, { beta, clusters: 2 }), { seed: 1, maxSteps: 30000 }).state
    expect(run(0.8 * bc).compression).toBeLessThan(1e-6)
    expect(run(0.8 * bc).relevance).toBeLessThan(1e-6)
    expect(run(1.5 * bc).compression).toBeGreaterThan(0.05)
    expect(run(1.5 * bc).relevance).toBeGreaterThan(0.01)
  })

  it('does the same for the planted joint', () => {
    const bc = bottleneckCriticalBeta(planted.joint)
    expect(bc).toBeGreaterThan(1)
    expect(bc).toBeLessThan(5)
    const below = runAlgorithm(informationBottleneckAlgorithm(planted.joint, { beta: 0.6 * bc }), {
      seed: 2,
      maxSteps: 30000,
    }).state
    expect(below.compression).toBeLessThan(1e-6)
  })
})

describe('the iteration', () => {
  it('lowers the bottleneck functional every iteration', () => {
    const beta = 4
    const states = Array.from(
      iterate(informationBottleneckAlgorithm(planted.joint, { beta, clusters: 4 }), { seed: 3, maxSteps: 120 }),
      (x) => x.state,
    )
    const objective = states.map((s) => s.compression - beta * s.relevance)
    for (let i = 1; i < objective.length; i++)
      expect(objective[i] as number).toBeLessThanOrEqual((objective[i - 1] as number) + 1e-9)
    expect((objective[0] as number) - (objective[objective.length - 1] as number)).toBeGreaterThan(0.01)
  })

  it('reports the informations of its encoder, as the library computes them', () => {
    const run = runAlgorithm(informationBottleneckAlgorithm(planted.joint, { beta: 5, clusters: 4 }), {
      seed: 4,
      maxSteps: 50,
    })
    const [ixt, ity] = informations(toFlat(planted.joint), 12, 3, run.state.encoder, 4)
    expect(run.state.compression).toBeCloseTo(ixt, 9)
    expect(run.state.relevance).toBeCloseTo(ity, 9)
  })

  it('obeys the data-processing bounds at every beta', () => {
    for (const beta of [0.5, 2, 6, 20, 100]) {
      const s = runAlgorithm(informationBottleneckAlgorithm(planted.joint, { beta, clusters: 5 }), {
        seed: 5,
        maxSteps: 3000,
      }).state
      expect(s.relevance).toBeLessThanOrEqual(planted.mutualInformation + 1e-9)
      expect(s.relevance).toBeLessThanOrEqual(s.compression + 1e-9)
      expect(s.compression).toBeLessThanOrEqual(Math.log2(12) + 1e-9)
    }
  })

  it('with a large beta clusters the values of X by their group and keeps all the relevance', () => {
    const s = runAlgorithm(informationBottleneckAlgorithm(planted.joint, { beta: 300, clusters: 3 }), {
      seed: 6,
      maxSteps: 5000,
    }).state
    expect(s.relevance).toBeCloseTo(planted.mutualInformation, 3)
    expect(s.compression).toBeCloseTo(planted.groupEntropy, 2)
    const best = (x: number): number => {
      let t = 0
      for (let k = 1; k < 3; k++) if ((s.encoder[x * 3 + k] ?? 0) > (s.encoder[x * 3 + t] ?? 0)) t = k
      return t
    }
    const assignment = Array.from({ length: 12 }, (_, x) => best(x))
    // Same group, same cluster; different groups, different clusters.
    for (let a = 0; a < 12; a++)
      for (let b = 0; b < 12; b++) expect(assignment[a] === assignment[b]).toBe(planted.groups[a] === planted.groups[b])
  })

  it('resumes exactly', () => {
    const alg = informationBottleneckAlgorithm(planted.joint, { beta: 5 })
    const straight = runAlgorithm(alg, { seed: 7, maxSteps: 40 })
    const early = runAlgorithm(alg, { seed: 7, maxSteps: 15 })
    const resumed = runAlgorithm(alg, {
      seed: 7,
      maxSteps: 25,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(Array.from(resumed.state.encoder)).toEqual(Array.from(straight.state.encoder))
  })

  it('rejects a negative beta, one cluster and a vector', () => {
    expect(() => informationBottleneckAlgorithm(planted.joint, { beta: -1 })).toThrow(/beta/)
    expect(() => informationBottleneckAlgorithm(planted.joint, { clusters: 1 })).toThrow(/two clusters/)
    expect(() => informationBottleneckAlgorithm([0.5, 0.5])).toThrow(/matrix/)
  })
})

describe('the information curve', () => {
  const betas = [0.5, 1, 2, 4, 8, 16, 64, 256]

  it('rises from the origin to the corner (H(G), I(X; Y)) as beta grows', () => {
    const run = runAlgorithm(bottleneckCurveAlgorithm(planted.joint, betas, { clusters: 4 }), { seed: 8, maxSteps: 20 })
    const pts = run.state.points
    expect(run.steps).toBe(betas.length)
    expect(pts[0]).toBeLessThan(1e-6)
    expect(pts[1]).toBeLessThan(1e-6)
    for (let k = 1; k < betas.length; k++) {
      expect(pts[2 * k + 1] as number).toBeGreaterThanOrEqual((pts[2 * k - 1] as number) - 1e-6)
      expect(pts[2 * k] as number).toBeGreaterThanOrEqual((pts[2 * k - 2] as number) - 1e-6)
    }
    expect(pts[2 * betas.length - 1]).toBeCloseTo(planted.mutualInformation, 3)
    expect(pts[2 * betas.length - 2]).toBeCloseTo(planted.groupEntropy, 1)
    expect(run.state.effectiveClusters).toBeGreaterThan(2.9)
    expect(run.state.effectiveClusters).toBeLessThan(4.1)
  })

  it('with as many clusters as groups it ends on exactly three', () => {
    const run = runAlgorithm(bottleneckCurveAlgorithm(planted.joint, betas, { clusters: 3 }), { seed: 8, maxSteps: 20 })
    expect(run.state.effectiveClusters).toBeCloseTo(3, 1)
  })

  it('merges clusters as beta falls', () => {
    const run = runAlgorithm(bottleneckCurveAlgorithm(planted.joint, [...betas].reverse(), { clusters: 4 }), {
      seed: 8,
      maxSteps: 20,
    })
    expect(run.state.points[run.state.points.length - 1]).toBeLessThan(1e-6)
    expect(run.state.effectiveClusters).toBeLessThan(1.01)
  })

  it('resumes exactly', () => {
    const alg = bottleneckCurveAlgorithm(planted.joint, betas, { clusters: 4 })
    const straight = runAlgorithm(alg, { seed: 9, maxSteps: 20 })
    const early = runAlgorithm(alg, { seed: 9, maxSteps: 3 })
    const resumed = runAlgorithm(alg, {
      seed: 9,
      maxSteps: 20,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(resumed.state.points).toEqual(straight.state.points)
    expect(() => bottleneckCurveAlgorithm(planted.joint, [])).toThrow(/betas/)
  })
})

describe('the agglomerative bottleneck', () => {
  it('each merge costs exactly the relevance it takes away, and merging within a group is free', () => {
    const states = Array.from(
      iterate(agglomerativeBottleneckAlgorithm(planted.joint), { seed: 0, maxSteps: 20 }),
      (x) => x.state,
    )
    expect(states).toHaveLength(12)
    expect(states[0]?.relevance).toBeCloseTo(planted.mutualInformation, 12)
    for (let k = 1; k < states.length; k++) {
      expect((states[k - 1]?.relevance ?? 0) - (states[k]?.relevance ?? 0)).toBeCloseTo(
        states[k]?.last?.cost ?? Number.NaN,
        12,
      )
    }
    // The first 9 merges happen inside the groups and cost nothing; the 10th crosses groups.
    for (let k = 1; k <= 9; k++) expect(states[k]?.last?.cost).toBeCloseTo(0, 12)
    expect(states[9]?.relevance).toBeCloseTo(planted.mutualInformation, 12)
    expect(states[9]?.compression).toBeCloseTo(planted.groupEntropy, 12)
    expect(states[10]?.last?.cost).toBeGreaterThan(0.01)
    expect(states[11]?.relevance).toBeCloseTo(0, 12)
    expect(states[11]?.compression).toBeCloseTo(0, 12)
    const clusters = (states[9]?.clusters ?? []).map((c) => [...c])
    expect(clusters).toEqual([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
    ])
  })

  it('matches the library mutual information of the clustered joint for any joint', () => {
    const joint = [
      [0.1, 0.05, 0.02],
      [0.04, 0.2, 0.01],
      [0.03, 0.02, 0.15],
      [0.08, 0.06, 0.24],
    ]
    const states = Array.from(
      iterate(agglomerativeBottleneckAlgorithm(joint), { seed: 0, maxSteps: 10 }),
      (x) => x.state,
    )
    for (const s of states) {
      const rows = s.clusters.map((c) => [0, 1, 2].map((y) => c.reduce((a, x) => a + (joint[x]?.[y] ?? 0), 0)))
      expect(s.relevance).toBeCloseTo(item(mutualInformation(rows, { base: 2 })), 12)
    }
    expect(() => agglomerativeBottleneckAlgorithm([1, 2])).toThrow(/matrix/)
  })
})
