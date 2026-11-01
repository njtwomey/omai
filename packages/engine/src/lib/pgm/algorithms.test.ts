import { describe, expect, it } from 'vitest'
import { foldIn, key, normal, uniform } from '../random/index.ts'
import { type Algorithm, policies, recorder, runAlgorithm } from '../trace/index.ts'
import { directLingam, directLingamAlgorithm, exogeneityMeasures, lingamContract } from './causal/discovery.ts'
import { fci, fciAlgorithm, fciContract } from './causal/fci.ts'
import { dag, dagEdges } from './graphs.ts'
import {
  effectiveSampleSize,
  importanceContract,
  importanceSamplers,
  likelihoodWeighting,
  likelihoodWeightingAlgorithm,
  rejectionSample,
  rejectionSampleAlgorithm,
} from './inference/sampling.ts'
import {
  cpdag,
  cpdagAlgorithm,
  immoralities,
  isMarkovEquivalent,
  meekContract,
  meekRules,
  meekRulesAlgorithm,
  pdag,
  pdagToDag,
  pdagToDagAlgorithm,
} from './learning/equivalence.ts'
import { chowLiu, chowLiuAlgorithm, chowLiuContract, mutualInformationEstimators } from './learning/structure.ts'
import { ancestralSample, bayesNet, forwardSample } from './models/bayes-net.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

/** Run to the end, and again from a JSON checkpoint cut at `cut` steps; both must end the same. */
function expectResumes<S>(alg: Algorithm<S>, cut: number, project: (s: S) => unknown): S {
  const whole = runAlgorithm(alg, { maxSteps: 10_000 })
  const half = runAlgorithm(alg, { maxSteps: cut })
  const stored = viaJson(alg.serialise?.(half.state) ?? null)
  const rest = runAlgorithm(alg, { maxSteps: 10_000, start: { state: stored, step: half.steps, serialised: true } })
  expect(project(rest.state)).toEqual(project(whole.state))
  expect(rest.status).toBe(whole.status)
  expect(rest.steps).toBe(whole.steps)
  return whole.state
}

/** The values logged to scalar and tensor paths of a run. */
function logged<S>(alg: Algorithm<S>, maxSteps = 10_000): Map<string, unknown[]> {
  const rec = recorder()
  runAlgorithm(alg, {
    maxSteps,
    tracer: rec,
    policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
  })
  const out = new Map<string, unknown[]>()
  for (const e of rec.events) {
    if (e.type === 'metric') out.set(e.name, [...(out.get(e.name) ?? []), e.value])
    if (e.type === 'tensor') out.set(e.name, [...(out.get(e.name) ?? []), Array.from(e.values as ArrayLike<number>)])
  }
  return out
}

const net = bayesNet([
  { name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] },
  { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.2, 0.8] },
])

describe('rejection sampling and likelihood weighting algorithms', () => {
  it('forwardSample is one draw of ancestralSample', () => {
    const k = key(5)
    const all = ancestralSample(net, k, 6)
    expect(all).toEqual(Array.from({ length: 6 }, (_, i) => forwardSample(net, foldIn(k, i))))
  })

  it('rejection keeps exactly the samples that agree with the evidence, one draw per step', () => {
    const run = runAlgorithm(rejectionSampleAlgorithm(net, key(1), 300, { b: '1' }), { maxSteps: 400 })
    expect(run.steps).toBe(300)
    expect(run.state.drawn).toBe(300)
    expect(run.state.samples.every((s) => s['b'] === 1)).toBe(true)
    const old = rejectionSample(net, key(1), 300, { b: '1' })
    expect(old.samples).toEqual(run.state.samples)
    expect(old.acceptance).toBe(run.state.samples.length / 300)
    // P(b = 1) = 0.5 * 0.1 + 0.5 * 0.8 = 0.45.
    expect(Math.abs(old.acceptance - 0.45)).toBeLessThan(0.1)
    expect(rejectionSample(net, key(1), 0).acceptance).toBe(0)
  })

  it('likelihood weighting gives the weights and effective sample size of the function', () => {
    const run = runAlgorithm(likelihoodWeightingAlgorithm(net, key(2), 400, { b: '1' }), { maxSteps: 400 })
    const old = likelihoodWeighting(net, key(2), 400, { b: '1' })
    expect(run.state.samples).toEqual(old.samples)
    expect(run.state.logWeights).toEqual(old.logWeights)
    expect(old.effectiveSampleSize).toBeCloseTo(effectiveSampleSize(run.state.logWeights), 9)
    expect(old.effectiveSampleSize).toBeLessThanOrEqual(400 + 1e-9)
    const l = logged(likelihoodWeightingAlgorithm(net, key(2), 400, { b: '1' }))
    const estimate = (l.get('is/estimate') as number[]).at(-1) as number
    expect(Math.abs(estimate - 0.45)).toBeLessThan(0.05)
    expect(l.get('is/weight')).toHaveLength(400)
    const ess = l.get('is/ess') as number[]
    expect(ess.at(-1)).toBeCloseTo(old.effectiveSampleSize, 9)
    expect(Object.keys(importanceContract)).toHaveLength(4)
  })

  it('rejected samples are logged with a weight of minus infinity and the estimate is the acceptance rate', () => {
    const l = logged(rejectionSampleAlgorithm(net, key(3), 50, { b: '1' }))
    const accepted = l.get('is/accepted') as number[]
    const weight = l.get('is/weight') as number[]
    for (const [i, a] of accepted.entries()) expect(weight[i]).toBe(a === 1 ? 0 : Number.NEGATIVE_INFINITY)
    expect((l.get('is/estimate') as number[]).at(-1)).toBeCloseTo(accepted.reduce((p, c) => p + c, 0) / 50, 12)
  })

  it('both resume from a JSON checkpoint, infinite log weights included', () => {
    const impossible = bayesNet([
      { name: 'a', states: ['0', '1'], parents: [], cpt: [1, 0] },
      { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] },
    ])
    for (const alg of [
      rejectionSampleAlgorithm(net, key(4), 40, { b: '1' }),
      likelihoodWeightingAlgorithm(net, key(4), 40, { b: '1' }),
      likelihoodWeightingAlgorithm(impossible, key(4), 10, { a: '1' }),
    ]) {
      expectResumes(alg, 7, (s) => ({ s: s.samples, w: s.logWeights.map(String), n: s.drawn }))
    }
  })

  it('the sampler table builds both', () => {
    expect(importanceSamplers.names).toEqual(['rejection', 'likelihood-weighting'])
    expect(importanceSamplers.where((p) => p.discards)).toEqual(['rejection'])
    expect(importanceSamplers.make({ kind: 'rejection' })(net, key(0), 3).name).toBe('pgm/rejection-sampling')
  })
})

describe('chowLiuAlgorithm', () => {
  const data = Array.from({ length: 120 }, (_, i) => ({
    a: i % 2,
    b: i % 2,
    c: (i % 2) ^ (i % 3 === 0 ? 1 : 0),
    d: (i * 7) % 5 < 2 ? 1 : 0,
  }))

  it('adds one edge per step, in non-increasing mutual information, to the tree chowLiu returns', () => {
    const run = runAlgorithm(chowLiuAlgorithm(data), { maxSteps: 20 })
    expect(run.status).toBe('converged')
    expect(run.state.edges).toHaveLength(3)
    expect(run.steps).toBe(3)
    const w = run.state.weights
    expect(w.every((x, i) => i === 0 || x <= (w[i - 1] ?? 0) + 1e-12)).toBe(true)
    const tree = chowLiu(data, { root: 'a' })
    expect(dagEdges(tree)).toHaveLength(3)
    // Each undirected edge of the run is an edge of the oriented tree, in one direction.
    const oriented = new Set(dagEdges(tree).map(([x, y]) => [x, y].sort().join('-')))
    for (const [x, y] of run.state.edges) expect(oriented.has([x, y].sort().join('-'))).toBe(true)
  })

  it('logs the edge, its information, the cycle-closing pairs skipped and the components', () => {
    const l = logged(chowLiuAlgorithm(data))
    expect(l.get('cl/components') as number[]).toEqual([3, 2, 1])
    expect(l.get('cl/edge')).toHaveLength(3)
    const skipped = l.get('cl/skipped') as number[]
    expect(skipped.every((x) => x >= 0)).toBe(true)
    expect(Object.keys(chowLiuContract)).toHaveLength(4)
  })

  it('resumes from a JSON checkpoint, and a class variable gives the augmented naive Bayes edges', () => {
    expectResumes(chowLiuAlgorithm(data), 1, (s) => ({ e: s.edges, w: s.weights, c: s.cursor }))
    const tan = chowLiu(data, { classVariable: 'd', root: 'a' })
    expect(dagEdges(tan).filter(([x]) => x === 'd')).toHaveLength(3)
    expect(() => chowLiuAlgorithm(data, { root: 'zzz' })).toThrow(/unknown root/)
  })

  it('the Miller-Madow correction lowers the estimate for independent columns and keeps the strong edge', () => {
    expect(mutualInformationEstimators.names).toEqual(['plug-in', 'miller-madow'])
    const independent = Array.from({ length: 40 }, (_, i) => ({ x: i % 2, y: Math.floor(i / 2) % 2 }))
    const plug = runAlgorithm(chowLiuAlgorithm(independent), { maxSteps: 5 }).state
    const mm = runAlgorithm(chowLiuAlgorithm(independent, { estimator: 'miller-madow' }), { maxSteps: 5 }).state
    expect(plug.weights[0]).toBeCloseTo(0, 12)
    expect(mm.weights[0]).toBeCloseTo(-1 / 80, 12)
    const strong = runAlgorithm(chowLiuAlgorithm(data, { estimator: 'miller-madow' }), { maxSteps: 20 }).state
    expect(strong.edges[0]).toEqual(runAlgorithm(chowLiuAlgorithm(data), { maxSteps: 20 }).state.edges[0])
  })
})

describe('directLingamAlgorithm', () => {
  const u = uniform(key(0), [300, 3]).data
  const rows = Array.from({ length: 300 }, (_, i) => {
    const x = (u[3 * i] ?? 0) - 0.5
    const y = 2 * x + ((u[3 * i + 1] ?? 0) - 0.5)
    const z = -1.5 * y + ((u[3 * i + 2] ?? 0) - 0.5)
    return { z, y, x }
  })

  it('places the causes first, one variable per step, and directLingam is it plus the effects', () => {
    const run = runAlgorithm(directLingamAlgorithm(rows), { maxSteps: 10 })
    expect(run.steps).toBe(3)
    expect(run.status).toBe('converged')
    expect(run.state.order).toEqual([2, 1, 0])
    const result = directLingam(rows)
    expect(result.order).toEqual(['x', 'y', 'z'])
    expect(result.adjacency[1]?.[2]).toBeCloseTo(2, 0)
    expect(result.adjacency[0]?.[1]).toBeCloseTo(-1.5, 0)
  })

  it('logs scores of the variables in play and the order growing', () => {
    const l = logged(directLingamAlgorithm(rows))
    expect(l.get('lingam/exogenous')).toEqual([2, 1, 0])
    const order = l.get('lingam/order') as number[][]
    expect(order.map((o) => o.length)).toEqual([1, 2, 3])
    const scores = (l.get('lingam/scores') as number[][])[0] as number[]
    expect(scores).toHaveLength(3)
    expect(Math.max(...scores)).toBe(scores[2])
    expect(Object.keys(lingamContract)).toHaveLength(3)
  })

  it('resumes from a JSON checkpoint with infinite scores, and the other measure finds the order too', () => {
    expectResumes(directLingamAlgorithm(rows), 1, (s) => ({
      o: s.order,
      r: s.remaining,
      c: s.scores.map(String),
      w: s.work.map((c) => Array.from(c, (v) => Math.round(v * 1e9))),
    }))
    const other = runAlgorithm(directLingamAlgorithm(rows, { measure: 'nonlinear-correlation' }), { maxSteps: 10 })
    expect(other.state.order[0]).toBe(2)
    expect(exogeneityMeasures.names).toEqual(['pwling', 'nonlinear-correlation'])
  })
})

describe('Meek rules, DAG extension and CPDAG algorithms', () => {
  // a -> b - c - d, with b - d: R1 orients b -> c, then the rest follow.
  const start = pdag(
    [],
    [['a', 'b']],
    [
      ['b', 'c'],
      ['c', 'd'],
    ],
  )

  it('one rule sweep per step, to the pdag meekRules returns', () => {
    const run = runAlgorithm(meekRulesAlgorithm(start), { maxSteps: 50 })
    expect(run.status).toBe('converged')
    expect(run.state.finished).toBe(true)
    expect(run.state.directed).toEqual([
      ['a', 'b'],
      ['b', 'c'],
      ['c', 'd'],
    ])
    expect(run.state.undirected).toEqual([])
    const old = meekRules(start)
    expect(old.directed).toEqual([
      ['a', 'b'],
      ['b', 'c'],
      ['c', 'd'],
    ])
    // The last cycle of three rules orients nothing.
    expect(run.steps % 3).toBe(0)
    expect(run.state.sweeps).toBe(run.steps)
  })

  it('logs the rule swept and the edges it oriented; the undirected count never grows', () => {
    const l = logged(meekRulesAlgorithm(start))
    const rules = l.get('meek/rule') as number[]
    expect(rules.slice(0, 3)).toEqual([1, 2, 3])
    const undirected = l.get('meek/undirected') as number[]
    expect(undirected.every((u, i) => i === 0 || u <= (undirected[i - 1] ?? 0))).toBe(true)
    expect(undirected.at(-1)).toBe(0)
    const oriented = l.get('meek/oriented') as number[][]
    // The first sweep of R1 orients b -> c and, seeing it, c -> d.
    expect(oriented[0]).toEqual([1, 2, 2, 3])
    expect(Object.keys(meekContract)).toHaveLength(4)
  })

  it('rule 4 is swept only when asked for', () => {
    expect(runAlgorithm(meekRulesAlgorithm(start, { r4: true }), { maxSteps: 50 }).steps % 4).toBe(0)
  })

  it('resumes from a JSON checkpoint at every cut', () => {
    for (const cut of [1, 2, 4]) expectResumes(meekRulesAlgorithm(start), cut, (s) => s)
  })

  it('cpdagAlgorithm reaches the cpdag of a DAG: v-structures oriented, a chain left undirected', () => {
    const collider = dag([
      ['a', 'c'],
      ['b', 'c'],
      ['c', 'd'],
    ])
    const run = runAlgorithm(cpdagAlgorithm(collider), { maxSteps: 50 })
    const old = cpdag(collider)
    expect(run.state.directed).toEqual(old.directed)
    expect(old.directed).toEqual([
      ['a', 'c'],
      ['b', 'c'],
      ['c', 'd'],
    ])
    const chain = runAlgorithm(
      cpdagAlgorithm(
        dag([
          ['a', 'b'],
          ['b', 'c'],
        ]),
      ),
      { maxSteps: 50 },
    )
    expect(chain.state.directed).toEqual([])
    expect(chain.state.undirected).toEqual([
      ['a', 'b'],
      ['b', 'c'],
    ])
  })

  it('the extension removes one node per step and keeps the skeleton, orientations and no new immoralities', () => {
    const p = pdag(
      [],
      [['a', 'b']],
      [
        ['b', 'c'],
        ['c', 'd'],
        ['b', 'd'],
      ],
    )
    const run = runAlgorithm(pdagToDagAlgorithm(p), { maxSteps: 10 })
    expect(run.steps).toBe(4)
    expect(run.state.undirected).toEqual([])
    const extension = pdagToDag(p)
    expect(dagEdges(extension)).toHaveLength(4)
    expect(isMarkovEquivalent(extension, extension)).toBe(true)
    // It keeps the directed edge and creates no immorality, which the PDAG's skeleton has none of.
    expect(dagEdges(extension)).toContainEqual(['a', 'b'])
    expect(immoralities(extension)).toEqual([])
    const l = logged(pdagToDagAlgorithm(p))
    expect(l.get('meek/node')).toHaveLength(4)
    expect(() =>
      runAlgorithm(
        pdagToDagAlgorithm(
          pdag(
            [],
            [],
            [
              ['a', 'b'],
              ['b', 'c'],
              ['c', 'd'],
              ['d', 'a'],
            ],
          ),
        ),
        { maxSteps: 10 },
      ),
    ).toThrow(/consistent extension/)
    expectResumes(pdagToDagAlgorithm(p), 2, (s) => s)
  })
})

describe('fciAlgorithm', () => {
  const z = normal(key(7), [400, 4]).data
  const rows = Array.from({ length: 400 }, (_, i) => {
    const x = z[4 * i] ?? 0
    const w = z[4 * i + 1] ?? 0
    const y = x + 0.5 * (z[4 * i + 2] ?? 0)
    return { x, y, w, t: y + w + 0.5 * (z[4 * i + 3] ?? 0) }
  })

  it('goes through the collider, pruning, collider and rule phases, to the PAG fci returns', () => {
    const alg = fciAlgorithm(rows)
    const run = runAlgorithm(alg, { maxSteps: 1000 })
    expect(run.status).toBe('converged')
    const old = fci(rows)
    expect(old.pag.edges).toHaveLength(3)
    const final = run.state.marks.map(([k, v]) => `${k.replace('\u0000', '>')}:${v}`).sort()
    const fromOld = old.pag.edges.flatMap((e) => [`${e.a}>${e.b}:${e.markB}`, `${e.b}>${e.a}:${e.markA}`]).sort()
    expect(final).toEqual(fromOld)
    expect(Object.keys(old.separatingSets).length).toBeGreaterThan(0)
    const l = logged(alg)
    const phase = l.get('fci/phase') as number[]
    expect(phase.every((p, i) => i === 0 || p >= (phase[i - 1] ?? 0))).toBe(true)
    expect(phase[0]).toBe(0)
    expect(phase.at(-1)).toBe(3)
    const rules = l.get('fci/rule') as number[]
    expect(rules.slice(-9)).toEqual([1, 2, 3, 4, 5, 6, 8, 9, 10])
    expect(Object.keys(fciContract)).toHaveLength(6)
  })

  it('orients the collider of a -> c <- b and leaves the other marks circles', () => {
    const w = normal(key(3), [500, 3]).data
    const collider = Array.from({ length: 500 }, (_, i) => {
      const a = w[3 * i] ?? 0
      const b = w[3 * i + 1] ?? 0
      return { a, b, c: a + b + 0.5 * (w[3 * i + 2] ?? 0) }
    })
    expect(fci(collider).pag.edges.map((e) => `${e.a} ${e.markA}-${e.markB} ${e.b}`)).toEqual([
      'a circle-arrow c',
      'b circle-arrow c',
    ])
  })

  it('never adds circles once oriented, and the last cycle of rules changes nothing', () => {
    const l = logged(fciAlgorithm(rows))
    const phase = l.get('fci/phase') as number[]
    const circles = l.get('fci/circles') as number[]
    const from = phase.indexOf(3)
    for (let i = from + 1; i < circles.length; i++) expect(circles[i] ?? 0).toBeLessThanOrEqual(circles[i - 1] ?? 0)
    const oriented = l.get('fci/oriented') as number[]
    expect(oriented.slice(-9).every((n) => n === 0)).toBe(true)
  })

  it('logs the removed edge and its separating set in the pruning phase', () => {
    const l = logged(fciAlgorithm(rows))
    const removed = (l.get('fci/removed') as number[][]).filter((r) => r.length === 2)
    expect(Array.isArray(removed)).toBe(true)
    for (const r of removed) expect(r[0]).not.toBe(r[1])
  })

  it('resumes from a JSON checkpoint at several cuts', () => {
    for (const cut of [1, 3, 12])
      expectResumes(fciAlgorithm(rows), cut, (s) => ({ m: s.marks, s: s.sepsets, p: s.phase }))
  }, 60_000)
})
