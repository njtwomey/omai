import { describe, expect, it } from 'vitest'
import { expectResumable } from '../../testing/mp-variants.ts'
import {
  beliefPropagation,
  calibrate,
  type Factor,
  factor,
  factorGraph,
  factorMarginal,
  factorValues,
  junctionTreeMarginals,
} from '../pgm/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import * as M from './index.ts'

const STATES = ['0', '1']
const name = (r: number, c: number): string => `x${r}${c}`

/** An Ising-type model on an n by m grid: agreeing neighbours weigh e^beta, a field of varying sign on every site. */
function grid(n: number, m: number, beta: number, field: number): ReturnType<typeof factorGraph> {
  const names: string[] = []
  const factors: Factor[] = []
  const e = Math.exp(beta)
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < m; c++) {
      names.push(name(r, c))
      if (c + 1 < m) factors.push(factor([name(r, c), name(r, c + 1)], [2, 2], [e, 1 / e, 1 / e, e]))
      if (r + 1 < n) factors.push(factor([name(r, c), name(r + 1, c)], [2, 2], [e, 1 / e, 1 / e, e]))
      const h = field * (((r * 3 + c * 5) % 4) - 1.5)
      factors.push(factor([name(r, c)], [2], [Math.exp(h), Math.exp(-h)]))
    }
  }
  return factorGraph(
    names.map((n0) => ({ name: n0, states: STATES })),
    factors,
  )
}

/** The largest error of a marginal probability against the exact (junction-tree) ones. */
function marginalError(g: M.RegionGraph, state: M.RegionState, exact: Record<string, Factor>): number {
  const snapshot = M.regionSnapshot(g)(state)
  let worst = 0
  for (const [v, f] of Object.entries(exact)) {
    const p = (snapshot.marginals[v] as { probs: number[] }).probs
    for (const [i, q] of factorValues(f).entries()) worst = Math.max(worst, Math.abs((p[i] as number) - q))
  }
  return worst
}

const run = (g: M.RegionGraph, options: M.RegionBpOptions = {}): ReturnType<typeof runAlgorithm<M.RegionState>> =>
  runAlgorithm(M.regionBpAlgorithm(g, { tol: 1e-12, maxSweeps: 3000, ...options }), { maxSteps: 100000 })

describe('region graphs: construction', () => {
  it('Bethe: factor regions with counting number 1 and variables with 1 minus their degree', () => {
    const fg = grid(2, 2, 0.4, 0)
    const g = M.regionGraph(fg, { kind: 'bethe' })
    expect(g.regions.filter((r) => r.variables.length === 2).map((r) => r.count)).toEqual([1, 1, 1, 1])
    expect(g.regions.filter((r) => r.variables.length === 1).map((r) => r.count)).toEqual([-1, -1, -1, -1])
    expect(g.edges.length).toBe(8)
  })
  it('Kikuchi on a 3 by 3 grid of 2 by 2 blocks: blocks 1, interior edges -1, the centre 1, the rest 0', () => {
    const g = M.regionGraph(grid(3, 3, 0.3, 0), { kind: 'clusters', clusters: M.blockClusters(3, 3, name, 2) })
    expect(g.regions.map((r) => r.count)).toEqual([1, 1, 1, 1, -1, -1, -1, -1, 1])
    const centre = g.regions.find((r) => r.variables.length === 1) as M.Region
    expect(centre.variables).toEqual(['x11'])
    expect(centre.parents.length).toBe(4)
  })
  it('junction: cliques 1 and one separator per tree edge -1, joined only to their two cliques', () => {
    const g = M.regionGraph(grid(3, 3, 0.3, 0), { kind: 'junction' })
    const cliques = g.regions.filter((r) => r.count === 1)
    const separators = g.regions.filter((r) => r.count === -1)
    expect(separators.length).toBe(cliques.length - 1)
    for (const s of separators) expect(s.parents.length).toBe(2)
  })
  it('every variable and every factor is counted once; refuses what is not a valid approximation', () => {
    const fg = grid(3, 3, 0.3, 0)
    for (const options of [
      { kind: 'bethe' } as const,
      { kind: 'clusters', clusters: M.blockClusters(3, 3, name, 2) } as const,
      { kind: 'junction' } as const,
    ]) {
      const g = M.regionGraph(fg, options)
      for (const v of fg.variables)
        expect(g.regions.reduce((a, r) => a + (r.variables.includes(v.name) ? r.count : 0), 0)).toBe(1)
    }
    // clusters that do not hold every pairwise factor
    expect(() =>
      M.regionGraph(fg, {
        kind: 'clusters',
        clusters: [
          ['x00', 'x01'],
          ['x11', 'x12'],
        ],
      }),
    ).toThrow(/sum to/)
    expect(() => M.regionGraph(fg, { kind: 'clusters', clusters: [['x00', 'nope']] })).toThrow(/unknown variable/)
    expect(() => M.blockClusters(2, 2, name, 3)).toThrow(/does not fit/)
  })
})

describe('Bethe regions: belief propagation', () => {
  it('is exact on a tree: the marginals and the log partition function', () => {
    const chain = grid(1, 6, 0.8, 0.3)
    const g = M.regionGraph(chain, { kind: 'bethe' })
    const r = run(g)
    expect(r.status).toBe('converged')
    expect(marginalError(g, r.state, junctionTreeMarginals(chain))).toBeLessThan(1e-9)
    expect(Math.abs(M.regionEvidence(g, r.state) - calibrate(chain).logPartition)).toBeLessThan(1e-9)
  })
  it('reproduces loopy BP on a grid: the beliefs and the Bethe log partition function of pgm and of the engine', () => {
    const fg = grid(3, 3, 0.4, 0.3)
    const g = M.regionGraph(fg, { kind: 'bethe' })
    const r = run(g)
    expect(r.status).toBe('converged')
    const ref = beliefPropagation(fg, { tol: 1e-13, maxIter: 5000 })
    expect(ref.converged).toBe(true)
    expect(Math.abs(M.regionEvidence(g, r.state) - ref.logPartition)).toBeLessThan(1e-7)
    expect(marginalError(g, r.state, ref.marginals)).toBeLessThan(1e-7)
  })
})

describe('junction regions: exact on any model', () => {
  const fg = grid(4, 4, 0.5, 0.3)
  const exact = junctionTreeMarginals(fg)
  const g = M.regionGraph(fg, { kind: 'junction' })
  const r = run(g)
  it('gives the exact marginals and log partition function of the loopy 4 by 4 grid', () => {
    expect(r.status).toBe('converged')
    expect(marginalError(g, r.state, exact)).toBeLessThan(1e-8)
    expect(Math.abs(M.regionEvidence(g, r.state) - calibrate(fg).logPartition)).toBeLessThan(1e-8)
  })
  it('the belief of every clique is the clique belief of the calibrated junction tree', () => {
    const calibrated = calibrate(fg)
    const beliefs = M.regionBeliefs(g, r.state)
    let matched = 0
    for (const [i, region] of g.regions.entries()) {
      if (region.count !== 1) continue
      const j = calibrated.tree.cliques.findIndex(
        (c) => c.length === region.variables.length && region.variables.every((v) => c.includes(v)),
      )
      expect(j).toBeGreaterThanOrEqual(0)
      const ref = factorValues(factorMarginal(calibrated.cliqueBeliefs[j] as Factor, region.variables))
      const total = ref.reduce((a, b) => a + b, 0)
      const mine = factorValues(beliefs[i] as Factor)
      for (const [k, v] of ref.entries()) expect(Math.abs((mine[k] as number) - v / total)).toBeLessThan(1e-8)
      matched++
    }
    expect(matched).toBe(g.regions.filter((x) => x.count === 1).length)
  })
})

describe('Kikuchi regions beat the Bethe approximation on a grid', () => {
  it('4 by 4 grid, 2 by 2 blocks: much smaller marginal and log Z errors against the junction tree', () => {
    const fg = grid(4, 4, 0.3, 0.3)
    const exact = junctionTreeMarginals(fg)
    const logZ = calibrate(fg).logPartition
    const bethe = M.regionGraph(fg, { kind: 'bethe' })
    const kikuchi = M.regionGraph(fg, { kind: 'clusters', clusters: M.blockClusters(4, 4, name, 2) })
    const rb = run(bethe)
    const rk = run(kikuchi)
    expect(rb.status).toBe('converged')
    expect(rk.status).toBe('converged')
    const eb = marginalError(bethe, rb.state, exact)
    const ek = marginalError(kikuchi, rk.state, exact)
    expect(ek).toBeLessThan(eb / 20)
    expect(Math.abs(M.regionEvidence(kikuchi, rk.state) - logZ)).toBeLessThan(
      Math.abs(M.regionEvidence(bethe, rb.state) - logZ) / 20,
    )
  })
  it('4 by 4 grid at a stronger coupling, 3 by 3 blocks: Bethe is off by 6e-2, Kikuchi by under 1e-5', () => {
    const fg = grid(4, 4, 0.5, 0.3)
    const exact = junctionTreeMarginals(fg)
    const logZ = calibrate(fg).logPartition
    const bethe = M.regionGraph(fg, { kind: 'bethe' })
    const kikuchi = M.regionGraph(fg, { kind: 'clusters', clusters: M.blockClusters(4, 4, name, 3) })
    const rb = run(bethe)
    const rk = run(kikuchi)
    expect(rk.status).toBe('converged')
    expect(marginalError(bethe, rb.state, exact)).toBeGreaterThan(5e-2)
    expect(marginalError(kikuchi, rk.state, exact)).toBeLessThan(1e-5)
    expect(Math.abs(M.regionEvidence(bethe, rb.state) - logZ)).toBeGreaterThan(0.3)
    expect(Math.abs(M.regionEvidence(kikuchi, rk.state) - logZ)).toBeLessThan(1e-4)
  })
  it('converges with either schedule on the 3 by 3 grid of 2 by 2 blocks, to the same fixed point', () => {
    const fg = grid(3, 3, 0.5, 0.3)
    const g = M.regionGraph(fg, { kind: 'clusters', clusters: M.blockClusters(3, 3, name, 2) })
    const a = run(g, { schedule: 'serial', damping: 0.5 })
    const b = run(g, { schedule: 'parallel', damping: 0.8 })
    expect(a.status).toBe('converged')
    expect(b.status).toBe('converged')
    const pa = M.regionSnapshot(g)(a.state).marginals
    const pb = M.regionSnapshot(g)(b.state).marginals
    for (const v of Object.keys(pa)) {
      const x = (pa[v] as { probs: number[] }).probs
      const y = (pb[v] as { probs: number[] }).probs
      expect(Math.abs((x[0] as number) - (y[0] as number))).toBeLessThan(1e-6)
    }
  })
})

describe('region graphs: the contract', () => {
  const fg = grid(3, 3, 0.3, 0.3)
  const g = M.regionGraph(fg, { kind: 'clusters', clusters: M.blockClusters(3, 3, name, 2) })
  it('serialises and resumes to the same state, serial and parallel', () => {
    for (const schedule of ['serial', 'parallel'] as const)
      expectResumable(M.regionBpAlgorithm(g, { schedule, granularity: 'edge', maxSweeps: 8, tol: 1e-14 }), 7)
  })
  it('runs anytime in slices that reach the same marginals as the uninterrupted run', () => {
    const alg = M.regionBpAlgorithm(g, { granularity: 'edge', maxSweeps: 6, tol: 1e-14 })
    const snapshot = M.regionSnapshot(g)
    let now = 0
    const first = M.runAnytimeWith(snapshot, alg, { budgetMs: 25, snapshotEveryMs: 5, clock: () => (now += 1) })
    expect(first.status).toBe('budget')
    expect(first.snapshots).toBeGreaterThanOrEqual(3)
    const second = M.runAnytimeWith(snapshot, alg, { budgetMs: 1e9, clock: () => (now += 1), from: first.slice })
    const whole = runAlgorithm(alg, { maxSteps: 100000 })
    expect(second.snapshot).toEqual(snapshot(whole.state))
  })
  it('puts the change, the evidence, the messages (region ids) and the marginals on the tape', () => {
    const rec = recorder()
    runAlgorithm(M.regionBpAlgorithm(g, { maxSweeps: 2 }), { maxSteps: 5, tracer: rec, policy: policies.teaching })
    const events = rec.run().events
    const names = new Set(events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    for (const metric of ['sweeps', 'updates', 'change', 'evidence']) expect(names.has(metric)).toBe(true)
    const sent = events.find((e) => e.type === 'message' && e.kind === 'message')
    expect(sent).toBeDefined()
    expect(events.some((e) => e.type === 'message' && e.kind === 'marginals')).toBe(true)
  })
  it('is a method whose regions are a knob, and refuses Kikuchi without clusters', () => {
    const method = M.regionBpMethod(fg, M.blockClusters(3, 3, name, 2))
    expect(method.name).toBe('region-bp')
    expect(Object.keys(method.config.spaces)).toEqual(
      expect.arrayContaining(['regions', 'schedule', 'granularity', 'damping', 'tol', 'maxSweeps']),
    )
    expect(Object.keys(method.contract as object)).toContain('rg/evidence')
    for (const regions of ['bethe', 'kikuchi', 'junction'])
      expect(method.make({ regions } as never, { kind: 'key', words: [0, 0] }).name).toBe('region-bp')
    expect(() => M.regionBpMethod(fg).make({ regions: 'kikuchi' } as never, { kind: 'key', words: [0, 0] })).toThrow(
      /clusters/,
    )
    expect(() => M.regionBpAlgorithm(g, { damping: 1 })).toThrow(/damping/)
  })
  it('a contradiction in the model ends the run as diverged, not as an exception', () => {
    const impossible = factorGraph(
      [
        { name: 'a', states: STATES },
        { name: 'b', states: STATES },
      ],
      [factor(['a', 'b'], [2, 2], [1, 0, 0, 1]), factor(['a'], [2], [1, 0]), factor(['b'], [2], [0, 1])],
    )
    const rg = M.regionGraph(impossible, { kind: 'bethe' })
    expect(run(rg).status).toBe('diverged')
  })
})
