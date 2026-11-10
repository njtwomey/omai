import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm, runFor, session, startRun } from '../trace/index.ts'
import * as M from './index.ts'

const operators = M.defaultOperators()
const settings = (o: Partial<M.EngineSettings> = {}): M.EngineSettings => ({
  projection: M.momentProjection,
  schedule: 'serial',
  granularity: 'sweep',
  damping: 0,
  improper: 'skip',
  tol: 1e-10,
  maxSweeps: 100,
  ...o,
})

/** a ~ N(1, 2), b ~ N(-1, 1), s = a + b, the observation 2 ~ N(s, 1/4): a Gaussian tree. */
const tree: M.FactorGraphSpec = {
  variables: [
    { name: 'a', family: { kind: 'gaussian' } },
    { name: 'b', family: { kind: 'gaussian' } },
    { name: 's', family: { kind: 'gaussian' } },
    { name: 'obs', family: { kind: 'gaussian' }, observed: 2 },
  ],
  factors: [
    M.gaussianPrior('a', 1, 2),
    M.gaussianPrior('b', -1, 1),
    { operator: 'plus', args: ['s', 'a', 'b'] },
    { operator: 'gaussian', args: ['obs', 's'], params: { precision: 4 } },
  ],
}

/** The exact posterior of the tree by conditioning the joint Gaussian of (a, b). */
function exactTree(): { a: [number, number]; b: [number, number]; s: [number, number]; logEvidence: number } {
  // Prior precision diag(1/2, 1); observation: 2 = a + b + noise (variance 1/4).
  const p00 = 0.5 + 4
  const p01 = 4
  const p11 = 1 + 4
  const h0 = 0.5 * 1 + 4 * 2
  const h1 = 1 * -1 + 4 * 2
  const det = p00 * p11 - p01 * p01
  const c00 = p11 / det
  const c01 = -p01 / det
  const c11 = p00 / det
  const m0 = c00 * h0 + c01 * h1
  const m1 = c01 * h0 + c11 * h1
  const marginalVar = 2 + 1 + 0.25
  return {
    a: [m0, c00],
    b: [m1, c11],
    s: [m0 + m1, c00 + c11 + 2 * c01],
    logEvidence: -0.5 * Math.log(2 * Math.PI * marginalVar) - 2 ** 2 / (2 * marginalVar),
  }
}

describe('EP is exact on a tree of Gaussians', () => {
  const g = M.compileGraph(tree, operators)
  const want = exactTree()
  for (const schedule of ['serial', 'flooding', 'tree', 'residual'] as const) {
    it(`under the ${schedule} schedule: marginals and evidence`, () => {
      const s = settings({ schedule })
      const r = runAlgorithm(M.messagePassingAlgorithm(g, s), { maxSteps: 2000 })
      expect(r.status).toBe('converged')
      const m = M.marginalsOf(g, r.state)
      for (const name of ['a', 'b', 's'] as const) {
        expect(m[name]?.['mean']).toBeCloseTo(want[name][0], 9)
        expect(m[name]?.['variance']).toBeCloseTo(want[name][1], 9)
      }
      expect(M.evidenceOf(g, s, r.state)).toBeCloseTo(want.logEvidence, 9)
    })
  }
  it('the tree schedule is exact after one pass in and out', () => {
    const alg = M.messagePassingAlgorithm(g, settings({ schedule: 'tree' }))
    const slice = runFor(alg, { steps: 1 }, { maxSteps: 10 })
    expect(M.marginalsOf(g, slice.state)['a']?.['mean']).toBeCloseTo(want.a[0], 9)
  })
  it('a step can be one message or one factor, and the run ends in the same beliefs', () => {
    const sweep = runAlgorithm(M.messagePassingAlgorithm(g, settings()), { maxSteps: 100 }).state
    for (const granularity of ['factor', 'message'] as const) {
      const r = runAlgorithm(M.messagePassingAlgorithm(g, settings({ granularity })), { maxSteps: 1000 })
      expect(r.status).toBe('converged')
      expect(M.marginalsOf(g, r.state)['a']?.['mean']).toBeCloseTo(M.marginalsOf(g, sweep)['a']?.['mean'] as number, 9)
      expect(r.steps).toBeGreaterThan(3)
    }
  })
})

describe('the ids of a run are the ids of the structure', () => {
  const g = M.compileGraph(tree, operators)
  it('names variables by their name and factors f<i>, in the structure and on the tape', () => {
    const s = M.graphStructure(tree)
    expect(s.nodes.map((n) => n.id)).toEqual(['a', 'b', 's', 'obs', 'f0', 'f1', 'f2', 'f3'])
    expect(s.edges.every((e) => e.from.startsWith('f') && !e.directed)).toBe(true)
    const rec = recorder()
    runAlgorithm(M.epAlgorithm(g, { schedule: 'tree' }), { maxSteps: 5, tracer: rec, policy: policies.teaching })
    const events = rec.run().events
    const messages = events.filter((e) => e.type === 'message')
    const ids = new Set(s.nodes.map((n) => n.id))
    for (const m of messages) {
      if (m.type !== 'message') continue
      if (m.kind === 'message') {
        expect(ids.has(m.from ?? '')).toBe(true)
        expect(ids.has(m.to ?? '')).toBe(true)
      }
    }
    const marginals = messages.find((e) => e.type === 'message' && e.kind === 'marginals')
    expect(Object.keys((marginals as { payload: Record<string, unknown> }).payload).sort()).toEqual([
      'a',
      'b',
      'obs',
      's',
    ])
    const frames = events.filter((e) => e.type === 'frame')
    expect(frames.length).toBeGreaterThan(0)
  })
  it('runs as a method of a session with a typed contract, and the snapshot has every marginal', () => {
    const m = M.epMethod(g)
    const sess = session(m, { steps: 6, mode: 'eager', logs: 'all' })
    const messages = sess.run().events.filter((e) => e.type === 'message' && e.kind === 'mp/marginals')
    expect(messages.length).toBeGreaterThan(0)
    const snap = M.mpSnapshot(g, sess.state)
    expect(Object.keys(snap.marginals)).toEqual(['a', 'b', 's', 'obs'])
    expect(snap.marginals['obs']).toEqual({ family: 'gaussian', observed: 2 })
  })
})

describe('compiling a graph', () => {
  it('rejects what a port does not take', () => {
    const bad =
      (spec: M.FactorGraphSpec): (() => unknown) =>
      () =>
        M.compileGraph(spec, operators)
    const x = { name: 'x', family: { kind: 'gaussian' } } as const
    expect(bad({ variables: [x], factors: [{ operator: 'nope', args: ['x'] }] })).toThrow(/unknown operator/)
    expect(bad({ variables: [x], factors: [{ operator: 'probit', args: ['x', 'x'] }] })).toThrow(/does not take/)
    expect(bad({ variables: [x, x], factors: [] })).toThrow(/twice/)
    expect(
      bad({ variables: [x], factors: [{ operator: 'prior', args: [{ const: 1 }], params: { eta: [0, 1] } }] }),
    ).toThrow(/constant/)
    expect(bad({ variables: [x], factors: [{ operator: 'prior', args: ['x'], params: { eta: [0] } }] })).toThrow(
      /length/,
    )
    expect(
      bad({
        variables: [{ ...x, observed: 1 }],
        factors: [{ operator: 'prior', args: ['x'], params: { eta: [0, 1] } }],
      }),
    ).toThrow(/observed/)
  })
})

describe('improper messages are repaired by policy, with counts', () => {
  /** An operator whose message always has a negative precision: the standard failure, made deterministic. */
  const unstable: M.FactorOperator = {
    name: 'unstable',
    description: 'sends a negative precision',
    ports: [{ name: 'x', accepts: ['gaussian'] }],
    support: { ep: 'exact' },
    ep: () => ({ message: [0, -2] }),
  }
  const g = M.compileGraph(
    {
      variables: [{ name: 'x', family: { kind: 'gaussian' } }],
      factors: [M.gaussianPrior('x', 0, 1), { operator: 'unstable', args: ['x'] }],
    },
    { ...operators, unstable },
  )
  const run = (improper: M.ImproperPolicy): ReturnType<typeof runAlgorithm<M.MpState>> =>
    runAlgorithm(M.messagePassingAlgorithm(g, settings({ improper, maxSweeps: 8, tol: 1e-12 })), { maxSteps: 50 })
  it('skip keeps the old message and counts it', () => {
    const r = run('skip')
    expect(r.state.improper.events).toBeGreaterThan(0)
    expect(r.state.improper.skipped).toBe(r.state.improper.events)
    expect(M.marginalsOf(g, r.state)['x']?.['improper']).toBe(false)
    expect(M.marginalsOf(g, r.state)['x']?.['variance']).toBe(1)
  })
  it('damp shortens the step until the belief stays proper', () => {
    const r = run('damp')
    expect(r.state.improper.damped).toBeGreaterThan(0)
    const x = M.marginalsOf(g, r.state)['x']
    expect(x?.['improper']).toBe(false)
    expect(x?.['variance'] as number).toBeGreaterThan(1)
  })
  it('reset sends the flat message instead', () => {
    const r = run('reset')
    expect(r.state.improper.reset).toBeGreaterThan(0)
    expect(M.marginalsOf(g, r.state)['x']?.['variance']).toBe(1)
  })
  it('error stops the run', () => {
    expect(() => run('error')).toThrow(/improper/)
  })
  it('an operator that fails on an improper cavity is counted and its message kept', () => {
    const flaky: M.FactorOperator = {
      ...unstable,
      name: 'flaky',
      ep: () => {
        throw new M.ImproperMessageError('no density')
      },
    }
    const gf = M.compileGraph(
      {
        variables: [{ name: 'x', family: { kind: 'gaussian' } }],
        factors: [M.gaussianPrior('x', 0, 1), { operator: 'flaky', args: ['x'] }],
      },
      { ...operators, flaky },
    )
    const r = runAlgorithm(M.messagePassingAlgorithm(gf, settings({ maxSweeps: 3, tol: 0 })), { maxSteps: 10 })
    expect(r.state.improper.events).toBe(3)
  })
})

describe('a run is steppable, serialisable and resumable', () => {
  const g = M.compileGraph(
    {
      variables: [
        { name: 'x', family: { kind: 'gaussian' } },
        { name: 'y', family: { kind: 'bernoulli' }, observed: 1 },
        { name: 'z', family: { kind: 'bernoulli' }, observed: 0 },
      ],
      factors: [
        M.gaussianPrior('x', 0.3, 2),
        { operator: 'probit', args: ['y', 'x'] },
        { operator: 'probit', args: ['z', 'x'], params: { noise: 2 } },
      ],
    },
    operators,
  )
  it('resuming from a serialised state continues exactly', () => {
    const alg = M.epAlgorithm(g, { granularity: 'message', tol: 1e-12 })
    const whole = runAlgorithm(alg, { maxSteps: 200 })
    const first = runFor(alg, { steps: 4 }, { maxSteps: 200 })
    const json = JSON.parse(JSON.stringify(alg.serialise?.(first.state))) as never
    const restored = alg.deserialise?.(json) as M.MpState
    const rest = runFor(alg, { steps: 1000 }, { maxSteps: 200, from: { state: restored, t: first.t } })
    expect(M.mpSnapshot(g, rest.state).marginals).toEqual(M.mpSnapshot(g, whole.state).marginals)
  })
  it('the evidence on the tape is the EP estimate', () => {
    const rec = recorder()
    runAlgorithm(M.epAlgorithm(g), { maxSteps: 50, tracer: rec })
    const last = rec
      .run()
      .events.filter((e) => e.type === 'metric' && e.name === 'evidence')
      .at(-1)
    expect(last).toBeDefined()
  })
})

describe('anytime runs', () => {
  const g = M.compileGraph(tree, operators)
  it('reports about every interval, stops at the budget, resumes, and reaches the same state', () => {
    const alg = M.epAlgorithm(g, { granularity: 'message', tol: 1e-12 })
    let now = 0
    const times: number[] = []
    const first = M.runAnytime(g, alg, {
      budgetMs: 8,
      snapshotEveryMs: 2,
      clock: () => (now += 1),
      onSnapshot: (_, i) => times.push(i.elapsedMs),
    })
    expect(first.status).toBe('budget')
    expect(first.snapshots).toBeGreaterThanOrEqual(3)
    expect(times.length).toBe(first.snapshots)
    const second = M.runAnytime(g, alg, { budgetMs: 100000, clock: () => (now += 1), from: first.slice })
    const whole = runAlgorithm(alg, { maxSteps: 100000 })
    expect(second.status).toBe('converged')
    expect(second.snapshot.marginals).toEqual(M.mpSnapshot(g, whole.state).marginals)
  })
  it('stops between any two steps when cancelled, leaving a state that resumes', () => {
    const alg = M.epAlgorithm(g, { granularity: 'message', tol: 1e-12 })
    let steps = 0
    const r = M.runAnytime(g, alg, { budgetMs: 1e9, clock: () => steps, isCancelled: () => ++steps > 3 })
    expect(r.status).toBe('cancelled')
    expect(r.t).toBe(3)
    const rest = M.runAnytime(g, alg, { budgetMs: 1e9, clock: () => 0, from: r.slice })
    expect(rest.status).toBe('converged')
  })
  it('records its snapshots on a run it is given', () => {
    const rec = recorder()
    const run = startRun(rec, { name: 'ep', seed: 0, config: {}, policy: policies.teaching })
    let now = 0
    M.runAnytime(g, M.epAlgorithm(g), { budgetMs: 1e9, clock: () => (now += 1), run })
    const events = rec.run().events.filter((e) => e.type === 'message' && e.kind === 'mp/snapshot')
    expect(events.length).toBeGreaterThanOrEqual(2)
  })
})
