import { describe, expect, it } from 'vitest'
import { expectResumable, probitRows } from '../../../testing/mp-variants.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import * as M from '../index.ts'

const operators = M.defaultOperators()
const close = (a: number, b: number, tol: number): void => {
  expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol * (1 + Math.abs(b)))
}

const prior3 = M.gaussianMvPrior('w', [0, 0, 0], [4, 0, 0, 0, 4, 0, 0, 0, 4])

/** The regression graph of n probit rows on three weights. */
function probitGraph(n: number): M.CompiledGraph {
  const { x, y } = probitRows(n)
  return M.compileGraph(
    {
      variables: [
        { name: 'w', family: { kind: 'gaussian-mv', size: 3 } },
        ...y.map((l, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: l })),
      ],
      factors: [
        prior3,
        ...x.map((a, i): M.FactorSpec => ({ operator: 'probit-row', args: [`y${i}`, 'w'], params: { a } })),
      ],
    },
    operators,
  )
}

const marginal = (snapshot: M.MpSnapshot): { mean: number[]; variance: number[] } => {
  const w = snapshot.marginals['w'] as Record<string, never>
  return { mean: w['mean'] as unknown as number[], variance: w['variance'] as unknown as number[] }
}

describe('stochastic EP: memory independent of the number of factors', () => {
  it('the state is one message however many data factors there are, where EP keeps one per factor', () => {
    const sizes = [60, 600].map((n) => {
      const g = probitGraph(n)
      const alg = M.stochasticEpAlgorithm(g, { order: 'sequential', batchSize: n })
      const r = runAlgorithm(alg, { maxSteps: 1 })
      const ep = runAlgorithm(M.epAlgorithm(g), { maxSteps: 1 })
      return {
        site: r.state.site.length,
        json: JSON.stringify(alg.serialise?.(r.state)).length,
        epEdges: ep.state.messages.length,
      }
    })
    expect(sizes[0]?.site).toBe(12)
    expect(sizes[1]?.site).toBe(12)
    // the serialised state differs only in the digits of the counters and numbers
    expect((sizes[1] as { json: number }).json / (sizes[0] as { json: number }).json).toBeLessThan(1.3)
    expect((sizes[0] as { epEdges: number }).epEdges).toBe(61)
    expect((sizes[1] as { epEdges: number }).epEdges).toBe(601)
  })
})

describe('stochastic EP agrees with EP on exchangeable data', () => {
  it('with a decaying step, on 200 probit rows, the posterior is close to the EP one', () => {
    const g = probitGraph(200)
    const ep = runAlgorithm(M.epAlgorithm(g, { tol: 1e-10 }), { maxSteps: 500 })
    expect(ep.status).toBe('converged')
    const e = marginal(M.mpSnapshot(g, ep.state))
    const r = runAlgorithm(
      M.stochasticEpAlgorithm(g, { step: 'decay', order: 'sequential', batchSize: 200, tol: 1e-12, maxSweeps: 60 }),
      { maxSteps: 100 },
    )
    const s = marginal(M.sepSnapshot(g)(r.state))
    for (let i = 0; i < 3; i++) {
      expect(Math.abs((s.mean[i] as number) - (e.mean[i] as number))).toBeLessThan(0.02)
      close(s.variance[i] as number, e.variance[i] as number, 0.03)
    }
  })

  it('with the published constant step the average fluctuates but stays within a few percent', () => {
    const g = probitGraph(200)
    const ep = runAlgorithm(M.epAlgorithm(g, { tol: 1e-10 }), { maxSteps: 500 })
    const e = marginal(M.mpSnapshot(g, ep.state))
    const r = runAlgorithm(
      M.stochasticEpAlgorithm(g, { order: 'random', seed: 4, batchSize: 200, maxSweeps: 30, tol: 1e-12 }),
      {
        maxSteps: 100,
      },
    )
    const s = marginal(M.sepSnapshot(g)(r.state))
    for (let i = 0; i < 3; i++) {
      expect(Math.abs((s.mean[i] as number) - (e.mean[i] as number))).toBeLessThan(0.1)
      close(s.variance[i] as number, e.variance[i] as number, 0.1)
    }
  })

  it('on Gaussian rows the fixed point is the exact conjugate posterior (the average of the rows is the average)', () => {
    const n = 100
    const { x } = probitRows(n)
    const tau = 4
    const y = x.map(
      (a, i) => (a[0] as number) * 0.5 + (a[1] as number) * 1.5 - (a[2] as number) * 0.7 + 0.3 * Math.sin(5 * i),
    )
    const g = M.compileGraph(
      {
        variables: [
          { name: 'w', family: { kind: 'gaussian-mv', size: 3 } },
          ...y.map((v, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'gaussian' }, observed: v })),
        ],
        factors: [
          prior3,
          ...x.map(
            (a, i): M.FactorSpec => ({
              operator: 'linear-regression-row',
              args: [`y${i}`, 'w'],
              params: { a, precision: tau },
            }),
          ),
        ],
      },
      operators,
    )
    // Bayesian linear regression: precision P = I / 4 + tau X'X, mean = P^-1 tau X'y
    const P = [0, 1, 2].map((i) =>
      [0, 1, 2].map((j) => (i === j ? 0.25 : 0) + tau * x.reduce((s, a) => s + (a[i] as number) * (a[j] as number), 0)),
    )
    const b = [0, 1, 2].map((i) => tau * x.reduce((s, a, k) => s + (a[i] as number) * (y[k] as number), 0))
    const M3 = P.map((row, i) => [...row, b[i] as number])
    for (let i = 0; i < 3; i++) {
      for (let k = i + 1; k < 3; k++) {
        const f = ((M3[k] as number[])[i] as number) / ((M3[i] as number[])[i] as number)
        for (let j = i; j < 4; j++)
          (M3[k] as number[])[j] = ((M3[k] as number[])[j] as number) - f * ((M3[i] as number[])[j] as number)
      }
    }
    const mean = [0, 0, 0]
    for (let i = 2; i >= 0; i--) {
      let s = (M3[i] as number[])[3] as number
      for (let j = i + 1; j < 3; j++) s -= ((M3[i] as number[])[j] as number) * (mean[j] as number)
      mean[i] = s / ((M3[i] as number[])[i] as number)
    }
    const r = runAlgorithm(
      M.stochasticEpAlgorithm(g, { step: 'decay', order: 'sequential', batchSize: n, tol: 1e-12, maxSweeps: 300 }),
      { maxSteps: 400 },
    )
    const s = marginal(M.sepSnapshot(g)(r.state))
    for (let i = 0; i < 3; i++) expect(Math.abs((s.mean[i] as number) - (mean[i] as number))).toBeLessThan(2e-4)
  })
})

describe('stochastic EP: the contract', () => {
  const g = probitGraph(30)
  it('serialises and resumes to the same state, in both orders', () => {
    for (const order of ['sequential', 'random'] as const)
      expectResumable(M.stochasticEpAlgorithm(g, { order, batchSize: 7, maxSweeps: 6, tol: 1e-12 }), 5)
  })
  it('runs anytime in slices that reach the same marginals as the uninterrupted run', () => {
    const alg = M.stochasticEpAlgorithm(g, { batchSize: 5, maxSweeps: 8, tol: 1e-12 })
    const snapshot = M.sepSnapshot(g)
    let now = 0
    const seen: number[] = []
    const first = M.runAnytimeWith(snapshot, alg, {
      budgetMs: 30,
      snapshotEveryMs: 10,
      clock: () => (now += 1),
      onSnapshot: (s) => seen.push(s.updates),
    })
    expect(first.status).toBe('budget')
    expect(seen.length).toBeGreaterThanOrEqual(3)
    const second = M.runAnytimeWith(snapshot, alg, { budgetMs: 1e9, clock: () => (now += 1), from: first.slice })
    const whole = runAlgorithm(alg, { maxSteps: 100000 })
    expect(second.snapshot).toEqual(snapshot(whole.state))
  })
  it('puts the change, the sweeps, the counts and the marginals on the tape', () => {
    const rec = recorder()
    runAlgorithm(M.stochasticEpAlgorithm(g, { maxSweeps: 3, batchSize: 30 }), {
      maxSteps: 10,
      tracer: rec,
      policy: policies.teaching,
    })
    const names = new Set(rec.run().events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    for (const name of ['sweeps', 'updates', 'improper', 'skipped', 'damped', 'reset'])
      expect(names.has(name)).toBe(true)
    expect(rec.run().events.some((e) => e.type === 'message' && e.kind === 'marginals')).toBe(true)
  })
  it('is a method with its knobs and typed tape', () => {
    const method = M.stochasticEpMethod(g)
    expect(method.name).toBe('sep')
    expect(Object.keys(method.config.spaces)).toEqual(
      expect.arrayContaining([
        'batchSize',
        'order',
        'seed',
        'damping',
        'step',
        'decaySweeps',
        'improper',
        'tol',
        'maxSweeps',
      ]),
    )
    expect(Object.keys(method.contract as object)).toContain('sep/snapshot')
    const alg = method.make({ step: 'decay', order: 'sequential', batchSize: 3 }, { kind: 'key', words: [0, 0] })
    expect(alg.name).toBe('sep')
  })
  it('refuses graphs it is not for', () => {
    expect(() => M.stochasticEpAlgorithm(g, { damping: 1 })).toThrow(/damping/)
    expect(() => M.stochasticEpAlgorithm(g, { batchSize: 0 })).toThrow(/batch/)
    expect(() => M.stochasticEpAlgorithm(g, { shared: 'nope' })).toThrow(/no variable/)
    expect(() => M.stochasticEpAlgorithm(g, { data: ['nope'] })).toThrow(/no factor/)
    const two = M.compileGraph(
      {
        variables: [
          { name: 'a', family: { kind: 'gaussian' } },
          { name: 'b', family: { kind: 'gaussian' } },
        ],
        factors: [M.gaussianPrior('a', 0, 1), M.gaussianPrior('b', 0, 1), { operator: 'plus', args: ['a', 'b', 'a'] }],
      },
      operators,
    )
    expect(() => M.stochasticEpAlgorithm(two)).toThrow(/shared variable/)
    expect(() => M.stochasticEpAlgorithm(two, { shared: 'a' })).toThrow(/touch the shared variable only/)
  })
})
