import { describe, expect, it } from 'vitest'
import { epAlgorithm as gpEpAlgorithm } from '../../gp/index.ts'
import { variationalMixtureAlgorithm, variationalPriors } from '../../mixture/index.ts'
import { ndtri } from '../../numerics/special/index.ts'
import {
  beliefPropagation,
  expectationPropagation,
  factor,
  factorGraph,
  factorValues,
  gaussianBeliefPropagation,
  gaussianEp,
  gmrf,
  trueskillUpdate,
  variationalMessagePassing,
} from '../../pgm/index.ts'
import { item } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as M from '../index.ts'
import { constantArray, zeroArray } from '../numerics/_arrays.ts'

const operators = M.defaultOperators()
const close = (a: number, b: number, tol = 1e-8): void => {
  expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol * (1 + Math.abs(b)))
}

describe('probit EP reproduces the Gaussian-latent EP of pgm', () => {
  const x = [
    [1, 0.5, -0.3],
    [1, -1.2, 0.8],
    [1, 0.3, 1.5],
    [1, 1.1, -0.7],
    [1, -0.4, -1.1],
  ]
  const labels = [1, 0, 1, 1, 0]
  const prior = {
    mean: [0, 0, 0],
    covariance: [
      [4, 0, 0],
      [0, 4, 0],
      [0, 0, 4],
    ],
  }
  const spec: M.FactorGraphSpec = {
    variables: [
      { name: 'w', family: { kind: 'gaussian-mv', size: 3 } },
      ...x.map((_, i): M.VariableSpec => ({ name: `s${i}`, family: { kind: 'gaussian' } })),
      ...x.map(
        (_, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: labels[i] as number }),
      ),
    ],
    factors: [
      M.gaussianMvPrior('w', prior.mean, [4, 0, 0, 0, 4, 0, 0, 0, 4]),
      ...x.flatMap((row, i): M.FactorSpec[] => [
        { operator: 'dot-constant', args: [`s${i}`, 'w'], params: { a: row } },
        { operator: 'probit', args: [`y${i}`, `s${i}`] },
      ]),
    ],
  }
  it('gives the same posterior and the same evidence', () => {
    const g = M.compileGraph(spec, operators)
    const s: M.EngineSettings = {
      projection: M.momentProjection,
      schedule: 'serial',
      granularity: 'sweep',
      damping: 0,
      improper: 'damp',
      tol: 1e-12,
      maxSweeps: 300,
    }
    const r = runAlgorithm(M.messagePassingAlgorithm(g, s), { maxSteps: 300 })
    expect(r.status).toBe('converged')
    const ref = gaussianEp(
      prior,
      x.map((row, i) => ({
        kind: 'probit' as const,
        weights: row,
        label: labels[i] === 1 ? (1 as const) : (-1 as const),
      })),
      { tol: 1e-13, maxIter: 500 },
    )
    const w = M.marginalsOf(g, r.state)['w'] as Record<string, never>
    const mean = w['mean'] as unknown as number[]
    const cov = w['covariance'] as unknown as number[][]
    for (let i = 0; i < 3; i++) {
      close(mean[i] as number, ref.mean[i] as number, 1e-7)
      for (let j = 0; j < 3; j++)
        close((cov[i] as number[])[j] as number, (ref.covariance[i] as number[])[j] as number, 1e-7)
    }
    close(M.evidenceOf(g, s, r.state), ref.logMarginalLikelihood, 1e-7)
  })
})

describe('probit EP reproduces the Gaussian-process classification EP of gp', () => {
  it('gives the same latent means and the same log marginal likelihood', () => {
    const pts = [-1.6, -0.9, -0.3, 0.4, 1.0, 1.7]
    const K = pts.map((a) => pts.map((b) => Math.exp(-0.5 * (a - b) ** 2) + (a === b ? 1e-6 : 0)))
    const y = [0, 0, 1, 0, 1, 1]
    const n = pts.length
    const spec: M.FactorGraphSpec = {
      variables: [
        { name: 'f', family: { kind: 'gaussian-mv', size: n } },
        ...pts.map((_, i): M.VariableSpec => ({ name: `s${i}`, family: { kind: 'gaussian' } })),
        ...pts.map(
          (_, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: y[i] as number }),
        ),
      ],
      factors: [
        M.gaussianMvPrior('f', zeroArray(n), K.flat()),
        ...pts.flatMap((_, i): M.FactorSpec[] => [
          { operator: 'dot-constant', args: [`s${i}`, 'f'], params: { a: pts.map((__, j) => (i === j ? 1 : 0)) } },
          { operator: 'probit', args: [`y${i}`, `s${i}`] },
        ]),
      ],
    }
    const g = M.compileGraph(spec, operators)
    const s: M.EngineSettings = {
      projection: M.momentProjection,
      schedule: 'serial',
      granularity: 'sweep',
      damping: 0,
      improper: 'damp',
      tol: 1e-12,
      maxSweeps: 300,
    }
    const r = runAlgorithm(M.messagePassingAlgorithm(g, s), { maxSteps: 300 })
    expect(r.status).toBe('converged')
    const ref = runAlgorithm(gpEpAlgorithm(K, y, { tol: 1e-13 }), { maxSteps: 2000 })
    expect(ref.status).toBe('converged')
    const mean = (M.marginalsOf(g, r.state)['f'] as Record<string, never>)['mean'] as unknown as number[]
    for (let i = 0; i < n; i++) close(mean[i] as number, ref.state.mu[i] as number, 1e-6)
    close(M.evidenceOf(g, s, r.state), ref.state.logMarginalLikelihood, 1e-6)
  })
})

describe('discrete tables reproduce the belief propagation, EP and VMP of pgm', () => {
  const states = ['0', '1']
  const pair = [0.9, 0.3, 0.2, 0.8]
  const vars = ['a', 'b', 'c']
  const fg = factorGraph(
    vars.map((name) => ({ name, states })),
    [
      factor(['a'], [2], [0.3, 0.7]),
      factor(['a', 'b'], [2, 2], pair),
      factor(['b', 'c'], [2, 2], [0.6, 0.4, 0.5, 0.9]),
      factor(['a', 'c'], [2, 2], [0.7, 0.5, 0.4, 0.8]),
    ],
  )
  const spec: M.FactorGraphSpec = {
    variables: vars.map((name) => ({ name, family: { kind: 'discrete', size: 2 } })),
    factors: fg.factors.map((f) => ({
      operator: 'table',
      args: [...f.variables],
      params: { states: [...f.cardinality], values: factorValues(f) },
    })),
  }
  const g = M.compileGraph(spec, operators)
  const probs = (state: M.MpState, name: string): number[] =>
    (M.marginalsOf(g, state)[name]?.['probs'] as number[]) ?? []
  it('BP on a loopy graph: marginals and the Bethe evidence', () => {
    const ref = beliefPropagation(fg, { schedule: 'flooding', tol: 1e-13, maxIter: 500 })
    const s: M.EngineSettings = {
      projection: M.sumProductProjection,
      schedule: 'flooding',
      granularity: 'sweep',
      damping: 0,
      improper: 'skip',
      tol: 1e-13,
      maxSweeps: 500,
    }
    const r = runAlgorithm(M.messagePassingAlgorithm(g, s), { maxSteps: 500 })
    expect(r.status).toBe('converged')
    for (const name of vars) {
      const want = factorValues(ref.marginals[name] as never)
      for (const [i, p] of probs(r.state, name).entries()) close(p, want[i] as number, 1e-9)
    }
    close(M.evidenceOf(g, s, r.state), ref.logPartition, 1e-8)
  })
  it('BP on a tree is exact (the tree schedule, one pass)', () => {
    const chain = factorGraph(
      vars.map((name) => ({ name, states })),
      [
        factor(['a'], [2], [0.3, 0.7]),
        factor(['a', 'b'], [2, 2], pair),
        factor(['b', 'c'], [2, 2], [0.6, 0.4, 0.5, 0.9]),
      ],
    )
    const cg = M.compileGraph(
      {
        variables: spec.variables,
        factors: chain.factors.map((f) => ({
          operator: 'table',
          args: [...f.variables],
          params: { states: [...f.cardinality], values: factorValues(f) },
        })),
      },
      operators,
    )
    const r = runAlgorithm(M.bpAlgorithm(cg, { schedule: 'tree', tol: 1e-13 }), { maxSteps: 20 })
    const ref = beliefPropagation(chain, { schedule: 'tree' })
    for (const name of vars) {
      const want = factorValues(ref.marginals[name] as never)
      for (const [i, p] of ((M.marginalsOf(cg, r.state)[name]?.['probs'] ?? []) as number[]).entries())
        close(p, want[i] as number, 1e-10)
    }
  })
  it('EP on discrete tables has the fixed points of BP and of pgm EP', () => {
    const ref = expectationPropagation(fg, { tol: 1e-13, maxIter: 500 })
    const r = runAlgorithm(M.epAlgorithm(g, { schedule: 'flooding', tol: 1e-13 }), { maxSteps: 500 })
    for (const name of vars) {
      const want = factorValues(ref.marginals[name] as never)
      for (const [i, p] of probs(r.state, name).entries()) close(p, want[i] as number, 1e-7)
    }
  })
  it('VMP on discrete tables reaches the mean-field fixed point of pgm', () => {
    const ref = variationalMessagePassing(fg, { tol: 1e-14, maxIter: 1000 })
    const r = runAlgorithm(M.vmpAlgorithm(g, { tol: 1e-14 }), { maxSteps: 1000 })
    expect(r.status).toBe('converged')
    for (const name of vars) {
      const want = factorValues(ref.marginals[name] as never)
      for (const [i, p] of probs(r.state, name).entries()) close(p, want[i] as number, 1e-7)
    }
  })
})

describe('Gaussian factors reproduce Gaussian belief propagation of pgm', () => {
  it('a loopy walk-summable field: means exact, and the variances of pgm', () => {
    const names = ['u', 'v', 'w']
    const P = [
      [2, -0.6, 0.4],
      [-0.6, 2, -0.5],
      [0.4, -0.5, 2],
    ]
    const mean = [0.5, -1, 2]
    const model = gmrf(names, mean, P)
    const h = P.map((row) => row.reduce((acc, v, j) => acc + v * (mean[j] as number), 0))
    const factors: M.FactorSpec[] = [
      ...names.map(
        (n, i): M.FactorSpec => ({
          operator: 'gaussian-joint',
          args: [n],
          params: { J: [P[i]?.[i] as number], h: [h[i] as number] },
        }),
      ),
      ...(
        [
          [0, 1],
          [1, 2],
          [0, 2],
        ] as [number, number][]
      ).map(
        ([i, j]): M.FactorSpec => ({
          operator: 'gaussian-joint',
          args: [names[i] as string, names[j] as string],
          params: { J: [0, P[i]?.[j] as number, P[i]?.[j] as number, 0], h: [0, 0] },
        }),
      ),
    ]
    const g = M.compileGraph(
      { variables: names.map((name) => ({ name, family: { kind: 'gaussian' } })), factors },
      operators,
    )
    const r = runAlgorithm(M.epAlgorithm(g, { schedule: 'flooding', tol: 1e-13 }), { maxSteps: 1000 })
    expect(r.status).toBe('converged')
    const ref = gaussianBeliefPropagation(model, { tol: 1e-13, maxIter: 2000 })
    for (const n of names) {
      close(M.marginalsOf(g, r.state)[n]?.['mean'] as number, ref.means[n] as number, 1e-8)
      close(M.marginalsOf(g, r.state)[n]?.['variance'] as number, ref.variances[n] as number, 1e-7)
    }
  })
})

/** TrueSkill as a factor graph: skills, performances, team sums, their difference, and a comparison with the draw margin. */
function trueskillGraph(
  winners: readonly { mu: number; sigma: number }[],
  losers: readonly { mu: number; sigma: number }[],
  draw: boolean,
): { spec: M.FactorGraphSpec; skills: string[] } {
  const sigma0 = 25 / 3
  const beta = sigma0 / 2
  const tau = sigma0 / 100
  const players = [...winners, ...losers]
  const epsilon = item(ndtri((0.1 + 1) / 2)) * Math.sqrt(players.length) * beta
  const variables: M.VariableSpec[] = []
  const factors: M.FactorSpec[] = []
  const skills: string[] = []
  for (const [i, p] of players.entries()) {
    variables.push({ name: `s${i}`, family: { kind: 'gaussian' } }, { name: `p${i}`, family: { kind: 'gaussian' } })
    skills.push(`s${i}`)
    factors.push(M.gaussianPrior(`s${i}`, p.mu, p.sigma ** 2 + tau ** 2), {
      operator: 'gaussian',
      args: [`p${i}`, `s${i}`],
      params: { precision: 1 / beta ** 2 },
    })
  }
  const team = (name: string, members: number[]): void => {
    variables.push({ name, family: { kind: 'gaussian' } })
    factors.push({ operator: 'sum', args: [name, ...members.map((m) => `p${m}`)], params: { n: members.length } })
  }
  team(
    'tw',
    winners.map((_, i) => i),
  )
  team(
    'tl',
    losers.map((_, i) => winners.length + i),
  )
  variables.push(
    { name: 'd', family: { kind: 'gaussian' } },
    { name: 'outcome', family: { kind: 'bernoulli' }, observed: 1 },
  )
  factors.push({ operator: 'difference', args: ['d', 'tw', 'tl'] })
  factors.push(
    draw
      ? { operator: 'is-between', args: ['outcome', 'd'], params: { lower: -epsilon, upper: epsilon } }
      : { operator: 'greater-than', args: ['outcome', 'd'], params: { threshold: epsilon } },
  )
  return { spec: { variables, factors }, skills }
}

describe('TrueSkill as a factor graph equals the update of pgm', () => {
  const a = { mu: 25, sigma: 25 / 3 }
  const b = { mu: 27, sigma: 5 }
  const c = { mu: 21, sigma: 7 }
  for (const [name, winners, losers, draw] of [
    ['one against one', [a], [b], false],
    ['two against one', [a, c], [b], false],
    ['a draw', [b], [c], true],
    ['a team draw', [a, b], [c], true],
  ] as const) {
    it(`${name}: the new skills`, () => {
      const { spec, skills } = trueskillGraph(winners, losers, draw)
      const g = M.compileGraph(spec, operators)
      const r = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 200 })
      expect(r.status).toBe('converged')
      const [w, l] = trueskillUpdate(winners, losers, draw)
      const want = [...(w ?? []), ...(l ?? [])]
      for (const [i, id] of skills.entries()) {
        const m = M.marginalsOf(g, r.state)[id] as Record<string, never>
        close(m['mean'] as unknown as number, want[i]?.mu as number, 1e-8)
        close(Math.sqrt(m['variance'] as unknown as number), want[i]?.sigma as number, 1e-8)
      }
    })
  }
})

describe('variational message passing on a Bayesian Gaussian mixture reproduces the library fit', () => {
  const data = [0.1, -0.2, 0.3, 0.0, 5.1, 4.8, 5.3, 5.0, 10.2, 9.7, 10.1, 9.9]
  const K = 3
  const x = data.map((v) => [v])
  const priors = variationalPriors(x, K)
  const start = data.map((_, i) => {
    const hard = Math.floor(i / 4)
    return [0, 1, 2].map((k) => (k === hard ? 0.8 : 0.1))
  })
  const n = data.length
  const spec: M.FactorGraphSpec = {
    variables: [
      { name: 'pi', family: { kind: 'dirichlet', size: K } },
      ...Array.from({ length: K }, (_, k): M.VariableSpec => ({ name: `theta${k}`, family: { kind: 'normal-gamma' } })),
      ...data.map((_, i): M.VariableSpec => ({ name: `z${i}`, family: { kind: 'discrete', size: K } })),
      ...data.map((v, i): M.VariableSpec => ({ name: `x${i}`, family: { kind: 'gaussian' }, observed: v })),
    ],
    factors: [
      M.dirichletPrior('pi', constantArray(K, priors.weightConcentration)),
      ...Array.from({ length: K }, (_, k) =>
        M.normalGammaPrior(
          `theta${k}`,
          priors.mean[0] as number,
          priors.meanPrecision,
          priors.degreesOfFreedom / 2,
          (priors.covariance[0]?.[0] as number) / 2,
        ),
      ),
      ...data.flatMap((_, i): M.FactorSpec[] => [
        { operator: 'discrete', args: [`z${i}`, 'pi'] },
        {
          operator: 'gaussian-mixture',
          args: [`x${i}`, `z${i}`, ...Array.from({ length: K }, (__, k) => `theta${k}`)],
          params: { components: K },
        },
      ]),
    ],
  }
  const g = M.compileGraph(spec, operators)
  const firstData = 1 + K
  const factorOf = (i: number): { discrete: number; mixture: number } => ({
    discrete: firstData + 2 * i,
    mixture: firstData + 2 * i + 1,
  })
  // Coordinate ascent: every parameter message from the responsibilities, then every assignment message.
  const order: M.CustomOrder = [
    ...Array.from({ length: 1 + K }, (_, f): [number, null] => [f, null]),
    ...data.flatMap((_, i): [number, number][] =>
      Array.from({ length: K }, (__, k): [number, number] => [factorOf(i).mixture, 2 + k]),
    ),
    ...data.map((_, i): [number, number] => [factorOf(i).discrete, 1]),
    ...data.map((_, i): [number, number] => [factorOf(i).mixture, 1]),
    ...data.map((_, i): [number, number] => [factorOf(i).discrete, 0]),
  ]
  const initial = start.map((r, i) => ({ factor: factorOf(i).mixture, port: 1, eta: M.discreteEta(r) }))
  const options = { schedule: 'custom' as const, order, initial, tol: 1e-12, maxSweeps: 500 }
  it('finds the posterior of the library fit', () => {
    const r = runAlgorithm(M.vmpAlgorithm(g, options), { maxSteps: 500 })
    expect(r.status).toBe('converged')
    const ref = runAlgorithm(
      variationalMixtureAlgorithm(x, start, priors, { weightPrior: 'dirichlet_distribution', tol: 1e-13 }),
      { maxSteps: 1000 },
    )
    const post = ref.state.posterior
    const marg = M.marginalsOf(g, r.state)
    const alpha = marg['pi']?.['alpha'] as number[]
    const ours = Array.from({ length: K }, (_, k) => marg[`theta${k}`] as Record<string, number>)
    for (let k = 0; k < K; k++) {
      close(alpha[k] as number, (post.concentration as number[])[k] as number, 1e-6)
      close(ours[k]?.['beta'] as number, post.meanPrecision[k] as number, 1e-6)
      close(ours[k]?.['m'] as number, (post.means[k] as number[])[0] as number, 1e-6)
      close(2 * (ours[k]?.['a'] as number), post.degreesOfFreedom[k] as number, 1e-6)
      close(
        (ours[k]?.['b'] as number) / (ours[k]?.['a'] as number),
        ((post.covariances[k] as number[][])[0] as number[])[0] as number,
        1e-6,
      )
    }
    expect(n).toBe(12)
  })
  it('raises the variational bound at every sweep', () => {
    const alg = M.vmpAlgorithm(g, options)
    const s: M.EngineSettings = {
      projection: M.klqpProjection,
      schedule: 'custom',
      order,
      initial,
      granularity: 'sweep',
      damping: 0,
      improper: 'skip',
      tol: 1e-12,
      maxSweeps: 500,
    }
    let state = alg.init(undefined as never)
    let last = Number.NEGATIVE_INFINITY
    let rises = 0
    for (let sweep = 0; sweep < 12; sweep++) {
      state = alg.step(state, undefined as never)
      const bound = M.evidenceOf(g, s, state)
      expect(bound).toBeGreaterThanOrEqual(last - 1e-9)
      if (bound > last + 1e-6) rises++
      last = bound
    }
    expect(rises).toBeGreaterThan(2)
  })
})
