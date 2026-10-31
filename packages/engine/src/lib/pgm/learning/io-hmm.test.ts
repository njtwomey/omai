import { describe, expect, it } from 'vitest'
import { smoothSolvers } from '../../optim/index.ts'
import { key } from '../../random/index.ts'
import { iterate, runAlgorithm as run } from '../../trace/index.ts'
import { forwardBackward } from '../inference/chain.ts'
import { baumWelchAlgorithm, type HmmParameters } from './baum-welch.ts'
import {
  type IoHmmParameters,
  ioHmmAlgorithm,
  ioHmmChain,
  ioHmmLogLikelihood,
  ioHmmSample,
  ioTransition,
  softmaxRegressionProblem,
} from './io-hmm.ts'

// A two-state model with a scalar input: the input pushes the transition towards state 1, and the emission of state 0
// depends on it too.
const tiny: IoHmmParameters = {
  kind: 'categorical',
  start: [0.6, 0.4],
  transition: [
    [
      [0.5, 1.2],
      [-0.3, -0.8],
    ],
    [
      [0.1, -1],
      [0.4, 0.9],
    ],
  ],
  emission: [
    [
      [0.3, 0.5],
      [-0.2, -0.4],
      [0, 0],
    ],
    [
      [-0.5, 0],
      [0.6, 0.3],
      [0.1, -0.2],
    ],
  ],
}

/** The marginal probability of the observations by summing over every state path. */
function brute(
  p: IoHmmParameters & { kind: 'categorical' },
  inputs: number[],
  x: number[],
): { total: number; marginals: number[][] } {
  const K = 2
  const T = x.length
  const emit = (k: number, t: number): number => {
    const z = (p.emission[k] ?? []).map((w) => (w[0] ?? 0) + (w[1] ?? 0) * (inputs[t] ?? 0))
    const e = z.map(Math.exp)
    return (e[x[t] ?? 0] ?? 0) / e.reduce((a, v) => a + v, 0)
  }
  let total = 0
  const marginals = Array.from({ length: T }, () => new Array<number>(K).fill(0))
  const paths = K ** T
  for (let code = 0; code < paths; code++) {
    const z = Array.from({ length: T }, (_, t) => Math.floor(code / K ** t) % K)
    let prob = (p.start[z[0] ?? 0] ?? 0) * emit(z[0] ?? 0, 0)
    for (let t = 1; t < T; t++)
      prob *= (ioTransition(p, [inputs[t] ?? 0])[z[t - 1] ?? 0]?.[z[t] ?? 0] ?? 0) * emit(z[t] ?? 0, t)
    total += prob
    for (const [t, k] of z.entries()) (marginals[t] as number[])[k] = (marginals[t]?.[k] ?? 0) + prob
  }
  return { total, marginals: marginals.map((r) => r.map((v) => v / total)) }
}

describe('input-output HMM inference', () => {
  it('matches the sum over all state paths: likelihood and posteriors', () => {
    const inputs = [0.5, -1, 2, 0, 1.5]
    const x = [0, 1, 2, 1, 0]
    const truth = brute(tiny as IoHmmParameters & { kind: 'categorical' }, inputs, x)
    const seq = { inputs: inputs.map((u) => [u]), observations: x }
    const fb = forwardBackward(ioHmmChain(tiny, seq))
    expect(fb.logLikelihood).toBeCloseTo(Math.log(truth.total), 12)
    for (let t = 0; t < x.length; t++)
      for (let k = 0; k < 2; k++) expect(fb.posteriors[t]?.[k]).toBeCloseTo(truth.marginals[t]?.[k] ?? Number.NaN, 12)
    expect(ioHmmLogLikelihood(tiny, [seq, seq])).toBeCloseTo(2 * Math.log(truth.total), 11)
  })

  it('with intercept-only features it is the hidden Markov model of the same probabilities', () => {
    const hmm: HmmParameters = {
      kind: 'categorical',
      start: [0.6, 0.4],
      transition: [
        [0.7, 0.3],
        [0.2, 0.8],
      ],
      emission: [
        [0.5, 0.3, 0.2],
        [0.1, 0.1, 0.8],
      ],
    }
    const io: IoHmmParameters = {
      kind: 'categorical',
      start: hmm.start,
      transition: hmm.transition.map((r) => r.map((v) => [Math.log(v)])),
      emission: (hmm as Extract<HmmParameters, { kind: 'categorical' }>).emission.map((r) =>
        r.map((v) => [Math.log(v)]),
      ),
    }
    const x = [0, 2, 2, 1, 0, 2]
    const seq = { inputs: x.map(() => []), observations: x }
    const reference = baumWelchAlgorithm([x], hmm)
    const first = reference.step(reference.init(key(0)), { t: 0, key: key(0) } as never)
    expect(ioHmmLogLikelihood(io, [seq])).toBeCloseTo(first.logLikelihood ?? Number.NaN, 11)
    // one EM step with a vanishing ridge gives the Baum-Welch update
    const out = run(ioHmmAlgorithm([seq], io, { ridge: 1e-9 }), { maxSteps: 1, key: key(0) }).state.params
    const bw = first.params as Extract<HmmParameters, { kind: 'categorical' }>
    for (let i = 0; i < 2; i++) {
      expect(ioTransition(out, [])[i]?.[0]).toBeCloseTo(bw.transition[i]?.[0] ?? Number.NaN, 6)
      expect(out.start[i]).toBeCloseTo(bw.start[i] ?? Number.NaN, 12)
    }
  })
})

describe('the softmax M-step problem', () => {
  const f = [
    [1, 0.5],
    [1, -1],
    [1, 2],
    [1, 0],
  ]
  const targets = [
    [0.7, 0.2, 0.1],
    [0.1, 0.6, 0.3],
    [0.5, 0.5, 0],
    [0.2, 0.2, 0.9],
  ]
  const problem = softmaxRegressionProblem(f, targets, 0.1)

  it('is minimised at the empirical proportions when the only feature is the intercept (ridge near zero)', () => {
    const p = softmaxRegressionProblem(
      [[1], [1]],
      [
        [3, 1],
        [1, 1],
      ],
      1e-12,
    )
    const out = run(smoothSolvers.make(p, { kind: 'newton', gradTol: 1e-12 }, key(0)), { maxSteps: 100, key: key(0) })
    const x = (out.state as { x: Float64Array }).x
    // total counts (4, 2): the probability of class 0 is 2/3
    expect(Math.exp(x[0] ?? 0) / (Math.exp(x[0] ?? 0) + Math.exp(x[1] ?? 0))).toBeCloseTo(2 / 3, 8)
  })

  it('Newton, L-BFGS and BFGS reach the same optimum (the solver is a slot)', () => {
    const optimum = (solver: object): { x: number[]; value: number } => {
      const out = run(smoothSolvers.make(problem, solver as never, key(0)), { maxSteps: 500, key: key(0) })
      const s = out.state as { x: Float64Array; value: number }
      return { x: Array.from(s.x), value: s.value }
    }
    const newton = optimum({ kind: 'newton', gradTol: 1e-11 })
    for (const solver of [
      { kind: 'lbfgs', gradTol: 1e-10 },
      { kind: 'bfgs', gradTol: 1e-10 },
    ]) {
      const other = optimum(solver)
      expect(other.value).toBeCloseTo(newton.value, 9)
      for (const [i, v] of other.x.entries()) expect(v).toBeCloseTo(newton.x[i] ?? Number.NaN, 5)
    }
  })
})

describe('input-output HMM EM', () => {
  const truth: IoHmmParameters = {
    kind: 'categorical',
    start: [0.5, 0.5],
    // the input u in {-1, +1} decides whether state 0 is sticky (u = 1) or leaves (u = -1)
    transition: [
      [
        [1.5, 2.5],
        [0, 0],
      ],
      [
        [1.5, -2.5],
        [0, 0],
      ],
    ],
    emission: [
      [
        [2, 0],
        [0, 0],
        [-1, 0],
      ],
      [
        [-1, 0],
        [0, 0],
        [2, 0],
      ],
    ],
  }
  const inputsFor = (seed: number, T: number): number[][] =>
    Array.from({ length: T }, (_, t) => [Math.sin((t + 1) * 12.9898 + seed * 78.233) > 0 ? 1 : -1])
  const sequences = [0, 1, 2].map((s) => {
    const inputs = inputsFor(s, 200)
    return { inputs, observations: ioHmmSample(truth, inputs, key(20 + s)).observations }
  })
  const start: IoHmmParameters = {
    kind: 'categorical',
    start: [0.5, 0.5],
    transition: [
      [
        [0.3, 0.3],
        [0, 0],
      ],
      [
        [0.3, -0.3],
        [0, 0],
      ],
    ],
    emission: [
      [
        [1, 0],
        [0, 0],
        [-0.5, 0],
      ],
      [
        [-0.5, 0],
        [0, 0],
        [1, 0],
      ],
    ],
  }

  it('raises the penalised log likelihood at every step, and Newton and L-BFGS give the same fit', () => {
    const trace = (solver: { kind: string; gradTol: number }): { objective: number[]; ll: number } => {
      const alg = ioHmmAlgorithm(sequences, start, { solver: solver as never, tol: -1 })
      const objective: number[] = []
      let ll = 0
      for (const { state } of iterate(alg, { key: key(0), maxSteps: 6 })) {
        if (state.objective !== null) objective.push(state.objective)
        ll = state.logLikelihood ?? ll
      }
      return { objective, ll }
    }
    const newton = trace({ kind: 'newton', gradTol: 1e-10 })
    for (let t = 1; t < newton.objective.length; t++)
      expect(newton.objective[t] ?? 0).toBeGreaterThanOrEqual((newton.objective[t - 1] ?? 0) - 1e-8)
    expect((newton.objective.at(-1) ?? 0) - (newton.objective[0] ?? 0)).toBeGreaterThan(3)
    const lbfgs = trace({ kind: 'lbfgs', gradTol: 1e-9 })
    expect(Math.abs(lbfgs.ll - newton.ll)).toBeLessThan(0.01)
  }, 120_000)

  it('recovers planted input-dependent transitions', () => {
    const fit = run(ioHmmAlgorithm(sequences, start, { ridge: 1e-3, tol: 1e-5 }), { maxSteps: 30, key: key(0) }).state
      .params
    // the probability of staying in state 0 at u = +1 and u = -1, planted 0.98 and 0.27 (logit 4 and 1, a softmax over two next states)
    const stay = (p: IoHmmParameters, u: number): number => ioTransition(p, [u])[0]?.[0] ?? Number.NaN
    expect(Math.abs(stay(fit, 1) - stay(truth, 1))).toBeLessThan(0.08)
    expect(Math.abs(stay(fit, -1) - stay(truth, -1))).toBeLessThan(0.1)
    const leave1 = (p: IoHmmParameters, u: number): number => ioTransition(p, [u])[1]?.[1] ?? Number.NaN
    expect(Math.abs(leave1(fit, 1) - leave1(truth, 1))).toBeLessThan(0.1)
  }, 120_000)

  it('Gaussian emissions are regressions on the input: the planted slopes and variances are recovered', () => {
    const g: IoHmmParameters = {
      kind: 'gaussian',
      start: [0.5, 0.5],
      transition: [
        [
          [1.5, 0],
          [-1.5, 0],
        ],
        [
          [-1.5, 0],
          [1.5, 0],
        ],
      ],
      coefficients: [[[-3, 1.5]], [[3, -1]]],
      variances: [[0.25], [0.25]],
    }
    const inputs = (seed: number): number[][] =>
      Array.from({ length: 300 }, (_, t) => [Math.sin((t + 1) * 4.1 + seed) * 2])
    const data = [0, 1].map((s) => ({
      inputs: inputs(s),
      observations: ioHmmSample(g, inputs(s), key(50 + s)).observations,
    }))
    const s0: IoHmmParameters = { ...g, coefficients: [[[-2, 0]], [[2, 0]]], variances: [[1], [1]] } as IoHmmParameters
    const fit = run(ioHmmAlgorithm(data, s0, { ridge: 1e-6 }), { maxSteps: 40, key: key(0) }).state.params
    if (fit.kind !== 'gaussian') throw new Error('gaussian expected')
    expect(fit.coefficients[0]?.[0]?.[0]).toBeCloseTo(-3, 1)
    expect(fit.coefficients[0]?.[0]?.[1]).toBeCloseTo(1.5, 1)
    expect(fit.coefficients[1]?.[0]?.[0]).toBeCloseTo(3, 1)
    expect(fit.coefficients[1]?.[0]?.[1]).toBeCloseTo(-1, 1)
    expect(Math.abs((fit.variances[0]?.[0] ?? 0) - 0.25)).toBeLessThan(0.06)
  })

  it('is deterministic and rejects mismatched inputs', () => {
    expect(ioHmmSample(truth, inputsFor(0, 10), key(1))).toEqual(ioHmmSample(truth, inputsFor(0, 10), key(1)))
    expect(() => ioHmmChain(truth, { inputs: [[1]], observations: [0, 1] })).toThrow(/one input/)
  })
})
