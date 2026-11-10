import { describe, expect, it } from 'vitest'
import { smoothSolvers } from '../../optim/index.ts'
import { key, normal, randint } from '../../random/index.ts'
import { asTensor, toFlat, type Value } from '../../tensor/index.ts'
import { iterate, runAlgorithm } from '../../trace/index.ts'
import {
  type CrfInput,
  crfDecode,
  crfExpectedFeatureCounts,
  crfFeatureCounts,
  crfObjective,
  crfScore,
  type LabelledSequence,
  linearChainCrf,
  withCrfParameters,
} from './crf.ts'
import { crfSmoothProblem, crfSolvers, crfTrainAlgorithm, trainCrf } from './crf-training.ts'

const labels = ['A', 'B', 'C']
const words = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5']

/** Tagging data labelled by a planted CRF's Viterbi decoding, so it is separable by construction. */
function planted(
  seed: number,
  sequences: number,
): { data: LabelledSequence[]; truth: ReturnType<typeof linearChainCrf> } {
  const base = linearChainCrf(labels, words)
  const w = Array.from(toFlat(normal(key(seed), [base.stateWeights.length + base.transitionWeights.length])))
  const truth = withCrfParameters(base, [
    ...w.slice(0, base.stateWeights.length).map((v) => 2 * v),
    ...w.slice(base.stateWeights.length).map((v) => 1.5 * v),
  ])
  const lengths = Array.from(toFlat(randint(key(seed + 1), [sequences], 4, 8)))
  const ids = Array.from(toFlat(randint(key(seed + 2), [sequences * 8], 0, words.length)))
  const data = lengths.map((len, n): LabelledSequence => {
    const x: CrfInput = Array.from({ length: len }, (_, t) => ({ [words[ids[n * 8 + t] ?? 0] ?? 'w0']: 1 }))
    return { x, y: crfDecode(truth, x).labels }
  })
  return { data, truth }
}

describe('CRF feature counts', () => {
  const crf = withCrfParameters(linearChainCrf(labels, ['f', 'g']), Array.from(toFlat(normal(key(3), [2 * 3 + 9]))))
  const x: readonly Record<string, number>[] = [{ f: 1, g: 2 }, { f: 1 }, { g: 1.5 }, { f: 2, g: 1 }]

  /** Every labelling with its probability: the exact expectation of the feature counts. */
  function enumerated(): { expected: Float64Array; logZ: number } {
    const all: string[][] = [[]]
    for (let t = 0; t < x.length; t++) {
      const next: string[][] = []
      for (const p of all.splice(0)) for (const l of labels) next.push([...p, l])
      all.push(...next)
    }
    const scores = all.map((y) => crfScore(crf, x, y))
    const m = Math.max(...scores)
    const logZ = m + Math.log(scores.reduce((a, s) => a + Math.exp(s - m), 0))
    const expected = new Float64Array(crfFeatureCounts(crf, x, all[0] ?? []).length)
    all.forEach((y, i) => {
      const p = Math.exp((scores[i] ?? 0) - logZ)
      for (const [q, v] of crfFeatureCounts(crf, x, y).entries()) expected[q] = (expected[q] ?? 0) + p * v
    })
    return { expected, logZ }
  }

  it('the score is the inner product of weights and counts; the expected counts are the enumerated expectation', () => {
    const y = ['A', 'C', 'C', 'B']
    const phi = crfFeatureCounts(crf, x, y)
    const w = [...crf.stateWeights, ...crf.transitionWeights]
    expect(phi.reduce((a, v, q) => a + v * (w[q] ?? 0), 0)).toBeCloseTo(crfScore(crf, x, y), 12)
    const truth = enumerated()
    const { counts, logPartition } = crfExpectedFeatureCounts(crf, x)
    expect(logPartition).toBeCloseTo(truth.logZ, 12)
    for (const [q, v] of counts.entries()) expect(v).toBeCloseTo(truth.expected[q] ?? Number.NaN, 12)
  })

  it('the objective gradient is the expected minus the observed counts, exactly', () => {
    const y = ['B', 'B', 'A', 'C']
    const g = crfObjective(crf, [{ x, y }]).gradient
    const e = crfExpectedFeatureCounts(crf, x).counts
    const phi = crfFeatureCounts(crf, x, y)
    for (const [q, v] of g.entries()) expect(v).toBeCloseTo((e[q] ?? 0) - (phi[q] ?? 0), 12)
  })
})

describe('the CRF objective as a smooth problem', () => {
  const { data } = planted(5, 6)
  const crf = linearChainCrf(labels, words)

  it('equals crfObjective with its gradient; batches are unbiased and the full batch is exact', () => {
    const problem = crfSmoothProblem({ crf, data, penalty: { c2: 0.05 } })
    const w = Array.from(toFlat(normal(key(9), [crfObjective(crf, []).gradient.length]))).map((v) => 0.3 * v)
    const full = crfObjective(withCrfParameters(crf, w), data, { c2: 0.05 })
    const f = problem.objective as (x: unknown, batch?: readonly number[]) => Value
    expect(problem.batches?.n).toBe(6)
    // the value through the tensor wrapper
    const value = (b?: readonly number[]): number => toFlat(f(asTensor(Float64Array.from(w)), b))[0] ?? Number.NaN
    expect(value()).toBeCloseTo(full.value, 10)
    expect(value([0, 1, 2, 3, 4, 5])).toBeCloseTo(full.value, 10)
    // averaged over all singleton batches the stochastic value is the full one
    const mean = [0, 1, 2, 3, 4, 5].reduce((a, i) => a + value([i]), 0) / 6
    expect(mean).toBeCloseTo(full.value, 9)
  })
})

describe('structured perceptron and friends through the slot', () => {
  const { data } = planted(21, 24)
  const crf = linearChainCrf(labels, words)
  const problem = { crf, data, penalty: { c2: 0 } }
  const run = (solver: object, steps = 200) =>
    runAlgorithm(crfTrainAlgorithm(problem, { solver: solver as never }), { maxSteps: steps, key: key(0) })

  it('the planted tagging data is separable: the perceptron and passive-aggressive reach zero sequence error', () => {
    for (const kind of ['perceptron', 'passive-aggressive']) {
      const out = run({ kind })
      expect(out.status).toBe('converged')
      expect(out.state.sequenceError).toBe(0)
      expect(out.steps).toBeGreaterThan(1)
    }
  })

  it('the averaged perceptron stops when the current weights make no mistake, and its average decodes most of the data', () => {
    const out = run({ kind: 'averaged' })
    expect(out.status).toBe('converged')
    expect(out.state.mistakes).toBe(0)
    expect(out.state.sequenceError).toBeLessThanOrEqual(0.25)
  })

  it('the mistakes of the perceptron fall to zero and the objective is reported on the tape', () => {
    const alg = crfTrainAlgorithm(problem, { solver: { kind: 'perceptron' } })
    const mistakes: number[] = []
    const objective: number[] = []
    const sequenceErrors: number[] = []
    for (const { state } of iterate(alg, { key: key(0), maxSteps: 200 })) {
      mistakes.push(state.mistakes)
      objective.push(state.objective)
      sequenceErrors.push(state.sequenceError)
    }
    // Novikoff: finitely many mistakes on separable data, so the epochs' mistakes end, and the first epoch made many
    const epochs = mistakes.slice(1)
    expect(epochs[0] ?? 0).toBeGreaterThan(5)
    expect(epochs.at(-1) ?? 0).toBeLessThan(epochs[0] ?? 0)
    expect(sequenceErrors.at(-1)).toBe(0)
    expect(objective.every(Number.isFinite)).toBe(true)
  })

  it('the exact solvers (L-BFGS, BFGS, conjugate gradient) agree on the penalised optimum; the online modes do not minimise it', () => {
    const penalised = { ...problem, penalty: { c2: 0.2 } }
    const fit = (kind: string) =>
      runAlgorithm(
        crfTrainAlgorithm(penalised, {
          solver: { kind, gradTol: 1e-10, ...(kind === 'lbfgs' ? { ftol: 1e-16 } : {}) } as never,
          tol: 1e-8,
        }),
        {
          maxSteps: 600,
          key: key(0),
        },
      ).state
    const lbfgs = fit('lbfgs')
    expect(lbfgs.gradient).toBeLessThan(1e-6)
    for (const kind of ['bfgs', 'conjugateGradient']) {
      const other = fit(kind)
      expect(other.objective).toBeCloseTo(lbfgs.objective, 7)
    }
    const perceptron = runAlgorithm(crfTrainAlgorithm(penalised, { solver: { kind: 'perceptron' } }), {
      maxSteps: 100,
      key: key(0),
    }).state
    expect(perceptron.objective).toBeGreaterThan(lbfgs.objective)
  })

  it("stochastic gradient (the slot's gradient mode with Adam and batches) approaches the exact optimum", () => {
    const penalised = { ...problem, penalty: { c2: 0.2 } }
    const exact = runAlgorithm(crfTrainAlgorithm(penalised, { tol: 1e-8 }), { maxSteps: 600, key: key(0) }).state
    const sgd = runAlgorithm(
      crfTrainAlgorithm(penalised, {
        solver: {
          kind: 'gradient',
          optimiser: { kind: 'adam' },
          learningRate: 0.05,
          batch: 6,
          schedule: { kind: 'constant' },
          gradTol: 1e-9,
        } as never,
        key: key(1),
      }),
      { maxSteps: 400, key: key(1) },
    ).state
    expect(sgd.objective).toBeLessThan(exact.objective * 1.1)
    expect(sgd.sequenceError).toBeLessThanOrEqual(exact.sequenceError + 0.2)
  })

  it('the table lists the online modes beside the generic solvers, with L-BFGS the default', () => {
    expect(crfSolvers.names).toEqual(
      expect.arrayContaining(['perceptron', 'averaged', 'passive-aggressive', 'lbfgs', 'gradient', 'newton']),
    )
    expect(crfSolvers.modeOf({}).name).toBe('lbfgs')
    void smoothSolvers
  })

  it('trainCrf fits a model that decodes the training data, and rejects empty data', () => {
    const fitted = trainCrf(crf, data, { c2: 0.001 }, 300)
    expect(fitted.sequenceError).toBeLessThan(0.05)
    expect(fitted.objective).toBeLessThan(5)
    expect(() => crfTrainAlgorithm({ crf, data: [] })).toThrow(/no training/)
  })
})
