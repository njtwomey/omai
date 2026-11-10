import { describe, expect, it } from 'vitest'
import { analyticObjective, smoothSolvers } from '../../optim/index.ts'
import { key, normal, randint } from '../../random/index.ts'
import { asTensor, toFlat, type Value } from '../../tensor/index.ts'
import { iterate, runAlgorithm } from '../../trace/index.ts'
import { chainOracle } from './chain-oracle.ts'
import { compileChainData } from './compiled.ts'
import { crfDecode, crfObjective, type LabelledSequence, linearChainCrf, withCrfParameters } from './crf.ts'
import { crfSmoothProblem, crfSolvers, crfTrainAlgorithm, trainCrf } from './crf-training.ts'
import { structuredObjective } from './objective.ts'
import {
  overStructuredProblem,
  type StructuredMode,
  type StructuredOnlineState,
  structuredErrors,
  structuredModes,
  structuredSolvers,
  structuredSolversWith,
  structuredTrainAlgorithm,
  trainStructured,
} from './solvers.ts'

const labels = ['A', 'B', 'C']
const words = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5']

/** Tagging data labelled by a planted CRF's Viterbi decoding, so it is separable by construction. */
function planted(seed: number, sequences: number): LabelledSequence[] {
  const base = linearChainCrf(labels, words)
  const w = Array.from(toFlat(normal(key(seed), [base.stateWeights.length + base.transitionWeights.length])))
  const truth = withCrfParameters(base, [
    ...w.slice(0, base.stateWeights.length).map((v) => 2 * v),
    ...w.slice(base.stateWeights.length).map((v) => 1.5 * v),
  ])
  const lengths = Array.from(toFlat(randint(key(seed + 1), [sequences], 4, 8)))
  const ids = Array.from(toFlat(randint(key(seed + 2), [sequences * 8], 0, words.length)))
  return lengths.map((len, n): LabelledSequence => {
    const x = Array.from({ length: len }, (_, t) => ({ [words[ids[n * 8 + t] ?? 0] ?? 'w0']: 1 }))
    return { x, y: crfDecode(truth, x).labels }
  })
}

/** Sparse data: only `w0` and `w1` decide the label (A or B); the other attributes are noise on every token. */
function sparse(seed: number, sequences: number): { data: LabelledSequence[]; noise: string[] } {
  const noise = Array.from({ length: 10 }, (_, i) => `n${i}`)
  const ids = Array.from(toFlat(randint(key(seed), [sequences * 6 * 3], 0, 10)))
  const signal = Array.from(toFlat(randint(key(seed + 1), [sequences * 6], 0, 2)))
  const data = Array.from({ length: sequences }, (_, n): LabelledSequence => {
    const x = Array.from({ length: 6 }, (_, t) => {
      const item: Record<string, number> = { [`w${signal[n * 6 + t]}`]: 1 }
      for (let f = 0; f < 3; f++) item[`n${ids[(n * 6 + t) * 3 + f]}`] = 1
      return item
    })
    return { x, y: x.map((item) => (item['w0'] === 1 ? 'A' : 'B')) }
  })
  return { data, noise }
}

describe('the generic training objective over the chain oracle', () => {
  const data = planted(5, 8)
  const crf = linearChainCrf(labels, words)
  const oracle = chainOracle(crf)
  const compiled = compileChainData(crf, data)

  it('L-BFGS through the structured slot reaches the optimum of the old crfObjective path to 1e-8', () => {
    const penalty = { c2: 0.2 }
    const old = analyticObjective({
      value: (w) => crfObjective(withCrfParameters(crf, w), data, penalty).value,
      gradient: (w) => crfObjective(withCrfParameters(crf, w), data, penalty).gradient,
    })
    const solver = { kind: 'lbfgs', gradTol: 1e-10, ftol: 1e-16 }
    const x0 = new Array<number>(oracle.size).fill(0)
    const reference = runAlgorithm(smoothSolvers.make({ objective: old, x0 }, solver, key(0)), {
      maxSteps: 600,
      key: key(0),
    }).state as { x: Float64Array; value: number }
    const fitted = trainStructured({ oracle, data: compiled, penalty }, { solver: solver as never, tol: 1e-9 }, 600)
    expect(fitted.objective).toBeCloseTo(reference.value, 8)
    // the compiled and the plain data train to the same point
    const plain = trainStructured({ oracle, data, penalty }, { solver: solver as never, tol: 1e-9 }, 600)
    for (const [q, v] of fitted.weights.entries()) {
      expect(v).toBeCloseTo(reference.x[q] ?? Number.NaN, 6)
      expect(plain.weights[q]).toBeCloseTo(v, 6)
    }
  })

  it('the CRF smooth problem honours c1 by its subgradient and the structured objective agrees', () => {
    const w = Array.from({ length: oracle.size }, (_, q) => 0.3 * Math.cos(q))
    const penalty = { c2: 0.05, c1: 0.2 }
    const problem = crfSmoothProblem({ crf, data, penalty })
    const f = problem.objective as (x: unknown) => Value
    const want = crfObjective(withCrfParameters(crf, w), data, penalty).value
    expect(toFlat(f(asTensor(Float64Array.from(w))))[0]).toBeCloseTo(want, 10)
    expect(structuredObjective(oracle, compiled, w, penalty).value).toBeCloseTo(want, 10)
  })
})

describe('OWL-QN for the L1 penalty of a CRF', () => {
  const { data } = sparse(31, 40)
  const attributes = ['w0', 'w1', ...Array.from({ length: 10 }, (_, i) => `n${i}`)]
  const crf = linearChainCrf(['A', 'B'], attributes)
  const oracle = chainOracle(crf)
  const compiled = compileChainData(crf, data)
  const penalty = { c1: 1, c2: 0.01 }

  const solve = (solver: object, steps = 400) =>
    runAlgorithm(
      structuredTrainAlgorithm({ oracle, data: compiled, penalty }, { solver: solver as never, tol: 1e-5 }),
      {
        maxSteps: steps,
        key: key(0),
      },
    )

  it('is the default for a problem with c1, and gives exact zeros and a sparse model that still decodes the data', () => {
    const run = runAlgorithm(structuredTrainAlgorithm({ oracle, data: compiled, penalty }, { tol: 1e-5 }), {
      maxSteps: 400,
      key: key(0),
    })
    expect(run.status).toBe('converged')
    const w = run.state.weights
    const zeros = w.filter((v) => v === 0).length
    expect(zeros).toBeGreaterThan(oracle.size / 2)
    expect(run.state.nonzeros).toBe(oracle.size - zeros)
    expect(run.state.error).toBe(0)
    // the signal attributes carry the weights: w0 pulls A, w1 pulls B
    const w0 = (w[0] ?? 0) - (w[1] ?? 0)
    const w1 = (w[2] ?? 0) - (w[3] ?? 0)
    expect(w0).toBeGreaterThan(0)
    expect(w1).toBeLessThan(0)
    // the noise attributes are all at zero
    for (let a = 2; a < attributes.length; a++) expect([w[2 * a], w[2 * a + 1]]).toEqual([0, 0])
  })

  it('satisfies the L1 optimality conditions and beats L-BFGS on the subgradient objective', () => {
    const owl = solve({ kind: 'owlqn' }).state
    const smooth = structuredObjective(oracle, compiled, owl.weights, { c2: penalty.c2 })
    for (const [q, w] of owl.weights.entries()) {
      const g = smooth.gradient[q] ?? 0
      if (w === 0) expect(Math.abs(g)).toBeLessThanOrEqual(penalty.c1 + 1e-4)
      else expect(g + penalty.c1 * Math.sign(w)).toBeCloseTo(0, 3)
    }
    // L-BFGS on the old subgradient objective cannot reach exact zeros and does no better on the penalised objective
    const sub = solve({ kind: 'lbfgs' }, 200).state
    expect(sub.weights.filter((v) => v === 0).length).toBeLessThan(owl.weights.filter((v) => v === 0).length)
    expect(owl.objective).toBeLessThanOrEqual(sub.objective + 1e-6)
  })

  it('trainCrf honours c1 through the default solver, and without c1 OWL-QN is L-BFGS on the L2 problem', () => {
    const sparseFit = trainCrf(crf, data, { c1: 1, c2: 0.01 }, 400)
    expect(Array.from(sparseFit.crf.stateWeights).filter((v) => v === 0).length).toBeGreaterThan(10)
    const l2 = { c2: 0.2 }
    const a = trainStructured(
      { oracle, data: compiled, penalty: l2 },
      { solver: { kind: 'owlqn' } as never, tol: 1e-8 },
    )
    const b = trainStructured({ oracle, data: compiled, penalty: l2 }, { tol: 1e-8 })
    expect(a.objective).toBeCloseTo(b.objective, 6)
    // the same through crfSolvers
    expect(crfSolvers.names).toContain('owlqn')
  })
})

describe('online structured learners over any oracle', () => {
  const data = planted(21, 24)
  const crf = linearChainCrf(labels, words)
  const oracle = chainOracle(crf)
  const compiled = compileChainData(crf, data)
  const run = (solver: object, rows: typeof compiled | typeof data = compiled, steps = 200) =>
    runAlgorithm(structuredTrainAlgorithm({ oracle, data: rows }, { solver: solver as never }), {
      maxSteps: steps,
      key: key(0),
    })

  it('the perceptron, passive-aggressive and AROW reach zero error on the separable data', () => {
    // AROW with its default knobs (variance 1, r 1) is a conservative learner and keeps a few mistakes for long; a large
    // initial variance and a small r make it aggressive enough to separate this data
    for (const solver of [
      { kind: 'perceptron' },
      { kind: 'passive-aggressive' },
      { kind: 'arow', variance: 100, gamma: 0.01 },
    ]) {
      const kind = solver.kind
      const out = run(solver)
      expect([kind, out.status, out.state.error, out.steps]).toEqual([kind, 'converged', 0, out.steps])
      expect(out.state.loss).toBe(0)
      expect(out.steps).toBeGreaterThan(1)
    }
  })

  it('compiled and plain data follow the same updates', () => {
    for (const kind of ['perceptron', 'arow']) {
      const a = run({ kind })
      const b = run({ kind }, data)
      expect(b.steps).toBe(a.steps)
      for (const [q, v] of a.state.weights.entries()) expect(b.state.weights[q]).toBeCloseTo(v, 10)
    }
  })

  it('the averaged perceptron reports the exact average of the weights after every example', () => {
    const epoch = (kind: string): StructuredOnlineState =>
      structuredSolvers
        .make({ oracle, data: compiled }, { kind, shuffle: false }, key(0))
        .step(
          structuredSolvers.make({ oracle, data: compiled }, { kind, shuffle: false }, key(0)).init(key(0)) as never,
          { key: key(0) } as never,
        ) as StructuredOnlineState
    const averaged = epoch('averaged')
    const plain = epoch('perceptron')
    // replay the perceptron by hand and average its weights over the examples
    const w = new Float64Array(oracle.size)
    const sum = new Float64Array(oracle.size)
    for (const { x, y } of compiled) {
      const found = oracle.decode(w, x)
      if (oracle.loss(y, found.y) !== 0) {
        const truth = oracle.features(x, y)
        const guess = oracle.features(x, found.y)
        for (let q = 0; q < w.length; q++) w[q] = (w[q] ?? 0) + (truth[q] ?? 0) - (guess[q] ?? 0)
      }
      for (let q = 0; q < w.length; q++) sum[q] = (sum[q] ?? 0) + (w[q] ?? 0)
    }
    for (const [q, v] of w.entries()) {
      expect(plain.x[q]).toBeCloseTo(v, 12)
      expect(averaged.w[q]).toBeCloseTo(v, 12)
      expect(averaged.x[q]).toBeCloseTo((sum[q] ?? 0) / compiled.length, 10)
    }
  })

  it('AROW shrinks the variance of the weights it moves and leaves the others at the start', () => {
    const algorithm = structuredSolvers.make(
      { oracle, data: compiled },
      { kind: 'arow', variance: 2, gamma: 1 },
      key(0),
    )
    const state = algorithm.step(algorithm.init(key(0)) as never, { key: key(0) } as never) as StructuredOnlineState
    expect(state.variance.every((v) => v <= 2)).toBe(true)
    expect(state.variance.some((v) => v < 2)).toBe(true)
    expect(state.mistakes).toBeGreaterThan(0)
  })

  it('the mistakes fall over the epochs and the penalised objective is on the tape for each', () => {
    const alg = structuredTrainAlgorithm(
      { oracle, data: compiled, penalty: { c2: 0.01 } },
      { solver: { kind: 'passive-aggressive' } },
    )
    const mistakes: number[] = []
    for (const { state } of iterate(alg, { key: key(0), maxSteps: 200 })) {
      mistakes.push(state.mistakes)
      expect(Number.isFinite(state.objective)).toBe(true)
    }
    expect(mistakes.at(-1) ?? 1).toBeLessThan(mistakes[1] ?? 0)
  })

  it('structuredErrors counts the loss of decoding', () => {
    const zero = structuredErrors({ oracle, data: compiled }, new Float64Array(oracle.size))
    expect(zero.error).toBeGreaterThan(0)
    expect(zero.loss).toBeGreaterThan(0)
  })
})

const perceptronMode = ((): StructuredMode => {
  const mode = structuredModes['perceptron']
  if (mode === undefined) throw new Error('no perceptron mode')
  return mode
})()

describe('the table of structured solvers', () => {
  const crf = linearChainCrf(labels, words)
  const oracle = chainOracle(crf)
  const data = compileChainData(crf, planted(7, 6))

  it('lists the generic solvers, OWL-QN and the online modes with L-BFGS the default', () => {
    expect(structuredSolvers.names).toEqual(
      expect.arrayContaining([
        'lbfgs',
        'bfgs',
        'newton',
        'gradient',
        'owlqn',
        'perceptron',
        'averaged',
        'passive-aggressive',
        'arow',
      ]),
    )
    expect(structuredSolvers.modeOf({}).name).toBe('lbfgs')
    expect(crfSolvers.names).toEqual(expect.arrayContaining(Object.keys(structuredModes)))
  })

  it('takes the modes of another family beside its own', () => {
    const table = structuredSolversWith({ always: { ...perceptronMode, description: 'a family mode' } })
    expect(table.names).toContain('always')
    expect(table.modeOf({ kind: 'always' }).description).toBe('a family mode')
    // and maps a family's own problem onto the structured one
    const over = overStructuredProblem({ perceptron: perceptronMode }, (p: { rows: typeof data }) => ({
      oracle,
      data: p.rows,
    }))
    expect(Object.keys(over)).toEqual(['perceptron'])
  })

  it('rejects empty data and trains through trainStructured to the same fit as the CRF wrapper', () => {
    expect(() => structuredTrainAlgorithm({ oracle, data: [] })).toThrow(/no training/)
    const plain = planted(7, 6)
    const a = trainStructured({ oracle, data, penalty: { c2: 0.1 } }, {}, 300)
    const b = trainCrf(crf, plain, { c2: 0.1 }, 300)
    expect(a.objective).toBeCloseTo(b.objective, 8)
    expect(a.error).toBe(b.sequenceError)
    expect(a.converged).toBe(true)
    const tape = runAlgorithm(crfTrainAlgorithm({ crf, data: plain, penalty: { c2: 0.1 } }), {
      maxSteps: 5,
      key: key(0),
    })
    expect(Object.keys(tape.state)).toEqual(expect.arrayContaining(['sequenceError', 'loss', 'nonzeros', 'mistakes']))
  })
})
