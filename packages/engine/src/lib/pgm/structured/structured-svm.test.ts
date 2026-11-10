import { describe, expect, it } from 'vitest'
import { qpArrays, qpInteriorPoint } from '../../optim/index.ts'
import { bernoulli, key, normal, randint } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { iterate } from '../../trace/index.ts'
import { chainOracle } from './chain-oracle.ts'
import { compileChainData } from './compiled.ts'
import { crfDecode, type LabelledSequence, linearChainCrf, withCrfParameters } from './crf.ts'
import type { GraphInput, GraphLabelling } from './graph-crf.ts'
import { graphOracle } from './graph-oracle.ts'
import { treeCrf, treeInput } from './graph-structures.ts'
import type { StructuredOracle, StructuredProblem } from './oracle.ts'
import { structuredErrors, structuredSolversWith, trainStructured } from './solvers.ts'
import {
  structuredHingeObjective,
  structuredSvmAlgorithm,
  structuredSvmModes,
  structuredSvmPenalty,
  structuredSvmSolvers,
  trainStructuredSvm,
} from './structured-svm.ts'

const labels = ['A', 'B', 'C']
const words = ['w0', 'w1', 'w2', 'w3', 'w4']

/** Tagging data labelled by a planted CRF's Viterbi decoding (separable); `flip` of the labels are then replaced. */
function planted(seed: number, sequences: number, flip = 0): LabelledSequence[] {
  const base = linearChainCrf(labels, words)
  const w = Array.from(toFlat(normal(key(seed), [base.stateWeights.length + base.transitionWeights.length])))
  const truth = withCrfParameters(base, [
    ...w.slice(0, base.stateWeights.length).map((v) => 2 * v),
    ...w.slice(base.stateWeights.length).map((v) => 1.5 * v),
  ])
  const lengths = Array.from(toFlat(randint(key(seed + 1), [sequences], 3, 6)))
  const ids = Array.from(toFlat(randint(key(seed + 2), [sequences * 8], 0, words.length)))
  const noise = Array.from(toFlat(bernoulli(key(seed + 3), flip, [sequences * 8])))
  const other = Array.from(toFlat(randint(key(seed + 4), [sequences * 8], 0, labels.length)))
  return lengths.map((len, n): LabelledSequence => {
    const x = Array.from({ length: len }, (_, t) => ({ [words[ids[n * 8 + t] ?? 0] ?? 'w0']: 1 }))
    const y = crfDecode(truth, x).labels.map((l, t) =>
      noise[n * 8 + t] === 1 ? (labels[other[n * 8 + t] ?? 0] ?? l) : l,
    )
    return { x, y }
  })
}

const crf = linearChainCrf(labels, words)
const oracle = chainOracle(crf)
const C2 = 0.3

const solvers = [
  { kind: 'subgradient' },
  { kind: 'bcfw' },
  { kind: 'one-slack', epsilon: 1e-6 },
  { kind: 'n-slack', epsilon: 1e-6 },
] as const

/** The training run of a solver on a problem: the final summary. */
function fit<X, Y>(problem: StructuredProblem<X, Y>, solver: object, steps = 400) {
  return trainStructuredSvm(problem, { solver: solver as never, key: key(3) }, steps)
}

describe('structured SVM: separable chain data', () => {
  const data = compileChainData(crf, planted(11, 12))
  const problem = { oracle, data, penalty: { c2: 0.01 } }

  it('every solver reaches zero training loss and error', () => {
    for (const solver of solvers) {
      const r = fit(problem, solver, 600)
      expect([solver.kind, r.loss, r.error]).toEqual([solver.kind, 0, 0])
    }
  })

  it('the cutting planes stop on their own and the gap is within epsilon', () => {
    for (const kind of ['one-slack', 'n-slack']) {
      const r = fit(problem, { kind, epsilon: 1e-6 })
      expect(r.converged).toBe(true)
      expect(r.gap ?? Number.NaN).toBeGreaterThanOrEqual(-1e-6)
      expect(r.gap ?? Number.NaN).toBeLessThanOrEqual(1e-6 + 1e-9)
    }
  })
})

describe('structured SVM: non-separable chain data', () => {
  const data = compileChainData(crf, planted(5, 10, 0.2))
  const problem = { oracle, data, penalty: { c2: C2 } }
  const reference = fit(problem, { kind: 'n-slack', epsilon: 1e-8 }, 600)

  it('the reference is a converged cutting plane with a tiny gap and a positive hinge', () => {
    expect(reference.converged).toBe(true)
    expect(reference.hinge).toBeGreaterThan(0.1)
    expect(Math.abs(reference.gap ?? 1)).toBeLessThan(1e-7)
    // the tape's objective is the structured hinge objective at the weights
    expect(structuredHingeObjective(oracle, data, reference.weights, problem.penalty).value).toBeCloseTo(
      reference.objective,
      9,
    )
  })

  it('all four solvers reach near the same primal objective', () => {
    const tolerance: Record<string, number> = { subgradient: 0.03, bcfw: 2e-3, 'one-slack': 1e-5, 'n-slack': 1e-6 }
    for (const solver of [{ ...solvers[0] }, { kind: 'bcfw', gapTol: 1e-3 }, solvers[2], solvers[3]]) {
      const r = fit(problem, solver, solver.kind === 'subgradient' ? 300 : 1500)
      const relative = (r.objective - reference.objective) / reference.objective
      // never below the optimum (beyond the solver's own slack), and near it
      expect(relative, solver.kind).toBeGreaterThanOrEqual(-1e-6)
      expect(relative, solver.kind).toBeLessThan(tolerance[solver.kind] ?? 0)
    }
  })

  it('beats the perceptron, averaged perceptron and passive-aggressive on the structured hinge objective', () => {
    const svm = reference.objective
    for (const kind of ['perceptron', 'averaged', 'passive-aggressive']) {
      const w = trainStructured({ oracle, data }, { solver: { kind } as never }, 60).weights
      expect(structuredHingeObjective(oracle, data, w, problem.penalty).value, kind).toBeGreaterThan(svm)
    }
  })

  it('the cutting planes do not depend on the order of the data', () => {
    const shuffled = [...data].reverse()
    for (const kind of ['one-slack', 'n-slack']) {
      const a = fit(problem, { kind, epsilon: 1e-8 }, 600)
      const b = fit({ ...problem, data: shuffled }, { kind, epsilon: 1e-8 }, 600)
      expect(b.objective, kind).toBeCloseTo(a.objective, 6)
      expect(b.dual ?? 0, kind).toBeCloseTo(a.dual ?? 0, 6)
    }
  })

  it('the one-slack and n-slack duals meet the Frank-Wolfe dual at the same optimum', () => {
    const bcfw = fit(problem, { kind: 'bcfw', gapTol: 1e-3 }, 2000)
    expect(bcfw.dual ?? Number.NaN).toBeLessThanOrEqual(reference.objective + 1e-6)
    expect(Math.abs((bcfw.dual ?? 0) - (reference.dual ?? 0)) / reference.objective).toBeLessThan(3e-3)
  })
})

describe('the tape of the duality gap and the working set', () => {
  const data = compileChainData(crf, planted(5, 10, 0.2))
  const problem = { oracle, data, penalty: { c2: C2 } }

  it('block-coordinate Frank-Wolfe: the gap is non-negative and shrinks, the dual never falls', () => {
    const gaps: number[] = []
    const duals: number[] = []
    const objectives: number[] = []
    const alg = structuredSvmAlgorithm(problem, { solver: { kind: 'bcfw', shuffle: false, gapTol: 1e-4 } })
    for (const { state } of iterate(alg, { key: key(0), maxSteps: 300 })) {
      gaps.push(state.gap ?? Number.NaN)
      duals.push(state.dual ?? Number.NaN)
      objectives.push(state.objective)
    }
    expect(gaps.length).toBeGreaterThan(5)
    for (const g of gaps) expect(g).toBeGreaterThanOrEqual(-1e-9)
    expect(gaps.at(-1) ?? 1).toBeLessThan((gaps[0] ?? 0) * 1e-2)
    for (let i = 1; i < duals.length; i++) expect(duals[i] ?? 0).toBeGreaterThanOrEqual((duals[i - 1] ?? 0) - 1e-9)
    // weak duality: every dual value is below every primal value
    expect(Math.max(...duals)).toBeLessThanOrEqual(Math.min(...objectives) + 1e-9)
  })

  it('averaging and the fixed step size also reach a small gap', () => {
    for (const solver of [
      { kind: 'bcfw', average: true },
      { kind: 'bcfw', lineSearch: false },
    ]) {
      const r = fit(problem, { ...solver, gapTol: 2e-2 }, 1500)
      expect(r.gap ?? Number.NaN).toBeGreaterThanOrEqual(-1e-9)
      expect(r.converged, JSON.stringify(solver)).toBe(true)
    }
  })

  it('cutting planes: the working set grows, then stops growing, and the gap is non-negative', () => {
    for (const kind of ['one-slack', 'n-slack']) {
      const sizes: number[] = []
      const gaps: number[] = []
      const alg = structuredSvmAlgorithm(problem, { solver: { kind, epsilon: 1e-6 } })
      let last = 0
      for (const { state } of iterate(alg, { key: key(0), maxSteps: 400 })) {
        sizes.push(state.constraints ?? Number.NaN)
        gaps.push(state.gap ?? Number.NaN)
        last = state.constraints ?? 0
      }
      for (let i = 1; i < sizes.length; i++) expect(sizes[i] ?? 0, kind).toBeGreaterThanOrEqual(sizes[i - 1] ?? 0)
      expect(sizes[0]).toBe(0)
      expect(last).toBeGreaterThan(1)
      // it stops: the last state has nothing pending and the next step would change nothing
      expect(sizes.length).toBeLessThan(400)
      for (const g of gaps) expect(g, kind).toBeGreaterThanOrEqual(-1e-7)
      expect(gaps.at(-1) ?? 1).toBeLessThanOrEqual(1e-6 + 1e-9)
      expect(gaps.at(-1) ?? 0).toBeLessThan(gaps[0] ?? 0)
    }
    // n-slack keeps a constraint per violated example, one-slack one per round
    const count = (kind: string): number => fit(problem, { kind, epsilon: 1e-6 }).constraints ?? 0
    expect(count('n-slack')).toBeGreaterThan(count('one-slack'))
  })

  it('the interior-point QP gives the same cutting-plane optimum as the active-set one', () => {
    const a = fit(problem, { kind: 'one-slack', epsilon: 1e-6, qp: 'interior-point' })
    const b = fit(problem, { kind: 'one-slack', epsilon: 1e-6, qp: 'active-set' })
    expect(b.objective).toBeCloseTo(a.objective, 4)
    expect(b.constraints).toBeGreaterThan(1)
  })

  it('reports the hinge objective, loss and error like the other modes, and the weights', () => {
    const alg = structuredSvmAlgorithm(problem, { solver: { kind: 'subgradient' } })
    const [first, second] = [...iterate(alg, { key: key(0), maxSteps: 2 })]
    expect(first?.state.objective).toBeCloseTo(
      structuredHingeObjective(oracle, data, new Float64Array(oracle.size), problem.penalty).value,
      12,
    )
    expect(second?.state.weights).toHaveLength(oracle.size)
    expect(second?.state.error).toBeGreaterThan(0)
    expect(second?.state.hinge).toBeGreaterThan(0)
    expect(second?.state.gap).toBeUndefined()
  })
})

/** An oracle that records how it is decoded. */
function spied<X, Y>(inner: StructuredOracle<X, Y>): { oracle: StructuredOracle<X, Y>; against: Y[]; plain: number } {
  const calls = { against: [] as Y[], plain: 0 }
  return {
    oracle: {
      ...inner,
      decode: (w, x, options) => {
        if (options?.against === undefined) calls.plain++
        else calls.against.push(options.against)
        return inner.decode(w, x, options)
      },
    },
    against: calls.against,
    get plain() {
      return calls.plain
    },
  }
}

describe('loss-augmented decoding', () => {
  const rows = planted(8, 6, 0.1)
  const data = compileChainData(crf, rows)

  it('every solver separates with `decode(w, x, { against: gold })` and only the error report decodes plainly', () => {
    for (const solver of solvers) {
      const spy = spied(oracle)
      const r = trainStructuredSvm({ oracle: spy.oracle, data, penalty: { c2: C2 } }, { solver: solver as never }, 3)
      expect(r.steps, solver.kind).toBeGreaterThan(0)
      expect(spy.against.length, solver.kind).toBeGreaterThanOrEqual(data.length)
      const golds = new Set(data.map((d) => d.y))
      for (const gold of spy.against) expect(golds.has(gold as never), solver.kind).toBe(true)
    }
  })

  it('the objective counts the cost: at zero weights its hinge loss is the summed Hamming size of the data', () => {
    const items = data.reduce((s, d) => s + d.y.length, 0)
    // at zero weights the best loss-augmented labelling differs everywhere (cost 1 per item, ties resolved to the cost)
    expect(structuredHingeObjective(oracle, data, new Float64Array(oracle.size), { c2: 1 }).value).toBe(items)
  })

  it('the subgradient is one: the objective lies above its linearisation at random points', () => {
    const g = structuredHingeObjective(oracle, data, new Float64Array(oracle.size), { c2: C2 })
    for (let seed = 0; seed < 4; seed++) {
      const w = Float64Array.from(toFlat(normal(key(seed), [oracle.size])), (v) => 0.7 * v)
      const at = structuredHingeObjective(oracle, data, w, { c2: C2 })
      let linear = at.value
      const v = Float64Array.from(toFlat(normal(key(seed + 9), [oracle.size])), (x) => 0.7 * x)
      for (let q = 0; q < v.length; q++) linear += (at.gradient[q] ?? 0) * ((v[q] ?? 0) - (w[q] ?? 0))
      expect(structuredHingeObjective(oracle, data, v, { c2: C2 }).value).toBeGreaterThanOrEqual(linear - 1e-9)
    }
    expect(g.hinge).toBeGreaterThan(0)
  })
})

describe('against the exact optimum of a tiny problem', () => {
  const tinyCrf = linearChainCrf(['A', 'B'], ['f', 'g'])
  const tiny = chainOracle(tinyCrf)
  const data = [
    { x: [{ f: 1 }, { g: 1 }, { f: 1 }], y: ['A', 'B', 'B'] },
    { x: [{ g: 1 }, { f: 1 }], y: ['B', 'B'] },
    { x: [{ f: 1 }, { f: 1 }, { g: 1 }], y: ['A', 'A', 'B'] },
  ]
  const c2 = 0.2

  /** The primal QP over (w, slacks) with every labelling of every example as a constraint, by the interior-point method. */
  function enumerated(): number {
    const P = tiny.size
    const n = data.length
    const rows: number[][] = []
    const h: number[] = []
    data.forEach((d, i) => {
      const len = d.x.length
      for (let code = 0; code < 2 ** len; code++) {
        const y = Array.from({ length: len }, (_, t) => ((code >> t) & 1 ? 'B' : 'A'))
        const gold = tiny.features(d.x, d.y)
        const guess = tiny.features(d.x, y)
        // -(w . psi) - xi_i <= -loss
        const row = new Array<number>(P + n).fill(0)
        for (let q = 0; q < P; q++) row[q] = -((gold[q] ?? 0) - (guess[q] ?? 0))
        row[P + i] = -1
        rows.push(row)
        h.push(-tiny.loss(d.y, y))
      }
    })
    const diagonal = Array.from({ length: P + n }, (_, a) =>
      Array.from({ length: P + n }, (_, b) => (a === b && a < P ? 2 * c2 : 1e-9 * Number(a === b))),
    )
    const qp = qpArrays({
      P: diagonal,
      q: [...new Array<number>(P).fill(0), ...new Array<number>(n).fill(1)],
      G: rows,
      h,
      lb: [...new Array<number>(P).fill(-Infinity), ...new Array<number>(n).fill(0)],
    })
    const solution = qpInteriorPoint(qp, { tol: 1e-10 })
    return solution.fun
  }

  it('every solver reaches the objective of the enumerated constraint set', () => {
    const exact = enumerated()
    const problem = { oracle: tiny, data, penalty: { c2 } }
    const tolerance: Record<string, number> = { subgradient: 0.05, bcfw: 1e-3, 'one-slack': 1e-5, 'n-slack': 1e-6 }
    for (const solver of solvers) {
      const r = fit(problem, solver, 1500)
      expect(Math.abs(r.objective - exact) / exact, solver.kind).toBeLessThan(tolerance[solver.kind] ?? 0)
    }
  })
})

describe('structured SVM on a planted tree CRF with exact inference', () => {
  const model = treeCrf(['a', 'b', 'c'], ['f', 'g'])
  const graph = graphOracle(model, { inference: 'exact' })
  const truth = Float64Array.from(toFlat(normal(key(70), [graph.size])), (v) => 1.5 * v)

  const data = Array.from({ length: 10 }, (_, n) => {
    const size = 5 + (n % 3)
    const parents = Array.from({ length: size }, (_, i) => (i === 0 ? -1 : (i * 7 + n) % i))
    const feats = toFlat(normal(key(100 + n), [size, 2]))
    const nodes = Array.from({ length: size }, (_, i) => ({ f: feats[2 * i] ?? 0, g: feats[2 * i + 1] ?? 0 }))
    const x: GraphInput = treeInput(nodes, { parents })
    const y: GraphLabelling = graph.decode(truth, x).y
    return { x, y }
  })

  it('every solver trains it: zero loss on the planted (separable) labels, all near one objective', () => {
    const problem = { oracle: graph, data, penalty: { c2: 0.05 } }
    const results = solvers.map((solver) => ({ solver, r: fit(problem, solver, 150) }))
    const best = Math.min(...results.map(({ r }) => r.objective))
    for (const { solver, r } of results) {
      expect(r.error, solver.kind).toBe(0)
      expect(r.objective, solver.kind).toBeLessThan(
        best * (solver.kind === 'subgradient' ? 2 : solver.kind === 'bcfw' ? 1.2 : 1.05) + 1e-3,
      )
    }
    expect(structuredErrors(problem, results[3]?.r.weights ?? []).error).toBe(0)
  })

  it('with noisy labels the cutting plane and Frank-Wolfe agree on the optimum, and the gap certifies it', () => {
    const noisy = data.map((d, n) => ({
      x: d.x,
      y: d.y.map((l, i) => ((i + n) % 5 === 0 ? (l === 'a' ? 'b' : 'a') : l)),
    }))
    const problem = { oracle: graph, data: noisy, penalty: { c2: 0.05 } }
    const cp = fit(problem, { kind: 'n-slack', epsilon: 1e-6 }, 300)
    const fw = fit(problem, { kind: 'bcfw', gapTol: 1e-3 }, 300)
    expect(cp.converged).toBe(true)
    expect(Math.abs(fw.objective - cp.objective) / cp.objective).toBeLessThan(5e-3)
    expect(fw.gap ?? Number.NaN).toBeGreaterThanOrEqual(-1e-9)
    expect(fw.objective - (fw.gap ?? 0)).toBeLessThanOrEqual(cp.objective + 1e-6)
  })
})

describe('the solver tables', () => {
  const data = compileChainData(crf, planted(2, 4))

  it('lists the four modes with Frank-Wolfe the default, and joins the structured table through the extras', () => {
    expect(structuredSvmSolvers.names).toEqual(['subgradient', 'bcfw', 'one-slack', 'n-slack'])
    expect(structuredSvmSolvers.modeOf({}).name).toBe('bcfw')
    const joined = structuredSolversWith(structuredSvmModes)
    expect(joined.names).toEqual(expect.arrayContaining(['lbfgs', 'perceptron', 'bcfw', 'n-slack']))
    expect(joined.modeOf({}).name).toBe('lbfgs')
    // the joined table builds an SVM mode over the same problem
    const mode = joined.make({ oracle, data, penalty: { c2: 0.1 } }, { kind: 'one-slack' }, key(0))
    expect(mode.name).toBe('structured-svm-one-slack')
  })

  it('maps the pystruct C to c2 and rejects problems that are not structured SVMs', () => {
    expect(structuredSvmPenalty(0.5)).toEqual({ c2: 1 })
    expect(() => structuredSvmPenalty(0)).toThrow(/positive/)
    expect(() => structuredSvmAlgorithm({ oracle, data: [] })).toThrow(/no training/)
    for (const kind of structuredSvmSolvers.names) {
      expect(() => structuredSvmSolvers.make({ oracle, data }, { kind }, key(0)), kind).toThrow(/c2/)
      expect(
        () => structuredSvmSolvers.make({ oracle, data, penalty: { c2: 1, c1: 1 } }, { kind }, key(0)),
        kind,
      ).toThrow(/L1/)
    }
  })

  it('every mode resumes from its serialised state to the same next state', () => {
    for (const kind of structuredSvmSolvers.names) {
      const algorithm = structuredSvmSolvers.make(
        { oracle, data, penalty: { c2: 0.1 } },
        { kind, shuffle: false },
        key(0),
      )
      const ctx = { key: key(0) } as never
      const s1 = algorithm.step(algorithm.init(key(0)) as never, ctx)
      const back = algorithm.deserialise?.(algorithm.serialise?.(s1) as never)
      const a = algorithm.step(s1 as never, ctx) as { x: Float64Array }
      const b = algorithm.step(back as never, ctx) as { x: Float64Array }
      expect(Array.from(b.x), kind).toEqual(Array.from(a.x))
    }
  })
})
