import { describe, expect, it } from 'vitest'
import { key, uniformStream } from '../random/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { coverMethods, sampleSatSchemes, unificationVariants, versionSpaceLearners } from './algorithm-options.ts'
import {
  candidateElimination,
  candidateEliminationAlgorithm,
  findS,
  findSAlgorithm,
  versionSpaceContract,
} from './ilp/version-spaces.ts'
import { mgu, parseTerm, substitutionToString, unifierOf, unify, unifyAlgorithm, unifyContract } from './index.ts'
import { sampleSat, sampleSatAlgorithm, sampleSatContract } from './markov-logic.ts'
import {
  evaluateFormula,
  formulaToString,
  minimalDnf,
  minimalFormOf,
  parseFormula,
  primeImplicants,
  primesOf,
  qmContract,
  quineMcCluskeyAlgorithm,
} from './propositional.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const logPolicy = { ...policies.minimal, metrics: true, detail: 'full' } as const
const metric = (rec: ReturnType<typeof recorder>, name: string): number[] =>
  rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))

describe('quineMcCluskeyAlgorithm', () => {
  const f = parseFormula('(a & b & ~c) | (a & b & c) | (~a & b & c) | (~a & ~b & c) | (a & ~b & c)')

  it('merges one round per step into the prime implicants, as primeImplicants does', () => {
    const run = runAlgorithm(quineMcCluskeyAlgorithm(f), { maxSteps: 20 })
    expect(run.status).toBe('converged')
    const vars = ['a', 'b', 'c']
    expect(primesOf(run.state, vars)).toEqual(primeImplicants(f))
    // Cubes shrink in number and gain absent variables each round; the rounds end with none left.
    const rec = recorder()
    runAlgorithm(quineMcCluskeyAlgorithm(f), { maxSteps: 20, tracer: rec, policy: logPolicy })
    expect(metric(rec, 'qm/round')).toEqual(Array.from({ length: run.steps }, (_, i) => i + 1))
    expect(metric(rec, 'qm/cubes').at(-1)).toBe(0)
    const primes = metric(rec, 'qm/primes')
    expect(primes.every((p, i) => i === 0 || p >= (primes[i - 1] ?? 0))).toBe(true)
    expect(Object.keys(qmContract)).toHaveLength(4)
  })

  it('covers the models exactly or greedily, and agrees with minimalDnf', () => {
    const vars = ['a', 'b', 'c']
    const exact = runAlgorithm(quineMcCluskeyAlgorithm(f, { cover: 'exact' }), { maxSteps: 20 })
    expect(formulaToString(minimalFormOf(exact.state, vars, f))).toBe(formulaToString(minimalDnf(f)))
    const greedy = runAlgorithm(quineMcCluskeyAlgorithm(f, { cover: 'greedy' }), { maxSteps: 20 })
    expect((greedy.state.chosen ?? []).length).toBeGreaterThanOrEqual((exact.state.chosen ?? []).length)
    // Both covers cover every model.
    for (const run of [exact, greedy]) {
      const dnf = minimalFormOf(run.state, vars, f)
      for (const m of [0, 1, 2, 3, 4, 5, 6, 7]) {
        const val = { a: (m & 4) !== 0, b: (m & 2) !== 0, c: (m & 1) !== 0 }
        expect(evaluateFormula(dnf, val)).toBe(evaluateFormula(f, val))
      }
    }
    expect(formulaToString(minimalDnf(parseFormula('p & ~p')))).toBe('false')
    expect(coverMethods.make({ kind: 'none' })).toEqual({ cover: 'none' })
  })

  it('serialises, resumes and rejects too many variables', () => {
    const alg = quineMcCluskeyAlgorithm(f, { cover: 'exact' })
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 20,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
    const many = Array.from({ length: 17 }, (_, i) => `x${i}`).join(' & ')
    expect(() => quineMcCluskeyAlgorithm(parseFormula(many))).toThrow(/16 variables/)
  })
})

describe('sampleSatAlgorithm', () => {
  const clauses = [
    [1, 2, 3],
    [-1, -2],
    [-2, -3],
    [1, -3],
    [2, 3, -1],
  ]

  it('flips one variable per step until a model is read, and the model satisfies the clauses', () => {
    const run = runAlgorithm(sampleSatAlgorithm(clauses, 3, key(3)), { maxSteps: 5000 })
    expect(run.status).toBe('converged')
    expect(run.state.unsat).toHaveLength(0)
    for (const c of clauses) expect(c.some((l) => (run.state.value[Math.abs(l) - 1] === 1) === l > 0)).toBe(true)
    expect(run.steps).toBe(run.state.flips)
  })

  it('agrees with sampleSat on the same stream of coins', () => {
    const run = runAlgorithm(sampleSatAlgorithm(clauses, 3, key(8)), { maxSteps: 5000 })
    const direct = sampleSat(clauses, 3, uniformStream(key(8), 256))
    expect(direct).not.toBeNull()
    expect(Array.from(run.state.value)).toEqual(Array.from(direct as Uint8Array))
  })

  it('resumes the coin stream from a JSON state, fails on an empty clause, and logs', () => {
    const alg = sampleSatAlgorithm(clauses, 3, key(5))
    const whole = runAlgorithm(alg, { maxSteps: 5000 })
    for (const cut of [1, 5, 17].filter((c) => c < whole.steps)) {
      const part = runAlgorithm(alg, { maxSteps: cut }).state
      const rest = runAlgorithm(alg, {
        maxSteps: 5000,
        start: { state: viaJson(alg.serialise?.(part) ?? null), step: cut, serialised: true },
      })
      expect(Array.from(rest.state.value)).toEqual(Array.from(whole.state.value))
      expect(rest.state.flips).toBe(whole.state.flips)
    }
    expect(runAlgorithm(sampleSatAlgorithm([[]], 1, key(0)), { maxSteps: 10 }).status).toBe('failed')
    expect(sampleSat([[]], 1, uniformStream(key(0)))).toBeNull()
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 5000, tracer: rec, policy: logPolicy })
    expect(metric(rec, 'sat/flips')).toHaveLength(whole.steps)
    expect(metric(rec, 'sat/unsat').at(-1)).toBe(0)
    expect(Object.keys(sampleSatContract)).toHaveLength(4)
    expect(sampleSatSchemes.make({ kind: 'walksat' }).annealing).toBe(0.1)
  })
})

describe('version-space learners', () => {
  const domains = [
    ['sunny', 'cloudy', 'rainy'],
    ['warm', 'cold'],
    ['normal', 'high'],
  ]
  const examples = [
    { values: ['sunny', 'warm', 'normal'], positive: true },
    { values: ['sunny', 'warm', 'high'], positive: true },
    { values: ['rainy', 'cold', 'high'], positive: false },
  ]

  it('findSAlgorithm generalises once per positive and equals findS', () => {
    const run = runAlgorithm(findSAlgorithm(examples), { maxSteps: 10 })
    expect([run.status, run.steps]).toEqual(['converged', 3])
    expect(run.state.hypothesis).toEqual(findS(examples))
    expect(findS(examples)).toEqual(['sunny', 'warm', '?'])
    expect(findS([{ values: ['a'], positive: false }])).toEqual([null])
    const alg = findSAlgorithm(examples)
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 10,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state).toEqual(run.state)
  })

  it('candidate elimination serialises, resumes and logs typed boundaries', () => {
    const alg = candidateEliminationAlgorithm(examples, domains)
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 10,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
    const vs = candidateElimination(examples, domains)
    expect(vs.specific).toEqual(whole.state.specific)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy: logPolicy })
    expect(metric(rec, 'vs/specific')).toHaveLength(3)
    expect(Object.keys(versionSpaceContract)).toHaveLength(3)
    expect(versionSpaceLearners.names).toEqual(['find-s', 'candidate-elimination'])
    expect(versionSpaceLearners.make({ kind: 'find-s' })(examples, domains).name).toBe('find-s')
  })
})

describe('unifyAlgorithm', () => {
  const pairs = [
    ['f(X, g(Y))', 'f(a, g(X))'],
    ['p(X, f(Y))', 'p(g(Z), f(a))'],
    ['f(a)', 'f(b)'],
    ['f(X, X)', 'f(a, b)'],
    ['X', 'f(X)'],
    ['h(X, Y, Z)', 'h(Y, Z, f(W))'],
  ] as const

  it('solves one equation per step with the bindings of unify', () => {
    for (const occursCheck of [false, true]) {
      for (const [l, r] of pairs) {
        const run = runAlgorithm(unifyAlgorithm(parseTerm(l), parseTerm(r), { occursCheck }), { maxSteps: 100 })
        const direct = unify(parseTerm(l), parseTerm(r), undefined, { occursCheck })
        expect(run.status === 'converged').toBe(direct !== null)
        const got = unifierOf(run.state)
        if (direct === null) expect(got).toBeNull()
        else expect(substitutionToString(got as Map<string, never>)).toBe(substitutionToString(direct))
      }
    }
    expect(substitutionToString(mgu(parseTerm('p(X, f(Y))'), parseTerm('p(g(Z), f(a))')) as never)).toBe(
      '{X/g(Z), Y/a}',
    )
  })

  it('applies the named rules, logs them, resumes from JSON, and the occurs check turns a cycle into a failure', () => {
    const rec = recorder()
    const alg = unifyAlgorithm(parseTerm('f(X, g(Y))'), parseTerm('f(a, g(X))'))
    const whole = runAlgorithm(alg, {
      maxSteps: 100,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, detail: 'full' },
    })
    expect(whole.steps).toBe(4)
    expect(metric(rec, 'unify/equations')).toEqual([2, 1, 1, 0])
    expect(metric(rec, 'unify/substitution').at(-1)).toBe(2)
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 100,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(rest.state.bindings).toEqual(whole.state.bindings)
    const cyc = runAlgorithm(unifyAlgorithm(parseTerm('X'), parseTerm('f(X)'), { occursCheck: true }), { maxSteps: 10 })
    expect([cyc.status, cyc.state.rule]).toEqual(['failed', 'occurs'])
    const clash = runAlgorithm(unifyAlgorithm(parseTerm('f(a)'), parseTerm('g(a)')), { maxSteps: 10 })
    expect([clash.status, clash.state.rule]).toEqual(['failed', 'clash'])
    expect(Object.keys(unifyContract)).toHaveLength(3)
    expect(unificationVariants.make({ kind: 'sound' })).toEqual({ occursCheck: true })
  })
})
