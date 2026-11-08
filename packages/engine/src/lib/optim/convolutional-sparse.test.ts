import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { pulseTrain } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  convolutionalAnalysis,
  convolutionalDictionaryContract,
  convolutionalDictionaryLearning,
  convolutionalDictionaryLearningAlgorithm,
  convolutionalSparseCode,
  convolutionalSparseCodeAlgorithm,
  convolutionalSynthesis,
  matchFilters,
} from './convolutional-sparse.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

// Two filters of length 4, and two maps of length 7 (a signal of 10 samples).
const D = [
  [1, -2, 0.5, 0],
  [0.3, 1, 1, -1],
]
const Z = [
  [0, 1, 0, 0, -0.5, 0, 2],
  [0.7, 0, 0, 0, 0, 1, 0],
]

// Six copies of the pulse (1, 3, 1) at irregular places and scales.
const pulses = ((): number[] => {
  const y = new Array<number>(60).fill(0)
  for (const [t, a] of [
    [3, 1],
    [11, -0.5],
    [19, 2],
    [30, 1.5],
    [41, -1],
    [50, 0.8],
  ] as const)
    [1, 3, 1].forEach((v, l) => {
      y[t + l] = (y[t + l] ?? 0) + a * v
    })
  return y
})()
const UNIT = [1, 3, 1].map((v) => v / Math.sqrt(11))

describe('convolutionalSynthesis', () => {
  it('equals the banded Toeplitz dictionary of all shifts times the stacked maps', () => {
    const [K, L, M] = [2, 4, 7]
    const N = M + L - 1
    // Column (k, m) of the dictionary is filter k placed at m.
    const columns = Array.from({ length: K * M }, (_, c) => {
      const [k, m] = [Math.floor(c / M), c % M]
      const col = new Array<number>(N).fill(0)
      for (const [l, v] of (D[k] ?? []).entries()) col[m + l] = v
      return col
    })
    const stacked = Z.flat()
    const expected = Array.from({ length: N }, (_, n) =>
      columns.reduce((a, col, c) => a + (col[n] ?? 0) * (stacked[c] ?? 0), 0),
    )
    const got = toFlat(convolutionalSynthesis(D, Z))
    expect(got.length).toBe(N)
    for (const [n, v] of expected.entries()) expect(got[n]).toBeCloseTo(v, 12)
  })

  it('places a filter at each non-zero activation, scaled', () => {
    expect(toArray(convolutionalSynthesis([[1, 2, 1]], [[0, 1, 0, 0, -0.5]]))).toEqual([0, 1, 2, 1, -0.5, -1, -0.5])
  })

  it('refuses a different number of maps and filters, and empty filters', () => {
    expect(() => convolutionalSynthesis(D, [Z[0] ?? []])).toThrow(/2 filters but 1/)
    expect(() => convolutionalSynthesis([], Z)).toThrow(/no filters|matrix/)
  })
})

describe('convolutionalAnalysis', () => {
  it('is the transpose of the synthesis: the same numbers as the dictionary applied to a signal', () => {
    const r = [0.3, -1, 2, 0.5, 0, 1, -0.4, 0.2, 1.5, -0.3]
    const a = toArray(convolutionalAnalysis(D, r)) as number[][]
    for (const k of [0, 1])
      for (let m = 0; m < 7; m++) {
        // <r, filter k placed at m>.
        const expected = (D[k] ?? []).reduce((s, v, l) => s + v * (r[m + l] ?? 0), 0)
        expect(a[k]?.[m]).toBeCloseTo(expected, 12)
      }
  })

  it('accepts a column and refuses a filter longer than the signal', () => {
    expect(toArray(convolutionalAnalysis([[1, 2, 1]], [[0], [1], [2], [1], [0]]))).toEqual([[4, 6, 4]])
    expect(() => convolutionalAnalysis([[1, 2, 1]], [1, 2])).toThrow(/longer than the signal/)
    expect(() =>
      convolutionalAnalysis(
        [[1]],
        [
          [1, 2],
          [3, 4],
        ],
      ),
    ).toThrow(/vector or a column/)
  })
})

describe('convolutionalSparseCode', () => {
  const y = [0, 0, 1, 2, 1, 0, 0, 0, -1, -2, -1, 0]

  it('matches the worked example: activations of plus or minus (1 - lambda/6) and exact zeros elsewhere', () => {
    const lambda = 0.1
    const { codes, converged } = convolutionalSparseCode([[1, 2, 1]], y, { lambda, maxSteps: 2000, gradTol: 1e-12 })
    expect(converged).toBe(true)
    for (const [m, v] of toFlat(codes).entries()) {
      if (m === 2) expect(v).toBeCloseTo(1 - lambda / 6, 9)
      else if (m === 8) expect(v).toBeCloseTo(-(1 - lambda / 6), 9)
      else expect(Math.abs(v)).toBe(0)
    }
  })

  it('reports the residual, its norm, the objective and the non-zeros consistently', () => {
    const r = convolutionalSparseCode([[1, 2, 1]], y, { lambda: 0.1 })
    const fitted = toFlat(convolutionalSynthesis([[1, 2, 1]], r.codes))
    const residual = toFlat(r.residual)
    for (const [n, v] of residual.entries()) expect(v).toBeCloseTo((y[n] ?? 0) - (fitted[n] ?? 0), 12)
    expect(r.residualNorm).toBeCloseTo(Math.hypot(...residual), 12)
    const l1 = toFlat(r.codes).reduce((s, v) => s + Math.abs(v), 0)
    expect(r.objective).toBeCloseTo(0.5 * r.residualNorm ** 2 + 0.1 * l1, 9)
    expect(r.nonzeros).toBe(2)
  })

  it('has the gradient of the smooth part: finite differences agree', () => {
    const signal = [0.3, -1, 2, 0.5, 0, 1, -0.4, 0.2, 1.5, -0.3]
    const alg = convolutionalSparseCodeAlgorithm(D, signal, {
      lambda: 0.1,
      init: [new Array(7).fill(0.1), new Array(7).fill(-0.2)],
    })
    const s0 = alg.init(key(1))
    const fit = (v: ArrayLike<number>): number => {
      const f = toFlat(convolutionalSynthesis(D, [Array.from(v).slice(0, 7), Array.from(v).slice(7)]))
      return 0.5 * f.reduce((a, g, n) => a + ((signal[n] ?? 0) - g) ** 2, 0)
    }
    for (let i = 0; i < s0.x.length; i++) {
      const h = 1e-6
      const up = Float64Array.from(s0.x)
      const down = Float64Array.from(s0.x)
      up[i] = (up[i] ?? 0) + h
      down[i] = (down[i] ?? 0) - h
      expect(s0.gy[i]).toBeCloseTo((fit(up) - fit(down)) / (2 * h), 6)
    }
  })

  it('reaches the same minimiser with and without acceleration, and FISTA needs fewer steps', () => {
    const options = { lambda: 0.1, maxSteps: 5000, gradTol: 1e-10 }
    const fista = convolutionalSparseCode([[1, 2, 1]], y, options)
    const ista = convolutionalSparseCode([[1, 2, 1]], y, { ...options, accelerate: false })
    expect(fista.objective).toBeCloseTo(ista.objective, 8)
    expect(fista.steps).toBeLessThanOrEqual(ista.steps)
  })

  it('codes nothing once lambda reaches the largest correlation of a filter with the signal', () => {
    const top = Math.max(
      ...toFlat(convolutionalAnalysis(D, [0.3, -1, 2, 0.5, 0, 1, -0.4, 0.2, 1.5, -0.3])).map(Math.abs),
    )
    const r = convolutionalSparseCode(D, [0.3, -1, 2, 0.5, 0, 1, -0.4, 0.2, 1.5, -0.3], { lambda: top * 1.001 })
    expect(r.nonzeros).toBe(0)
  })

  it('warm starts from given maps and refuses wrong shapes, a bad penalty and long filters', () => {
    const cold = convolutionalSparseCode([[1, 2, 1]], y, { lambda: 0.1, maxSteps: 400 })
    const warm = convolutionalSparseCodeAlgorithm([[1, 2, 1]], y, { lambda: 0.1, init: cold.codes })
    expect(warm.init(key(0)).value).toBeCloseTo(cold.objective, 8)
    expect(() => convolutionalSparseCodeAlgorithm([[1, 2, 1]], y, { lambda: 0.1, init: [[0, 0]] })).toThrow(
      /init must be/,
    )
    expect(() => convolutionalSparseCodeAlgorithm([[1, 2, 1]], y, { lambda: 0 })).toThrow(/lambda/)
    expect(() => convolutionalSparseCodeAlgorithm([[1, 2, 1]], [1, 2], { lambda: 1 })).toThrow(/longer than the signal/)
  })

  it('serialises, resumes to the same end, and logs the proximal gradient channels', () => {
    const alg = convolutionalSparseCodeAlgorithm([[1, 2, 1]], y, { lambda: 0.1 })
    const whole = runAlgorithm(alg, { maxSteps: 12 })
    const half = runAlgorithm(alg, { maxSteps: 5 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    // The budget of a resumed run counts from where it starts.
    const rest = runAlgorithm(alg, { maxSteps: 7, start: { state: stored, step: 5, serialised: true } })
    expect(Array.from(rest.state.x)).toEqual(Array.from(whole.state.x))
    expect(rest.state.value).toBe(whole.state.value)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 3, tracer: rec, policy: { ...policies.minimal, metrics: true } })
    const names = new Set(rec.events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    expect(names.has('loss')).toBe(true)
  })
})

describe('convolutionalDictionaryLearning', () => {
  it('never increases the objective, and keeps every filter in the unit ball', () => {
    const r = convolutionalDictionaryLearning(pulses, { filters: 2, length: 6, lambda: 0.1, maxSteps: 15 })
    for (let i = 1; i < r.history.length; i++)
      expect(r.history[i] as number).toBeLessThanOrEqual((r.history[i - 1] as number) * (1 + 1e-9))
    for (const f of toArray(r.filters) as number[][]) expect(Math.hypot(...f)).toBeLessThanOrEqual(1 + 1e-9)
    expect(r.history.length).toBe(r.steps + 1)
  })

  it('recovers a planted pulse at some shift within the filter when the start lands on the whole pulse', () => {
    for (const k of [0, 1, 2, 3]) {
      const { filters } = convolutionalDictionaryLearning(pulses, {
        filters: 1,
        length: 6,
        lambda: 0.1,
        maxSteps: 60,
        initialisation: 'energy',
        key: key(k),
      })
      const [m] = matchFilters(filters, [UNIT])
      expect(m?.score).toBeGreaterThan(0.999)
    }
  })

  it('starting from random stretches reaches the pulse from most keys, or a piece of it, never worse than its start', () => {
    const scores = [0, 1, 2, 3, 4, 5, 6, 7].map((k) => {
      const r = convolutionalDictionaryLearning(pulses, {
        filters: 1,
        length: 6,
        lambda: 0.1,
        maxSteps: 60,
        key: key(k),
      })
      expect(r.objective).toBeLessThanOrEqual((r.history[0] as number) + 1e-9)
      return matchFilters(r.filters, [UNIT])[0]?.score ?? 0
    })
    expect(scores.filter((s) => s > 0.999).length).toBeGreaterThanOrEqual(3)
  })

  it('has a map that is non-zero once per copy of the pulse', () => {
    const r = convolutionalDictionaryLearning(pulses, {
      filters: 1,
      length: 6,
      lambda: 0.1,
      maxSteps: 60,
      initialisation: 'energy',
    })
    expect(toFlat(r.codes).filter((v) => v !== 0).length).toBe(6)
  })

  it('recovers pulses planted in noise from the pulse train generator: the learnt filters match the planted ones', () => {
    const g = generate(pulseTrain, {
      key: key(5),
      knobs: { n: 400, pulses: 2, length: 12, occurrences: 8, noise: 0.01 },
    })
    const planted = g.truth.extra?.['filters'] as number[][]
    for (const initialisation of ['random', 'energy'] as const)
      for (const k of [0, 1, 2]) {
        const alg = convolutionalDictionaryLearningAlgorithm(g.data.values, {
          filters: 2,
          length: 16,
          lambda: 0.1,
          initialisation,
        })
        const start = runAlgorithm(alg, { maxSteps: 0, key: key(k) }).state
        const end = runAlgorithm(alg, { maxSteps: 40, key: key(k) }).state
        const learnt = Array.from({ length: 2 }, (_, i) => Array.from(end.filters.subarray(i * 16, (i + 1) * 16)))
        const matches = matchFilters(learnt, planted)
        // Each planted pulse is found, by a filter of its own, up to shift and sign.
        expect(matches.map((m) => m.learnt).sort()).toEqual([0, 1])
        for (const m of matches) expect(m.score).toBeGreaterThan(0.98)
        // From random stretches of the signal the reconstruction error falls by an order of magnitude.
        if (initialisation === 'random') expect(end.fit).toBeLessThan(start.fit / 10)
        else expect(end.fit).toBeLessThan(start.fit)
      }
  })

  it('takes the initial filters as given (scaled to unit norm), and refuses the wrong shape', () => {
    const alg = convolutionalDictionaryLearningAlgorithm(pulses, {
      filters: 1,
      length: 6,
      lambda: 0.1,
      init: [[0, 0, 2, 6, 2, 0]],
    })
    const s = alg.init(key(0))
    expect(Math.hypot(...s.filters)).toBeCloseTo(1, 12)
    expect(s.filters[3]).toBeCloseTo(3 / Math.sqrt(11), 12)
    expect(() =>
      convolutionalDictionaryLearningAlgorithm(pulses, {
        filters: 2,
        length: 6,
        lambda: 0.1,
        init: [[1, 2, 3, 4, 5, 6]],
      }).init(key(0)),
    ).toThrow(/init must be 2 x 6/)
  })

  it('rejects filters longer than the signal, a bad count and a bad penalty', () => {
    expect(() => convolutionalDictionaryLearningAlgorithm([1, 2], { filters: 1, length: 3, lambda: 0.1 })).toThrow(
      /length/,
    )
    expect(() => convolutionalDictionaryLearningAlgorithm(pulses, { filters: 0, length: 3, lambda: 0.1 })).toThrow(
      /filters/,
    )
    expect(() => convolutionalDictionaryLearningAlgorithm(pulses, { filters: 1, length: 3, lambda: -1 })).toThrow(
      /lambda/,
    )
  })

  it('keeps its filters when the penalty is so large that every map is zero, instead of dividing by nothing', () => {
    const y = new Array<number>(24).fill(0)
    y[14] = -0.2
    const r = convolutionalDictionaryLearning(y, { filters: 1, length: 2, lambda: 0.3, maxSteps: 4 })
    expect(Number.isFinite(r.objective)).toBe(true)
    expect(r.status).toBe('converged')
    expect(toFlat(r.codes).every((v) => v === 0)).toBe(true)
  })

  it('gives K spikes when the signal has no stretch that is not flat', () => {
    const alg = convolutionalDictionaryLearningAlgorithm(new Array(20).fill(0), { filters: 2, length: 4, lambda: 0.1 })
    expect(Array.from(alg.init(key(0)).filters)).toEqual([1, 0, 0, 0, 0, 1, 0, 0])
  })
})

describe('the learner as an algorithm', () => {
  const alg = convolutionalDictionaryLearningAlgorithm(pulses, {
    filters: 2,
    length: 6,
    lambda: 0.1,
    codeSteps: 8,
    filterSteps: 10,
  })

  it('serialises a state through JSON and resumes to the same end (and the same init from the same key)', () => {
    const whole = runAlgorithm(alg, { maxSteps: 6, key: key(2) })
    const half = runAlgorithm(alg, { maxSteps: 3, key: key(2) }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const back = alg.deserialise?.(stored)
    expect(Array.from(back?.filters ?? [])).toEqual(Array.from(half.filters))
    const rest = runAlgorithm(alg, { maxSteps: 3, key: key(2), start: { state: stored, step: 3, serialised: true } })
    expect(Array.from(rest.state.filters)).toEqual(Array.from(whole.state.filters))
    expect(rest.state.objective).toBe(whole.state.objective)
    expect(rest.steps).toBe(whole.steps)
  })

  it('logs the contract paths each round and observes the filters and maps', () => {
    const rec = recorder()
    runAlgorithm(alg, {
      maxSteps: 4,
      key: key(0),
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['param', 'state'], detail: 'full' },
    })
    const names = new Set(rec.events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    for (const n of ['objective', 'fit', 'nonzeros', 'change']) expect(names.has(n)).toBe(true)
    const tensors = new Set(rec.events.flatMap((e) => (e.type === 'tensor' ? [e.name] : [])))
    expect(tensors.has('filters')).toBe(true)
    expect(tensors.has('codes')).toBe(true)
  })

  it('ends converged when the objective stops changing, and the fit never exceeds the objective', () => {
    const run = runAlgorithm(alg, { maxSteps: 200, key: key(0) })
    expect(run.status).toBe('converged')
    expect(run.state.fit).toBeLessThanOrEqual(run.state.objective)
  })

  it('has probes: filters, maps and the reconstruction as curve records', () => {
    const contract = convolutionalDictionaryContract(pulses, { filters: 2, length: 6, lambda: 0.1 })
    expect(Object.keys(contract).sort()).toEqual([
      'cdl/change',
      'cdl/codes',
      'cdl/filters',
      'cdl/fit',
      'cdl/nonzeros',
      'cdl/objective',
      'cdl/reconstruction',
    ])
    const s = runAlgorithm(alg, { maxSteps: 3, key: key(0) }).state
    const probe = (path: 'cdl/filters' | 'cdl/codes' | 'cdl/reconstruction') => {
      const source = contract[path]
      return source?.kind === 'derived'
        ? (source.compute(s) as { kind: string; series: { name: string; y: number[] }[] })
        : undefined
    }
    expect(probe('cdl/filters')?.series.map((x) => x.y.length)).toEqual([6, 6])
    expect(probe('cdl/codes')?.series.map((x) => x.y.length)).toEqual([55, 55])
    const rec = probe('cdl/reconstruction')
    expect(rec?.kind).toBe('curves')
    expect(rec?.series[0]?.y.length).toBe(60)
    expect(rec?.series[1]?.y.length).toBe(60)
  })
})

describe('matchFilters', () => {
  it('finds a filter up to shift and sign, and picks the better of two learnt filters', () => {
    const [m] = matchFilters(
      [
        [1, 0, 0, 0, 0],
        [0, -1, -3, -1, 0],
      ],
      [[1, 3, 1]],
    )
    expect(m).toMatchObject({ planted: 0, learnt: 1, shift: 1, sign: -1 })
    expect(m?.score).toBeCloseTo(1, 12)
  })

  it('scores below one for a different shape', () => {
    const [m] = matchFilters([[1, 1, 1, 1]], [[1, 3, 1]])
    expect(m?.score).toBeLessThan(0.95)
  })
})
