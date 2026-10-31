import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../../testing/close.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal, randint } from '../../random/index.ts'
import { item, toArray, toFlat, type Value } from '../../tensor/index.ts'
import { iterate, metricSeries, recorder, runAlgorithm } from '../../trace/index.ts'
import {
  classicalMemory,
  denseAssociativeMemory,
  hopfieldEnergy,
  hopfieldRecallAlgorithm,
  hopfieldWeights,
  ModernHopfield,
  modernHopfieldAlgorithm,
  modernHopfieldEnergy,
  modernHopfieldUpdate,
} from './hopfield.ts'

const signs = (seed: number, p: number, n: number): number[][] =>
  Array.from({ length: p }, (_, i) => toFlat(randint(key(seed + i), [n], 0, 2)).map((b) => (b === 1 ? 1 : -1)))

const flip = (s: readonly number[], idx: readonly number[]): number[] => s.map((v, j) => (idx.includes(j) ? -v : v))

describe('classical Hopfield network', () => {
  it('Hebb weights are symmetric with a zero diagonal and store a few patterns as fixed points', () => {
    const patterns = signs(1, 3, 64)
    const w = hopfieldWeights(patterns)
    const rows = toArray(w) as number[][]
    for (const [i, r] of rows.entries()) {
      expect(r[i]).toBe(0)
      for (const [j, v] of r.entries()) expect(v).toBeCloseTo(rows[j]?.[i] ?? Number.NaN, 14)
    }
    const m = classicalMemory(w)
    for (const p of patterns) for (let i = 0; i < 64; i++) expect(m.update(p, i)).toBe(p[i])
  })

  it('the pseudo-inverse rule stores correlated patterns that the Hebb rule corrupts', () => {
    // Three patterns sharing most of their entries: Hebb crosstalk is large, the projection is exact.
    const base = signs(7, 1, 40)[0] as number[]
    const patterns = [base, flip(base, [0, 1, 2, 3]), flip(base, [4, 5, 6, 7, 8, 9])]
    const stable = (rule: 'hebb' | 'pseudo-inverse'): boolean => {
      const m = classicalMemory(hopfieldWeights(patterns, rule))
      return patterns.every((p) => p.every((v, i) => m.update(p, i) === v))
    }
    expect(stable('pseudo-inverse')).toBe(true)
    expect(stable('hebb')).toBe(false)
  })

  it('the energy of a batch is the quadratic form', () => {
    const w = [
      [0, 1],
      [1, 0],
    ]
    expect(
      toArray(
        hopfieldEnergy(w, [
          [1, 1],
          [1, -1],
        ]),
      ),
    ).toEqual([-1, 1])
    expect(item(hopfieldEnergy(w, [1, 1], [1, 2]))).toBe(-4)
  })

  it('asynchronous updates never raise the energy, for any symmetric weights and order', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 10_000 }), fc.constantFrom('sequential', 'random'), (seed, order) => {
        const n = 12
        const a = toArray(normal(key(seed), [n, n])) as number[][]
        const w = a.map((r, i) => r.map((_, j) => (i === j ? 0 : ((a[i]?.[j] ?? 0) + (a[j]?.[i] ?? 0)) / 2)))
        const b = toFlat(normal(key(seed + 1), [n]))
        const start = signs(seed + 2, 1, n)[0] as number[]
        const algo = hopfieldRecallAlgorithm(classicalMemory(w, b), start, { order })
        const energies = Array.from(iterate(algo, { seed, maxSteps: 400 }), (x) => x.state.energy)
        for (let i = 1; i < energies.length; i++)
          expect(energies[i] as number).toBeLessThanOrEqual((energies[i - 1] as number) + 1e-12)
        // The reported energy is the energy of the state.
        const last = Array.from(iterate(algo, { seed, maxSteps: 400 })).at(-1)?.state
        expect(last?.energy).toBeCloseTo(item(hopfieldEnergy(w, last?.state ?? [], b)), 10)
      }),
      { numRuns: 25 },
    )
  })

  it('recalls a stored pattern from a corrupted one and logs the energy falling and the overlap rising', () => {
    const patterns = signs(3, 3, 80)
    const target = patterns[1] as number[]
    const start = flip(
      target,
      Array.from({ length: 12 }, (_, i) => i * 6),
    )
    const algo = hopfieldRecallAlgorithm(classicalMemory(hopfieldWeights(patterns)), start, { patterns })
    const rec = recorder()
    const result = runAlgorithm(algo, { maxSteps: 2000, tracer: rec })
    expect(result.status).toBe('converged')
    expect(result.state.state).toEqual(target)
    const run = rec.run()
    const energy = metricSeries(run, 'recall/energy').values
    const overlap = metricSeries(run, 'recall/overlap').values
    expect(energy.at(-1) as number).toBeLessThan(energy[0] as number)
    expect(overlap[0]).toBeCloseTo(1 - 24 / 80, 12)
    expect(overlap.at(-1)).toBe(1)
  })

  it('capacity: recall fails once far more than 0.14 n patterns are stored', () => {
    const n = 100
    const recalls = (p: number): number => {
      const patterns = signs(50, p, n)
      const m = classicalMemory(hopfieldWeights(patterns))
      return patterns.filter((q) => q.every((v, i) => m.update(q, i) === v)).length / p
    }
    expect(recalls(5)).toBe(1)
    expect(recalls(40)).toBeLessThan(0.5)
  })

  it('state round-trips through serialise', () => {
    const algo = hopfieldRecallAlgorithm(
      classicalMemory(hopfieldWeights(signs(1, 2, 10))),
      signs(9, 1, 10)[0] as number[],
    )
    const s = runAlgorithm(algo, { maxSteps: 7 }).state
    expect(algo.deserialise?.(JSON.parse(JSON.stringify(algo.serialise?.(s))))).toEqual(s)
  })
})

describe('dense associative memory', () => {
  it('stores far more patterns than the classical network and its updates never raise the energy', () => {
    const n = 60
    const patterns = signs(20, 25, n)
    const classic = classicalMemory(hopfieldWeights(patterns))
    const dense = denseAssociativeMemory(patterns, { interaction: 'polynomial', degree: 4 })
    const fixed = (m: { update: (s: number[], i: number) => number }): number =>
      patterns.filter((q) => q.every((v, i) => m.update(q, i) === v)).length
    expect(fixed(dense)).toBe(25)
    expect(fixed(classic)).toBeLessThan(25)
    for (const interaction of ['polynomial', 'exponential'] as const) {
      const memory = denseAssociativeMemory(patterns, { interaction, degree: 3, beta: 10 })
      const start = flip(patterns[4] as number[], [0, 5, 10, 15, 20, 25, 30])
      const energies = Array.from(
        iterate(hopfieldRecallAlgorithm(memory, start, { order: 'random' }), { seed: 1, maxSteps: 600 }),
        (x) => x.state.energy,
      )
      for (let i = 1; i < energies.length; i++)
        expect(energies[i] as number).toBeLessThanOrEqual((energies[i - 1] as number) + 1e-9)
      const final = runAlgorithm(hopfieldRecallAlgorithm(memory, start), { maxSteps: 2000 }).state.state
      expect(final).toEqual(patterns[4])
    }
  })
})

describe('modern Hopfield network', () => {
  const patterns = [
    [1, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 1],
  ]
  it('the update is attention over the stored patterns', () => {
    const q = [[0.5, 0.2, 0.1, 0.9]]
    const manual = toArray(modernHopfieldUpdate(patterns, q, 2.5)) as number[][]
    // softmax of beta * (X q) weights the rows.
    const scores = patterns.map((p) => 2.5 * p.reduce((a, v, j) => a + v * (q[0]?.[j] ?? 0), 0))
    const m = Math.max(...scores)
    const e = scores.map((s) => Math.exp(s - m))
    const z = e.reduce((a, b) => a + b, 0)
    for (let j = 0; j < 4; j++) {
      const want = patterns.reduce((a, p, i) => a + ((e[i] as number) / z) * (p[j] as number), 0)
      expect(manual[0]?.[j]).toBeCloseTo(want, 12)
    }
  })

  it('the energy never rises along the retrieval and the pattern is a fixed point at large beta', () => {
    const unit = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
    ]
    const algo = modernHopfieldAlgorithm(unit, [0.3, 0.8, 0.2, 0.1], 4)
    const states = Array.from(iterate(algo, { maxSteps: 50 }), (x) => x.state)
    for (let i = 1; i < states.length; i++)
      expect(states[i]?.energy as number).toBeLessThanOrEqual((states[i - 1]?.energy as number) + 1e-12)
    expect(states.at(-1)?.state.map((v) => Math.round(v))).toEqual([0, 1, 0, 0])
    const fixed = toFlat(modernHopfieldUpdate(patterns, [1, 0, 0, 1], 60))
    expectClose(fixed, patterns[0] as number[])
  })

  it('the layer has the energy as its objective: its gradient in the patterns checks out', () => {
    const layer = new ModernHopfield(4, 3, { patterns, beta: 1.5 })
    const x = normal(key(4), [2, 4])
    expect(
      checkPrimitive((p: Value) => layer.apply({ patterns: p }, x), [layer.params['patterns'] as Value], { rtol: 1e-5 })
        .failures,
    ).toEqual([])
    expect(checkPrimitive((s: Value) => modernHopfieldEnergy(patterns, s, 1.5), [x], { rtol: 1e-5 }).failures).toEqual(
      [],
    )
  })
})
