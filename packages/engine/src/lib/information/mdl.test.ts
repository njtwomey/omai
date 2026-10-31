import { describe, expect, it } from 'vitest'
import { solve } from '../numerics/linalg/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  arithmeticEncode,
  bestMdl,
  dirichletCodeLength,
  gaussianDataBits,
  histogramMdl,
  mdlSelectionAlgorithm,
  memorylessSource,
  multinomialParametricComplexity,
  nmlCodeLength,
  nmlRegret,
  nmlRegretAsymptotic,
  parameterBits,
  polynomialMdl,
} from './index.ts'

/** Every sequence of length n over k categories, as its counts. */
function allCounts(k: number, n: number): number[][] {
  const out: number[][] = []
  const seq = new Array<number>(n).fill(0)
  for (let i = 0; i < k ** n; i++) {
    let v = i
    const counts = new Array<number>(k).fill(0)
    for (let t = 0; t < n; t++) {
      seq[t] = v % k
      v = Math.floor(v / k)
      counts[seq[t] ?? 0] = (counts[seq[t] ?? 0] ?? 0) + 1
    }
    out.push(counts)
  }
  return out
}

describe('the multinomial universal codes', () => {
  it("the parametric complexity is Shtarkov's sum of maximum-likelihood probabilities over all sequences", () => {
    for (const [k, n] of [
      [2, 9],
      [3, 6],
      [4, 5],
      [5, 4],
    ] as const) {
      let sum = 0
      for (const c of allCounts(k, n)) sum += c.reduce((a, h) => a * (h === 0 ? 1 : (h / n) ** h), 1)
      expect(multinomialParametricComplexity(k, n)).toBeCloseTo(sum, 10)
    }
    expect(multinomialParametricComplexity(2, 2)).toBeCloseTo(2.5, 14)
    expect(multinomialParametricComplexity(3, 2)).toBeCloseTo(4.5, 14)
    expect(multinomialParametricComplexity(1, 50)).toBe(1)
    expect(multinomialParametricComplexity(7, 0)).toBe(1)
  })

  it('the NML code is a probability distribution over sequences', () => {
    for (const [k, n] of [
      [3, 6],
      [2, 10],
    ] as const) {
      const total = allCounts(k, n).reduce((a, c) => a + 2 ** -nmlCodeLength(c), 0)
      expect(total).toBeCloseTo(1, 10)
    }
  })

  it('the NML length is the maximum-likelihood length plus the regret, the same for every sequence of the family', () => {
    expect(nmlCodeLength([2, 0])).toBeCloseTo(Math.log2(2.5), 12)
    const ml = (c: number[]): number => -c.reduce((a, h) => a + (h > 0 ? h * Math.log2(h / 10) : 0), 0)
    expect(nmlCodeLength([5, 5]) - ml([5, 5])).toBeCloseTo(nmlRegret(2, 10), 12)
    expect(nmlCodeLength([10, 0]) - ml([10, 0])).toBeCloseTo(nmlRegret(2, 10), 12)
  })

  it("the regret approaches Rissanen's expansion", () => {
    let last = Number.POSITIVE_INFINITY
    for (const n of [10, 100, 1000, 10000]) {
      const gap = Math.abs(nmlRegret(3, n) - nmlRegretAsymptotic(3, n))
      expect(gap).toBeLessThan(last)
      last = gap
    }
    // The error is of order 1 / sqrt(n): the scaled gap is about constant.
    const scaled = (n: number): number => Math.abs(nmlRegret(3, n) - nmlRegretAsymptotic(3, n)) * Math.sqrt(n)
    expect(scaled(10000) / scaled(1000)).toBeGreaterThan(0.9)
    expect(scaled(10000) / scaled(1000)).toBeLessThan(1.1)
    expect(last).toBeLessThan(0.02)
    expect(nmlRegretAsymptotic(2, 1000)).toBeCloseTo(0.5 * Math.log2(500) + Math.log2(Math.sqrt(Math.PI)), 10)
  })

  it('the sequential Dirichlet codes are distributions too, and Laplace is what an adaptive arithmetic coder does', () => {
    for (const alpha of [0.5, 1, 2]) {
      expect(allCounts(3, 5).reduce((a, c) => a + 2 ** -dirichletCodeLength(c, alpha), 0)).toBeCloseTo(1, 10)
    }
    expect(dirichletCodeLength([1, 0], 1)).toBeCloseTo(1, 12)
    const s = memorylessSource(key(2), [0.5, 0.3, 0.15, 0.05], 400)
    const counts = [0, 1, 2, 3].map((v) => s.symbols.filter((x) => x === v).length)
    const bits = arithmeticEncode(s.symbols, { kind: 'adaptive', alphabetSize: 4 }).length
    const laplace = dirichletCodeLength(counts, 1)
    expect(bits).toBeGreaterThanOrEqual(laplace - 3)
    expect(bits).toBeLessThan(laplace + 2.5)
  })

  it('the Krichevsky-Trofimov code stays within a constant of the NML code', () => {
    for (const c of [
      [50, 50],
      [90, 10],
      [100, 0],
      [30, 40, 30],
    ]) {
      expect(Math.abs(dirichletCodeLength(c, 0.5) - nmlCodeLength(c))).toBeLessThan(1.1)
    }
  })

  it('rejects bad counts and parameters', () => {
    expect(() => nmlCodeLength([1.5, 1])).toThrow(/integers/)
    expect(() => dirichletCodeLength([1, 1], 0)).toThrow(/alpha/)
    expect(() => multinomialParametricComplexity(0, 3)).toThrow(/k >= 1/)
  })
})

describe('two-part codes', () => {
  it('states a parameter for half a bit per doubling of the data', () => {
    expect(parameterBits(4, 256)).toBe(16)
    expect(parameterBits(1, 1024) - parameterBits(1, 512)).toBeCloseTo(0.5, 14)
    expect(gaussianDataBits(1, 2)).toBeCloseTo(Math.log2((2 * Math.PI * Math.E) / 2), 12)
    expect(() => gaussianDataBits(0, 5)).toThrow(/rss/)
  })

  /** A cubic with noise from a seeded generator. */
  const cubic = (seed: number, n: number): { x: number[]; y: number[] } => {
    const x = Array.from({ length: n }, (_, i) => -1 + (2 * i) / (n - 1))
    const noise = memorylessSource(key(seed), [1, 1, 1, 1, 1, 1, 1, 1], n).symbols
    return { x, y: x.map((v, i) => 1 - 2 * v + 3 * v ** 3 + 0.15 * ((noise[i] ?? 0) - 3.5)) }
  }

  it('the total is the BIC in bits, which a normal-equations fit reproduces', () => {
    const { x, y } = cubic(1, 60)
    const rows = polynomialMdl(x, y, 5)
    for (const r of rows) {
      const d = r.complexity
      // Independent fit: the normal equations on x scaled to [-1, 1] (already so here).
      const design = x.map((v) => Array.from({ length: d + 1 }, (_, j) => v ** j))
      const ata = Array.from({ length: d + 1 }, (_, a) =>
        Array.from({ length: d + 1 }, (_, b) => design.reduce((s, row) => s + (row[a] ?? 0) * (row[b] ?? 0), 0)),
      )
      const aty = Array.from({ length: d + 1 }, (_, a) =>
        design.reduce((s, row, i) => s + (row[a] ?? 0) * (y[i] ?? 0), 0),
      )
      const beta = toFlat(solve(ata, aty))
      const rss = design.reduce(
        (s, row, i) => s + (row.reduce((p, v, j) => p + v * (beta[j] ?? 0), 0) - (y[i] ?? 0)) ** 2,
        0,
      )
      const k = d + 2
      const bic = x.length * Math.log((2 * Math.PI * Math.E * rss) / x.length) + k * Math.log(x.length)
      expect(r.bic).toBeCloseTo(bic, 6)
      expect(r.total * 2 * Math.LN2).toBeCloseTo(bic, 6)
      expect(r.total).toBeCloseTo(r.modelBits + r.dataBits, 12)
    }
  })

  it('the data bits fall and the model bits rise with the degree, and the total is smallest at the true degree', () => {
    for (const seed of [1, 2, 3]) {
      const { x, y } = cubic(seed, 120)
      const rows = polynomialMdl(x, y, 7)
      for (let i = 1; i < rows.length; i++) {
        expect(rows[i]?.modelBits as number).toBeGreaterThan(rows[i - 1]?.modelBits as number)
        expect(rows[i]?.dataBits as number).toBeLessThanOrEqual((rows[i - 1]?.dataBits as number) + 1e-9)
      }
      expect(bestMdl(rows).complexity).toBe(3)
    }
  })

  it('rejects mismatched data, too high a degree and constant x', () => {
    expect(() => polynomialMdl([1, 2, 3], [1, 2], 1)).toThrow(/differ/)
    expect(() => polynomialMdl([1, 2, 3, 4], [1, 2, 3, 4], 3)).toThrow(/maxDegree/)
    expect(() => polynomialMdl([2, 2, 2, 2], [1, 2, 3, 4], 1)).toThrow(/constant/)
  })
})

describe('histogram bins by description length', () => {
  const uniform = (n: number): number[] => Array.from({ length: n }, (_, i) => (i + 0.5) / n)
  const bimodal = (n: number, seed: number): number[] => {
    const u = memorylessSource(key(seed), Array<number>(64).fill(1), n).symbols
    return u.map((v, i) => (i % 2 === 0 ? 0.25 : 0.75) + 0.04 * (v / 64 - 0.5))
  }

  it('picks one bin for flat data and many for structured data', () => {
    expect(bestMdl(histogramMdl(uniform(500), 40)).complexity).toBeLessThanOrEqual(2)
    const k = bestMdl(histogramMdl(bimodal(2000, 4), 60)).complexity
    expect(k).toBeGreaterThan(4)
    expect(k).toBeLessThan(40)
  })

  it('the model bits (the regret) rise with the number of bins and match the NML regret', () => {
    const rows = histogramMdl(bimodal(300, 5), 10)
    for (let i = 1; i < rows.length; i++)
      expect(rows[i]?.modelBits as number).toBeGreaterThan(rows[i - 1]?.modelBits as number)
    expect(rows[3]?.modelBits).toBeCloseTo(nmlRegret(4, 300), 12)
  })

  it('the best number of bins grows with the amount of data', () => {
    const bins = (n: number): number =>
      bestMdl(
        histogramMdl(
          bimodal(n, 6).map((v, i) => v + 0.1 * Math.sin(i)),
          80,
        ),
      ).complexity
    expect(bins(4000)).toBeGreaterThan(bins(200))
  })

  it('rejects constant data and a bad precision', () => {
    expect(() => histogramMdl([1, 1, 1], 3)).toThrow(/constant/)
    expect(() => histogramMdl([1, 2, 3], 3, 0)).toThrow(/precision/)
    expect(() => histogramMdl([1], 3)).toThrow(/two observations/)
  })
})

describe('model selection as an algorithm', () => {
  const rows = [
    { complexity: 1, modelBits: 1, dataBits: 9, total: 10 },
    { complexity: 2, modelBits: 2, dataBits: 4, total: 6 },
    { complexity: 3, modelBits: 3, dataBits: 3.5, total: 6.5 },
    { complexity: 4, modelBits: 4, dataBits: 3.4, total: 7.4 },
  ]

  it('scores one candidate per step and tracks the best', () => {
    const run = runAlgorithm(mdlSelectionAlgorithm(rows), { seed: 0, maxSteps: 10 })
    expect(run.steps).toBe(4)
    expect(run.state.best).toBe(2)
    expect(run.state.table).toHaveLength(16)
    expect(bestMdl(rows).complexity).toBe(2)
  })

  it('resumes exactly and rejects no candidates', () => {
    const alg = mdlSelectionAlgorithm(rows)
    const early = runAlgorithm(alg, { seed: 0, maxSteps: 2 })
    const resumed = runAlgorithm(alg, {
      seed: 0,
      maxSteps: 5,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(resumed.state.table).toEqual(runAlgorithm(alg, { seed: 0, maxSteps: 10 }).state.table)
    expect(() => mdlSelectionAlgorithm([])).toThrow(/no candidates/)
    expect(() => bestMdl([])).toThrow(/no candidates/)
  })
})
