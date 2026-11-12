import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { minOf } from '../base/index.ts'
import { sum, type TensorLike, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  aliasSample,
  aliasTable,
  bernoulli,
  beta,
  binomial,
  categorical,
  cauchy,
  chisquare,
  choice,
  defaultRng,
  dirichlet,
  exponential,
  foldIn,
  Generator,
  gamma,
  gammaVariates,
  geometric,
  gumbel,
  halton,
  key,
  keyFromPath,
  laplace,
  logistic,
  multinomial,
  normal,
  permutation,
  philox4x32,
  poisson,
  poissonEach,
  randint,
  randomBits,
  randomSplit,
  split,
  studentT,
  truncatedNormal,
  uniform,
  vonMises,
} from './index.ts'

const mean = (xs: ArrayLike<number>): number => Array.from(xs).reduce((a, b) => a + b, 0) / xs.length
const variance = (xs: ArrayLike<number>): number => {
  const m = mean(xs)
  return Array.from(xs).reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)
}

describe('Philox4x32-10', () => {
  it('matches the Random123 known-answer vectors', () => {
    expect(philox4x32([0, 0, 0, 0], [0, 0])).toEqual([0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8])
    expect(philox4x32([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff])).toEqual([
      0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd,
    ])
    expect(philox4x32([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0])).toEqual([
      0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1,
    ])
  })
})

describe('keys', () => {
  it('are reproducible and independent', () => {
    expect(toFlat(uniform(key(7), [5]))).toEqual(toFlat(uniform(key(7), [5])))
    const [a, b] = split(key(7))
    expect(toFlat(uniform(a as never, [3]))).not.toEqual(toFlat(uniform(b as never, [3])))
    expect(foldIn(key(1), 2).words).toEqual(foldIn(key(1), 2).words)
    expect(keyFromPath(3, 'a', 1).words).not.toEqual(keyFromPath(3, 'a', 2).words)
  })

  it('rejects invalid seeds', () => {
    expect(() => key(-1)).toThrow(/seed/)
    expect(() => key(1.5)).toThrow(/seed/)
  })
})

describe('samplers have the right moments', () => {
  const n = 40000
  it('uniform and normal', () => {
    const u = uniform(key(1), [n]).data
    expect(mean(u)).toBeCloseTo(0.5, 2)
    expect(variance(u)).toBeCloseTo(1 / 12, 2)
    const z = normal(key(2), [n], { loc: 1, scale: 2 }).data
    expect(mean(z)).toBeCloseTo(1, 1)
    expect(Math.sqrt(variance(z))).toBeCloseTo(2, 1)
  })

  it('exponential, gamma and beta', () => {
    expect(mean(exponential(key(3), [n], 2).data)).toBeCloseTo(0.5, 2)
    const g = gamma(key(4), [n], 3).data
    expect(mean(g)).toBeCloseTo(3, 1)
    expect(variance(g)).toBeCloseTo(3, 0)
    expect(mean(gamma(key(5), [n], 0.3).data)).toBeCloseTo(0.3, 1)
    expect(mean(beta(key(6), [n], 2, 6).data)).toBeCloseTo(0.25, 2)
  })

  it('truncated normal stays in range with the right mean', () => {
    const t = truncatedNormal(key(7), [n], 2, Infinity).data
    expect(minOf(Array.from(t.slice(0, 1000)))).toBeGreaterThanOrEqual(2)
    // E[Z | Z > 2] = phi(2) / (1 - Phi(2)) = 2.373215
    expect(mean(t)).toBeCloseTo(2.3732, 1)
  })

  it('poissonEach draws each count from its own key', () => {
    const rates = [0.5, 3, 80, 0, 12]
    const counts = poissonEach(rates, (i) => foldIn(key(21), 100 + i))
    expect(Array.from(counts.data)).toEqual(rates.map((r, i) => poisson(foldIn(key(21), 100 + i), r, [1]).data[0]))
    expect(counts.shape).toEqual([5])
    expect(() => poissonEach([1, -1], (i) => key(i))).toThrow()
  })

  it('discrete samplers', () => {
    expect(mean(poisson(key(8), 3, [n]).data)).toBeCloseTo(3, 1)
    expect(mean(poisson(key(9), 80, [n]).data)).toBeCloseTo(80, 0)
    expect(mean(binomial(key(10), 20, 0.3, [n]).data)).toBeCloseTo(6, 1)
    expect(mean(binomial(key(11), 20, 0.8, [n]).data)).toBeCloseTo(16, 1)
    expect(mean(bernoulli(key(12), 0.3, [n]).data)).toBeCloseTo(0.3, 2)
    const c = categorical(key(13), [1, 2, 7], [n]).data
    expect(mean(Array.from(c, (v) => (v === 2 ? 1 : 0)))).toBeCloseTo(0.7, 2)
    const lc = categorical(key(14), [0, Math.log(3)], [n], { logits: true }).data
    expect(mean(lc)).toBeCloseTo(0.75, 2)
    const a = aliasSample(key(15), aliasTable([1, 2, 7]), [n]).data
    expect(mean(Array.from(a, (v) => (v === 2 ? 1 : 0)))).toBeCloseTo(0.7, 2)
    expect(randint(key(16), [n], -2, 3).data.every((v) => v >= -2 && v < 3)).toBe(true)
  })

  it('dirichlet rows sum to one and von Mises concentrates', () => {
    const d = dirichlet(key(17), [0.5, 1, 2], 1000)
    for (let r = 0; r < 1000; r++) {
      expect((d.data[r * 3] ?? 0) + (d.data[r * 3 + 1] ?? 0) + (d.data[r * 3 + 2] ?? 0)).toBeCloseTo(1, 12)
    }
    expect(mean(Array.from(vonMises(key(18), [n], 1, 50).data))).toBeCloseTo(1, 1)
  })
})

describe('permutations and choices', () => {
  it('permutation is a permutation', () => {
    expect(Array.from(permutation(key(1), 50).data).sort((a, b) => a - b)).toEqual(
      Array.from({ length: 50 }, (_, i) => i),
    )
  })

  it('choice without replacement is distinct; weighted choice respects zero weights', () => {
    expect(new Set(choice(key(2), 10, 10, { replace: false }).data).size).toBe(10)
    const w = choice(key(3), 4, 3, { replace: false, weights: [1, 0, 1, 1] })
    expect(Array.from(w.data)).not.toContain(1)
  })

  it('randomSplit partitions', () => {
    const parts = randomSplit(key(4), 10, [0.7, 0.3])
    expect(parts.map((p) => p.shape[0])).toEqual([7, 3])
    expect(new Set([...(parts[0]?.data ?? []), ...(parts[1]?.data ?? [])]).size).toBe(10)
  })

  it('Generator is reproducible', () => {
    expect(toFlat(new Generator(5).normal([4]))).toEqual(toFlat(new Generator(5).normal([4])))
  })

  it('halton starts with the radical inverses', () => {
    expect(Array.from(halton(3, 1).data)).toEqual([0.5, 0.25, 0.75])
  })
})

describe('continuous edge cases', () => {
  it('truncated normals on narrow and far intervals stay inside and cost little', () => {
    for (const [lo, hi] of [
      [10, 10.0001],
      [-1e-6, 1e-6],
      [0.4, 0.400001],
      [-40, -39],
      [30, Infinity],
      [-Infinity, -30],
    ] as const) {
      const x = truncatedNormal(key(1), [2000], lo, hi).data
      expect(x.every((v) => v >= lo && v <= hi)).toBe(true)
    }
    // E[Z | 0.4 < Z < 0.4 + h] is the midpoint to first order.
    expect(mean(truncatedNormal(key(2), [4000], 0.4, 0.401).data)).toBeCloseTo(0.4005, 4)
    expect(() => truncatedNormal(key(0), [1], 1, 1)).toThrow(/upper > lower/)
  })

  it('beta and Dirichlet with tiny shapes stay finite with the right means', () => {
    const b = beta(key(3), [20000], 0.01, 0.03).data
    expect(b.every((v) => v >= 0 && v <= 1)).toBe(true)
    // Nearly all the mass sits at the ends, with P(near 1) = a / (a + b) = 0.25.
    expect(mean(b)).toBeCloseTo(0.25, 1)
    const d = dirichlet(key(4), [0.01, 0.01, 0.02], 2000).data
    expect(d.every((v) => Number.isFinite(v))).toBe(true)
    for (let r = 0; r < 2000; r++) {
      expect((d[3 * r] ?? 0) + (d[3 * r + 1] ?? 0) + (d[3 * r + 2] ?? 0)).toBeCloseTo(1, 12)
    }
    expect(() => beta(key(0), [1], 0, 1)).toThrow(/positive/)
    expect(() => dirichlet(key(0), [1, -1], 1)).toThrow(/positive/)
  })

  it('laplace never reaches an infinite endpoint; gammaVariates draws one per shape', () => {
    expect(laplace(key(5), [10000]).data.every(Number.isFinite)).toBe(true)
    const g = gammaVariates(key(6), [0.5, 2, 30])
    expect(g.length).toBe(3)
    expect(g[2]).toBeGreaterThan(10)
  })
})

describe('discrete edge cases', () => {
  it('binomial at the boundaries and for very large n', () => {
    expect(Array.from(binomial(key(1), 10, 0, [3]).data)).toEqual([0, 0, 0])
    expect(Array.from(binomial(key(1), 10, 1, [3]).data)).toEqual([10, 10, 10])
    expect(Array.from(binomial(key(1), 0, 0.5, [2]).data)).toEqual([0, 0])
    const big = binomial(key(2), 2 ** 30, 0.25, [2000]).data
    expect(mean(big) / 2 ** 28).toBeCloseTo(1, 4)
    expect(() => binomial(key(0), 2.5, 0.5, [1])).toThrow(/integer/)
  })

  it('multinomial rows sum to n, zero probabilities get nothing, and large n is cheap', () => {
    const m = multinomial(key(3), 1e9, [0.2, 0, 0.5, 0.3], 500).data
    for (let r = 0; r < 500; r++) {
      expect((m[4 * r] ?? 0) + (m[4 * r + 1] ?? 0) + (m[4 * r + 2] ?? 0) + (m[4 * r + 3] ?? 0)).toBe(1e9)
      expect(m[4 * r + 1]).toBe(0)
    }
    expect(Array.from(multinomial(key(3), 5, [0, 0, 1], 1).data)).toEqual([0, 0, 5])
    expect(() => multinomial(key(0), -1, [1], 1)).toThrow(/non-negative/)
  })

  it('weighted choice without replacement picks one item with probability proportional to its weight', () => {
    const w = [1, 2, 3, 4]
    const counts = [0, 0, 0, 0]
    for (let t = 0; t < 8000; t++) {
      const i = choice(key(t), 4, 1, { replace: false, weights: w }).data[0] ?? 0
      counts[i] = (counts[i] ?? 0) + 1
    }
    counts.forEach((c, i) => {
      expect(c / 8000).toBeCloseTo((w[i] ?? 0) / 10, 1)
    })
  })
})

describe('bits and generators', () => {
  it('randomBits is deterministic, uint32, and any prefix of a longer draw', () => {
    const a = randomBits(key(9), 10)
    expect(Array.from(randomBits(key(9), 7))).toEqual(Array.from(a.slice(0, 7)))
    expect(a.every((v) => Number.isInteger(v) && v >= 0 && v < 2 ** 32)).toBe(true)
  })

  it('Generator methods advance the key and reproduce from a seed', () => {
    const g = defaultRng(4)
    const first = Array.from(g.uniform([3]).data)
    expect(Array.from(g.uniform([3]).data)).not.toEqual(first)
    expect(Array.from(defaultRng(4).uniform([3]).data)).toEqual(first)
    expect(
      defaultRng(1)
        .integers(0, 5, [100])
        .data.every((v) => v >= 0 && v < 5),
    ).toBe(true)
    expect(Array.from(defaultRng(1).permutation(5).data).sort()).toEqual([0, 1, 2, 3, 4])
  })
})

describe('degenerate parameters', () => {
  it('never alias key path parts: a part that would wrap or truncate is an error', () => {
    for (const part of [-1, 1.5, Number.NaN, 2 ** 32, Number.POSITIVE_INFINITY])
      expect(() => keyFromPath(0, 'a', part), String(part)).toThrow(expect.objectContaining({ code: 'value' }))
    expect(() => foldIn(key(0), -1)).toThrow(/integer in/)
    expect(keyFromPath(0, 'a', 0xffffffff).words).not.toEqual(keyFromPath(0, 'a', 0).words)
  })

  it('reject parameters that are invalid or non-finite, and draws that do not fit an int32', () => {
    const k = key(3)
    const invalid: [string, () => unknown][] = [
      ['normal with a negative scale', () => normal(k, [2], { scale: -1 })],
      ['normal with an infinite scale', () => normal(k, [2], { scale: Number.POSITIVE_INFINITY })],
      ['exponential with rate 0', () => exponential(k, [2], 0)],
      ['laplace with a negative scale', () => laplace(k, [2], 0, -1)],
      ['logistic with a NaN scale', () => logistic(k, [2], 0, Number.NaN)],
      ['gumbel with a negative scale', () => gumbel(k, [2], 0, -1)],
      ['cauchy with a negative scale', () => cauchy(k, [2], 0, -1)],
      ['uniform with high below low', () => uniform(k, [2], { low: 1, high: 0 })],
      ['studentT with df 0', () => studentT(k, [2], 0)],
      ['chisquare with a negative df', () => chisquare(k, [2], -1)],
      ['vonMises with a negative kappa', () => vonMises(k, [2], 0, -1)],
      ['bernoulli with p above 1', () => bernoulli(k, 1.5, [2])],
      ['poisson with an infinite rate', () => poisson(k, Number.POSITIVE_INFINITY, [1])],
      ['geometric with p so small the draw wraps', () => geometric(k, 1e-18, [3])],
    ]
    for (const [name, draw] of invalid) expect(draw, name).toThrow(expect.objectContaining({ code: 'value' }))
    // a zero scale is a point mass, and an ordinary p keeps its values
    expect(Array.from(normal(k, [2], { loc: 3, scale: 0 }).data)).toEqual([3, 3])
    expect(Array.from(geometric(k, 1, [2]).data)).toEqual([1, 1])
    // log1p keeps the digits of a small p: the mean of the geometric is 1 / p
    expect(mean(geometric(k, 1e-4, [20000]).data) / 1e4).toBeCloseTo(1, 1)
  })
})

describe('tensor-in samplers (D-009: were number[])', () => {
  const w = [1, 2, 7]
  const rules: readonly [string, (x: TensorLike) => readonly number[]][] = [
    ['dirichlet', (x) => toFlat(dirichlet(key(1), x, 3))],
    ['categorical', (x) => toFlat(categorical(key(2), x, [20]))],
    ['multinomial', (x) => toFlat(multinomial(key(3), 50, x, 4))],
    ['aliasTable', (x) => toFlat(aliasSample(key(4), aliasTable(x), [20]))],
    ['aliasTable.probability', (x) => toFlat(aliasTable(x).probability)],
    ['choice', (x) => toFlat(choice(key(5), 3, 2, { replace: false, weights: x }))],
    ['randomSplit', (x) => randomSplit(key(6), 10, x).flatMap((p) => toFlat(p))],
  ]

  it.each(rules)(
    '%s: a tensor, a typed array and a nested array give the same draws; a matrix is a shape error; a traced weight is rejected',
    (name, run) => {
      const frac = name === 'randomSplit' ? [0.5, 0.25, 0.25] : w
      const same = run(frac)
      expect(run(tensor(frac))).toEqual(same)
      expect(run(Float64Array.from(frac))).toEqual(same)
      expect(() => run([frac, frac])).toThrow(expect.objectContaining({ code: 'shape' }))
      const traced = (x: Value): Value => {
        run(x)
        return sum(x)
      }
      expect(() => grad(traced)(tensor(frac))).toThrow(expect.objectContaining({ code: 'not-differentiable' }))
    },
  )
})
