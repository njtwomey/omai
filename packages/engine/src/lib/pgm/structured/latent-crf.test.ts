import { describe, expect, it } from 'vitest'
import { analyticObjective, lbfgs } from '../../optim/index.ts'
import { key, normal, randint } from '../../random/index.ts'
import { toFlat, type Value } from '../../tensor/index.ts'
import {
  type CrfInput,
  crfDecode,
  crfExpectedFeatureCounts,
  crfFeatureCounts,
  crfScore,
  linearChainCrf,
  withCrfParameters,
} from './crf.ts'
import { hiddenStateCrf, hiddenStateOracle, type PartialLabelling, partialLabelOracle } from './latent-crf.ts'
import type { StructuredOracle } from './oracle.ts'

const randomVector = (seed: number, n: number, scale = 1): Float64Array =>
  Float64Array.from(toFlat(normal(key(seed), [n])), (v) => scale * v)

const words = ['w0', 'w1', 'w2', 'w3']
const lse = (xs: readonly number[]): number => {
  const m = Math.max(...xs)
  return m + Math.log(xs.reduce((a, v) => a + Math.exp(v - m), 0))
}

function paths(n: number, K: number): number[][] {
  let all: number[][] = [[]]
  for (let t = 0; t < n; t++) all = all.flatMap((p) => Array.from({ length: K }, (_, k) => [...p, k]))
  return all
}

/** The latent negative log-likelihood of examples: free log Z minus clamped log Z, and its gradient. */
function latentObjective<Y>(
  oracle: StructuredOracle<CrfInput, Y>,
  data: readonly { x: CrfInput; y: Y }[],
  w: ArrayLike<number>,
  c2 = 0,
): { value: number; gradient: Float64Array } {
  const gradient = new Float64Array(oracle.size)
  let value = 0
  for (const { x, y } of data) {
    const free = oracle.expectations(w, x)
    const clamped = oracle.clamped?.(w, x, y) as NonNullable<ReturnType<NonNullable<typeof oracle.clamped>>>
    value += free.logPartition - clamped.logPartition
    for (let q = 0; q < gradient.length; q++)
      gradient[q] = (gradient[q] ?? 0) + (free.counts[q] ?? 0) - (clamped.counts[q] ?? 0)
  }
  for (let q = 0; q < gradient.length; q++) {
    value += c2 * (w[q] ?? 0) ** 2
    gradient[q] = (gradient[q] ?? 0) + 2 * c2 * (w[q] ?? 0)
  }
  return { value, gradient }
}

function finiteDifferenceGap(
  f: (w: Float64Array) => { value: number; gradient: Float64Array },
  w: Float64Array,
): number {
  const g = f(w).gradient
  let gap = 0
  for (let q = 0; q < w.length; q++) {
    const h = 1e-6
    const up = Float64Array.from(w)
    const down = Float64Array.from(w)
    up[q] = (up[q] ?? 0) + h
    down[q] = (down[q] ?? 0) - h
    gap = Math.max(gap, Math.abs((f(up).value - f(down).value) / (2 * h) - (g[q] ?? 0)))
  }
  return gap
}

const minimise = (
  f: (w: Float64Array) => { value: number; gradient: Float64Array },
  x0: Float64Array,
): Float64Array => {
  const r = lbfgs(
    analyticObjective({
      value: (w) => f(Float64Array.from(w)).value,
      gradient: (w) => f(Float64Array.from(w)).gradient,
    }),
    x0,
    { gradTol: 1e-7, maxIterations: 300 },
  )
  return Float64Array.from(toFlat(r.x as Value))
}

describe('the partial-label CRF oracle', () => {
  const labels = ['A', 'B', 'C']
  const crf = linearChainCrf(labels, words)
  const oracle = partialLabelOracle(crf)
  const w = randomVector(3, oracle.size, 0.8)
  const x: readonly Record<string, number>[] = [{ w0: 1 }, { w1: 1, w2: 0.5 }, { w3: 1 }, { w0: 2 }]
  const clamped = (y: PartialLabelling): { counts: Float64Array; logPartition: number } =>
    (oracle.clamped as NonNullable<typeof oracle.clamped>)(w, x, y)

  /** The exact clamped log Z and expected counts by enumerating the compatible labellings. */
  function enumerated(y: PartialLabelling): { logZ: number; counts: Float64Array } {
    const model = withCrfParameters(crf, w)
    const ok = paths(x.length, 3).filter((p) =>
      p.every((k, t) => {
        const set = y[t] ?? null
        return (
          set === null ||
          (typeof set === 'object' ? set : [set]).some((l) => (typeof l === 'number' ? l : labels.indexOf(l)) === k)
        )
      }),
    )
    const scores = ok.map((p) => crfScore(model, x, p))
    const logZ = lse(scores)
    const counts = new Float64Array(oracle.size)
    ok.forEach((p, i) => {
      const prob = Math.exp((scores[i] ?? 0) - logZ)
      for (const [q, v] of crfFeatureCounts(crf, x, p).entries()) counts[q] = (counts[q] ?? 0) + prob * v
    })
    return { logZ, counts }
  }

  it('clamped sums over the compatible labellings only', () => {
    for (const y of [
      ['A', null, ['B', 'C'], 'C'],
      [null, null, null, null],
      [['A', 'B'], 'B', null, ['A']],
    ] as PartialLabelling[]) {
      const truth = enumerated(y)
      const r = clamped(y)
      expect(r.logPartition).toBeCloseTo(truth.logZ, 10)
      for (const [q, v] of r.counts.entries()) expect(v).toBeCloseTo(truth.counts[q] ?? Number.NaN, 10)
    }
  })

  it('with every label allowed the example has probability one and the clamped sums are the free ones', () => {
    const free = oracle.expectations(w, x)
    for (const y of [[null, null, null, null], Array.from({ length: 4 }, () => labels)] as PartialLabelling[]) {
      const r = clamped(y)
      expect(r.logPartition).toBeCloseTo(free.logPartition, 12)
      for (const [q, v] of r.counts.entries()) expect(v).toBeCloseTo(free.counts[q] ?? Number.NaN, 12)
      expect(latentObjective(oracle, [{ x, y }], w).value).toBeCloseTo(0, 12)
    }
  })

  it('with one allowed label at every position it is the supervised CRF objective', () => {
    const y = ['B', 'A', 'C', 'C']
    const r = clamped(y)
    expect(r.logPartition).toBeCloseTo(crfScore(withCrfParameters(crf, w), x, y), 12)
    const phi = crfFeatureCounts(crf, x, y)
    for (const [q, v] of r.counts.entries()) expect(v).toBeCloseTo(phi[q] ?? Number.NaN, 12)
    const free = crfExpectedFeatureCounts(withCrfParameters(crf, w), x)
    const objective = latentObjective(oracle, [{ x, y }], w)
    expect(objective.value).toBeCloseTo(free.logPartition - crfScore(withCrfParameters(crf, w), x, y), 12)
    expect(objective.value).toBeCloseTo(-(crfScore(withCrfParameters(crf, w), x, y) - free.logPartition), 12)
  })

  it('the negative log-likelihood of a partial example is the supervised one marginalised over the compatible labellings', () => {
    const y: PartialLabelling = ['A', null, ['B', 'C'], 'C']
    const model = withCrfParameters(crf, w)
    const free = crfExpectedFeatureCounts(model, x).logPartition
    const compatible = paths(x.length, 3).filter((p) => p[0] === 0 && (p[2] === 1 || p[2] === 2) && p[3] === 2)
    const likelihood = compatible.reduce((a, p) => a + Math.exp(crfScore(model, x, p) - free), 0)
    expect(latentObjective(oracle, [{ x, y }], w).value).toBeCloseTo(-Math.log(likelihood), 10)
  })

  it('has the gradient of finite differences', () => {
    const data = [
      { x, y: ['A', null, ['B', 'C'], 'C'] as PartialLabelling },
      { x: [{ w1: 1 }, { w2: 1 }], y: [null, 'B'] as PartialLabelling },
    ]
    expect(
      finiteDifferenceGap((v) => latentObjective(oracle, data, v, 0.05), randomVector(4, oracle.size, 0.5)),
    ).toBeLessThan(1e-6)
  })

  it('decodes with cost augmentation against the gold sets, and the loss counts labels outside them', () => {
    const gold: PartialLabelling = ['A', null, ['B', 'C'], 'C']
    expect(oracle.loss(gold, [0, 2, 0, 1])).toBe(2)
    expect(oracle.loss(gold, [0, 1, 1, 2])).toBe(0)
    const model = withCrfParameters(crf, w)
    const augmented = paths(x.length, 3).map((p) => crfScore(model, x, p) + oracle.loss(gold, p))
    const r = oracle.decode(w, x, { against: gold })
    expect(r.score).toBeCloseTo(Math.max(...augmented), 10)
    expect(oracle.decode(w, x).score).toBeCloseTo(crfDecode(model, x).score, 10)
  })

  it('has feature counts only for a complete labelling', () => {
    expect(oracle.features(x, ['A', 'B', 'C', 'A'])[0]).toBe(crfFeatureCounts(crf, x, ['A', 'B', 'C', 'A'])[0])
    expect(() => oracle.features(x, ['A', null, 'C', 'A'])).toThrow()
    expect(() => oracle.features(x, ['A', ['A', 'B'], 'C', 'A'])).toThrow()
    expect(() => oracle.clamped?.(w, x, ['A', 'Z', 'C', 'A'])).toThrow()
  })

  describe('learning with half of the labels missing', () => {
    const base = linearChainCrf(['A', 'B', 'C'], words)
    const truth = withCrfParameters(base, randomVector(5, oracle.size, 1.5))
    const sequences = (seed: number, count: number): CrfInput[] => {
      const lengths = Array.from(toFlat(randint(key(seed), [count], 5, 9)))
      const ids = Array.from(toFlat(randint(key(seed + 1), [count * 8], 0, words.length)))
      return lengths.map((len, n) =>
        Array.from({ length: len }, (_, t) => ({ [words[ids[n * 8 + t] ?? 0] ?? 'w0']: 1 })),
      )
    }
    const label = (xs: CrfInput[]): { x: CrfInput; y: string[] }[] =>
      xs.map((s) => ({ x: s, y: crfDecode(truth, s).labels }))
    const full = label(sequences(11, 60))
    const hidden = Array.from(toFlat(randint(key(13), [60 * 8], 0, 2)))
    const partial = full.map(({ x: s, y }, n) => ({
      x: s,
      y: y.map((l, t) => (hidden[n * 8 + t] === 1 ? null : l)) as PartialLabelling,
    }))
    const test = label(sequences(21, 40))
    const accuracy = (model: ReturnType<typeof withCrfParameters>): number => {
      let right = 0
      let total = 0
      for (const { x: s, y } of test) {
        const decoded = crfDecode(model, s).labels
        right += decoded.filter((l, t) => l === y[t]).length
        total += y.length
      }
      return right / total
    }

    it('recovers the planted chain', () => {
      const w = minimise((v) => latentObjective(oracle, partial, v, 0.02), new Float64Array(oracle.size))
      const missing = partial.flatMap((p) => p.y).filter((l) => l === null).length / partial.flatMap((p) => p.y).length
      expect(missing).toBeGreaterThan(0.4)
      expect(missing).toBeLessThan(0.6)
      expect(accuracy(withCrfParameters(base, w))).toBeGreaterThan(0.9)
    })
  })
})

describe('the hidden-state CRF oracle', () => {
  const labels = ['A', 'B']
  const H = 2
  const model = hiddenStateCrf(labels, words, H)
  const oracle = hiddenStateOracle(model)
  const w = randomVector(6, oracle.size, 0.7)
  const x: readonly Record<string, number>[] = [{ w0: 1 }, { w1: 1, w2: 0.5 }, { w3: 1 }, { w0: 1 }]
  const hiddenOf = (s: number): number => Math.floor(s / H)

  /** Every hidden path with its score, and the ones that map onto y. */
  function enumerate(y: readonly number[]): {
    freeLogZ: number
    freeCounts: Float64Array
    logZ: number
    counts: Float64Array
    best: { path: number[]; score: number }
    all: { path: number[]; score: number }[]
  } {
    const crf = withCrfParameters(model.crf, w)
    const all = paths(x.length, 4).map((path) => ({ path, score: crfScore(crf, x, path) }))
    const freeLogZ = lse(all.map((p) => p.score))
    const sum = (rows: typeof all): { logZ: number; counts: Float64Array } => {
      const logZ = lse(rows.map((p) => p.score))
      const counts = new Float64Array(oracle.size)
      for (const p of rows) {
        const prob = Math.exp(p.score - logZ)
        for (const [q, v] of crfFeatureCounts(model.crf, x, p.path).entries()) counts[q] = (counts[q] ?? 0) + prob * v
      }
      return { logZ, counts }
    }
    const consistent = all.filter((p) => p.path.every((s, t) => hiddenOf(s) === y[t]))
    const c = sum(consistent)
    const best = consistent.reduce((a, p) => (p.score > a.score ? p : a))
    return { freeLogZ, freeCounts: sum(all).counts, ...c, best, all }
  }

  it('sums over the hidden paths that map onto the gold labels (clamped) and over all of them (free)', () => {
    for (const y of [
      [0, 1, 1, 0],
      [1, 1, 1, 1],
    ]) {
      const truth = enumerate(y)
      const r = oracle.clamped?.(
        w,
        x,
        y.map((k) => labels[k] as string),
      ) as { counts: Float64Array; logPartition: number }
      expect(r.logPartition).toBeCloseTo(truth.logZ, 10)
      for (const [q, v] of r.counts.entries()) expect(v).toBeCloseTo(truth.counts[q] ?? Number.NaN, 10)
      const free = oracle.expectations(w, x)
      expect(free.logPartition).toBeCloseTo(truth.freeLogZ, 10)
      for (const [q, v] of free.counts.entries()) expect(v).toBeCloseTo(truth.freeCounts[q] ?? Number.NaN, 10)
    }
  })

  it('has the gradient of finite differences', () => {
    const data = [
      { x, y: ['A', 'B', 'B', 'A'] },
      { x: [{ w2: 1 }, { w3: 1 }, { w1: 1 }], y: ['B', 'B', 'A'] },
    ]
    expect(
      finiteDifferenceGap((v) => latentObjective(oracle, data, v, 0.05), randomVector(8, oracle.size, 0.5)),
    ).toBeLessThan(1e-6)
  })

  it('completes an example with its best consistent hidden path', () => {
    const y = [0, 1, 1, 0]
    const truth = enumerate(y)
    const r = oracle.completion(w, x, y)
    expect(r.score).toBeCloseTo(truth.best.score, 10)
    expect(r.path.map(hiddenOf)).toEqual(y)
    const phi = crfFeatureCounts(model.crf, x, r.path)
    expect(Array.from(r.counts)).toEqual(Array.from(phi))
  })

  it('decodes by marginals (score: the mass of the decoded labels) or by Viterbi, with cost augmentation', () => {
    const truth = enumerate([0, 0, 0, 0])
    const marginals = oracle.labelMarginals(w, x)
    for (let t = 0; t < x.length; t++)
      for (let k = 0; k < 2; k++) {
        const exact = truth.all.reduce(
          (a, p) => a + (hiddenOf(p.path[t] ?? 0) === k ? Math.exp(p.score - truth.freeLogZ) : 0),
          0,
        )
        expect(marginals[t]?.[k]).toBeCloseTo(exact, 10)
      }
    const byMarginal = oracle.decode(w, x)
    expect(byMarginal.y).toEqual(marginals.map((row) => ((row[1] ?? 0) > (row[0] ?? 0) ? 1 : 0)))
    expect(byMarginal.score).toBeCloseTo(enumerate(byMarginal.y as number[]).logZ, 10)
    const viterbiOracle = hiddenStateOracle(model, { decoding: 'viterbi' })
    const best = truth.all.reduce((a, p) => (p.score > a.score ? p : a))
    expect(viterbiOracle.decode(w, x).y).toEqual(best.path.map(hiddenOf))
    const gold = ['A', 'B', 'A', 'B']
    const augmented = truth.all.map(
      (p) => p.score + p.path.reduce((a, s, t) => a + (hiddenOf(s) === (gold[t] === 'A' ? 0 : 1) ? 0 : 1), 0),
    )
    expect(oracle.decode(w, x, { against: gold }).score).toBeCloseTo(Math.max(...augmented), 10)
    expect(oracle.loss(gold, ['A', 'A', 'A', 'B'])).toBe(1)
  })

  it('has no feature counts of its own, because the hidden path is latent', () => {
    expect(() => oracle.features(x, ['A', 'B', 'B', 'A'])).toThrow()
  })

  it('with one substate it is the linear-chain CRF', () => {
    const plain = linearChainCrf(['A', 'B'], words)
    const single = hiddenStateOracle(hiddenStateCrf(['A', 'B'], words, 1))
    const v = randomVector(9, single.size, 0.8)
    const y = ['B', 'A', 'A', 'B']
    const clamped = single.clamped?.(v, x, y) as { counts: Float64Array; logPartition: number }
    const crf = withCrfParameters(plain, v)
    expect(clamped.logPartition).toBeCloseTo(crfScore(crf, x, y), 12)
    for (const [q, c] of clamped.counts.entries())
      expect(c).toBeCloseTo(crfFeatureCounts(plain, x, y)[q] ?? Number.NaN, 12)
    const free = single.expectations(v, x)
    const reference = crfExpectedFeatureCounts(crf, x)
    expect(free.logPartition).toBeCloseTo(reference.logPartition, 12)
    for (const [q, c] of free.counts.entries()) expect(c).toBeCloseTo(reference.counts[q] ?? Number.NaN, 12)
    expect(single.decode(v, x, { against: y }).score).toBeCloseTo(
      Math.max(
        ...paths(4, 2).map((p) => crfScore(crf, x, p) + p.filter((k, t) => k !== (y[t] === 'A' ? 0 : 1)).length),
      ),
      10,
    )
  })

  it('learns a label whose hidden substates alternate, which a chain CRF with the same features cannot', () => {
    // "walk" alternates the attributes u and v, "stand" repeats one of them: u alone says neither.
    const sequence = (pattern: string): CrfInput => [...pattern].map((c) => ({ [c]: 1 }))
    const examples = [
      { x: sequence('uvuvuv'), y: Array<string>(6).fill('walk') },
      { x: sequence('vuvuvu'), y: Array<string>(6).fill('walk') },
      { x: sequence('uuuuuu'), y: Array<string>(6).fill('stand') },
      { x: sequence('vvvvvv'), y: Array<string>(6).fill('stand') },
    ]
    const vocabulary = ['u', 'v']
    const hidden = hiddenStateCrf(['walk', 'stand'], vocabulary, 2)
    const hiddenOracle = hiddenStateOracle(hidden)
    const wHidden = minimise(
      (v) => latentObjective(hiddenOracle, examples, v, 0.001),
      randomVector(12, hiddenOracle.size, 0.3),
    )
    const chain = partialLabelOracle(linearChainCrf(['walk', 'stand'], vocabulary))
    const wChain = minimise((v) => latentObjective(chain, examples, v, 0.001), new Float64Array(chain.size))
    const errors = (decode: (s: CrfInput) => readonly (string | number)[]): number =>
      examples.reduce(
        (a, { x: s, y }) =>
          a + decode(s).filter((l, t) => (typeof l === 'number' ? ['walk', 'stand'][l] : l) !== y[t]).length,
        0,
      )
    const hiddenErrors = errors((s) => hiddenOracle.decode(wHidden, s).y)
    const chainErrors = errors((s) => chain.decode(wChain, s).y as readonly number[])
    expect(hiddenErrors).toBe(0)
    expect(chainErrors).toBeGreaterThan(5)
    expect(latentObjective(hiddenOracle, examples, wHidden).value).toBeLessThan(
      latentObjective(chain, examples, wChain).value / 4,
    )
  })
})
