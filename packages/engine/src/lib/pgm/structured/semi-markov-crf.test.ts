import { describe, expect, it } from 'vitest'
import { analyticObjective, lbfgs } from '../../optim/index.ts'
import { key, normal, randint } from '../../random/index.ts'
import { toFlat, type Value } from '../../tensor/index.ts'
import { crfDecode, crfExpectedFeatureCounts, crfMarginals, linearChainCrf, withCrfParameters } from './crf.ts'
import {
  type Segmentation,
  type SegmentInput,
  type SemiMarkovCrf,
  segmentationToTags,
  semiMarkovCrf,
  semiMarkovDecode,
  semiMarkovExpectedFeatureCounts,
  semiMarkovFeatureCounts,
  semiMarkovLayout,
  semiMarkovMarginals,
  semiMarkovOracle,
  semiMarkovScore,
  tagsToSegmentation,
} from './semi-markov-crf.ts'

const words = ['w0', 'w1', 'w2', 'w3']

const randomWeights = (seed: number, n: number, scale = 1): Float64Array =>
  Float64Array.from(toFlat(normal(key(seed), [n])), (v) => scale * v)

/** Sequences of random word items (a word, sometimes with a second attribute), lengths from low to high. */
function inputs(seed: number, count: number, low: number, high: number): SegmentInput[] {
  const lengths = Array.from(toFlat(randint(key(seed), [count], low, high + 1)))
  const ids = Array.from(toFlat(randint(key(seed + 1), [count * high], 0, words.length)))
  return lengths.map((len, n) =>
    Array.from({ length: len }, (_, t) => ({ [words[ids[n * high + t] ?? 0] ?? 'w0']: 1 })),
  )
}

/** Every segmentation of n items into segments of length at most L with labels from K. */
function allSegmentations(n: number, K: number, L: number): Segmentation[] {
  if (n === 0) return [[]]
  const out: Segmentation[] = []
  for (let l = 1; l <= Math.min(L, n); l++)
    for (const rest of allSegmentations(n - l, K, L))
      for (let k = 0; k < K; k++)
        out.push([{ label: k, start: 0, end: l }, ...rest.map((s) => ({ ...s, start: s.start + l, end: s.end + l }))])
  return out
}

const lse = (xs: readonly number[]): number => {
  const m = Math.max(...xs)
  return m + Math.log(xs.reduce((a, v) => a + Math.exp(v - m), 0))
}

/** The exact log Z, expected counts, segment marginals and best score by enumeration. */
function brute(model: SemiMarkovCrf, w: Float64Array, x: SegmentInput) {
  const all = allSegmentations(x.length, model.labels.length, model.maxLength)
  const scores = all.map((y) => semiMarkovScore(model, w, x, y))
  const logZ = lse(scores)
  const counts = new Float64Array(semiMarkovLayout(model).size)
  const segments = new Map<string, number>()
  all.forEach((y, i) => {
    const p = Math.exp((scores[i] ?? 0) - logZ)
    for (const [q, v] of semiMarkovFeatureCounts(model, x, y).entries()) counts[q] = (counts[q] ?? 0) + p * v
    for (const s of y) {
      const id = `${s.start}:${s.end}:${s.label}`
      segments.set(id, (segments.get(id) ?? 0) + p)
    }
  })
  return { all, scores, logZ, counts, segments }
}

for (const boundary of [false, true]) {
  describe(`semi-Markov CRF against enumeration of segmentations (boundary features ${boundary})`, () => {
    const model = semiMarkovCrf(['A', 'B', 'C'], words, 3, { boundary })
    const layout = semiMarkovLayout(model)
    const w = randomWeights(7, layout.size, 0.8)
    const xs = inputs(11, 4, 1, 5)

    it('the score is the inner product of the weights and the feature counts', () => {
      for (const x of xs) {
        const all = allSegmentations(x.length, 3, 3)
        const y = all[Math.floor(all.length / 2)] as Segmentation
        const phi = semiMarkovFeatureCounts(model, x, y)
        expect(phi.reduce((a, v, q) => a + v * (w[q] ?? 0), 0)).toBeCloseTo(semiMarkovScore(model, w, x, y), 10)
      }
    })

    it('log Z, expected counts and segment marginals equal the enumerated ones', () => {
      for (const x of xs) {
        const truth = brute(model, w, x)
        const r = semiMarkovExpectedFeatureCounts(model, w, x)
        expect(r.logPartition).toBeCloseTo(truth.logZ, 10)
        for (const [q, v] of r.counts.entries()) expect(v).toBeCloseTo(truth.counts[q] ?? Number.NaN, 10)
        const m = semiMarkovMarginals(model, w, x)
        expect(m.logPartition).toBeCloseTo(truth.logZ, 10)
        for (const s of m.segments)
          expect(s.probability).toBeCloseTo(truth.segments.get(`${s.start}:${s.end}:${s.label}`) ?? 0, 10)
        // the label marginals are the segment marginals summed over the segments covering an item
        for (const [i, row] of m.labels.entries()) expect(row.reduce((a, p) => a + p, 0)).toBeCloseTo(1, 10)
        expect(m.labels.length).toBe(x.length)
      }
    })

    it('Viterbi finds the best segmentation', () => {
      for (const x of xs) {
        const truth = brute(model, w, x)
        const best = Math.max(...truth.scores)
        const r = semiMarkovDecode(model, w, x)
        expect(r.score).toBeCloseTo(best, 10)
        expect(semiMarkovScore(model, w, x, r.y)).toBeCloseTo(best, 10)
      }
    })

    it('cost-augmented decoding maximises score plus loss, for the token and the segment loss', () => {
      for (const loss of ['token', 'segment'] as const) {
        const oracle = semiMarkovOracle(model, { loss })
        for (const x of xs) {
          const all = allSegmentations(x.length, 3, 3)
          const gold = all[Math.floor(all.length / 3)] as Segmentation
          const augmented = all.map((y) => semiMarkovScore(model, w, x, y) + oracle.loss(gold, y))
          const r = oracle.decode(w, x, { against: gold })
          expect(r.score).toBeCloseTo(Math.max(...augmented), 10)
          expect(semiMarkovScore(model, w, x, r.y) + oracle.loss(gold, r.y)).toBeCloseTo(r.score, 10)
        }
      }
    })
  })
}

describe('semi-Markov CRF with maximum length 1 is the linear-chain CRF', () => {
  const labels = ['A', 'B', 'C']
  const chain = withCrfParameters(linearChainCrf(labels, words), randomWeights(3, words.length * 3 + 9))
  const model = semiMarkovCrf(labels, words, 1)
  const layout = semiMarkovLayout(model)
  // the chain's parameters are state then transition weights: the content and transition blocks of the semi-Markov CRF
  const w = new Float64Array(layout.size)
  w.set(chain.stateWeights, layout.content)
  w.set(chain.transitionWeights, layout.transition)
  const x = inputs(5, 1, 6, 6)[0] as SegmentInput

  it('has the same log-partition function and expected counts', () => {
    const c = crfExpectedFeatureCounts(chain, x)
    const s = semiMarkovExpectedFeatureCounts(model, w, x)
    expect(s.logPartition).toBeCloseTo(c.logPartition, 10)
    const expected = new Float64Array(layout.size)
    expected.set(c.counts.slice(0, chain.stateWeights.length), layout.content)
    expected.set(c.counts.slice(chain.stateWeights.length), layout.transition)
    // each item is a segment of length one: the length counts are the label marginals summed over the items
    const labelMarginals = semiMarkovMarginals(model, w, x).labels
    for (let k = 0; k < 3; k++) expected[layout.length + k] = labelMarginals.reduce((a, row) => a + (row[k] ?? 0), 0)
    for (const [q, v] of s.counts.entries()) expect(v).toBeCloseTo(expected[q] ?? Number.NaN, 10)
  })

  it('has the same marginals and the same best labelling', () => {
    const post = crfMarginals(chain, x)
    for (const [i, row] of semiMarkovMarginals(model, w, x).labels.entries())
      for (const [k, p] of row.entries()) expect(p).toBeCloseTo(post[i]?.[k] ?? Number.NaN, 10)
    const decoded = semiMarkovDecode(model, w, x)
    const tags = decoded.y.map((s) => labels[s.label])
    const reference = crfDecode(chain, x)
    expect(tags).toEqual(reference.labels)
    expect(decoded.score).toBeCloseTo(reference.score, 10)
  })
})

describe('segmentations and BIO tags', () => {
  const labels = ['O', 'PER', 'LOC']

  it('convert both ways, with an outside segment of length one per outside item', () => {
    const tags = ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'B-LOC', 'I-LOC']
    const seg = tagsToSegmentation(labels, tags, { outside: 'O' })
    expect(seg.map((s) => [s.label, s.start, s.end])).toEqual([
      [1, 0, 2],
      [0, 2, 3],
      [0, 3, 4],
      [2, 4, 5],
      [2, 5, 7],
    ])
    expect(segmentationToTags(labels, seg, { outside: 'O' })).toEqual(tags)
    // an outside segment longer than one item gives the same tags
    expect(segmentationToTags(labels, [{ label: 0, start: 0, end: 3 }], { outside: 'O' })).toEqual(['O', 'O', 'O'])
  })

  it('rejects an O without an outside label and an unknown entity type', () => {
    expect(() => tagsToSegmentation(labels, ['O'])).toThrow()
    expect(() => tagsToSegmentation(labels, ['B-ORG'], { outside: 'O' })).toThrow()
  })

  it('rejects segments that do not tile the input or are too long', () => {
    const model = semiMarkovCrf(['A'], words, 2)
    const w = new Float64Array(semiMarkovLayout(model).size)
    const x = [{ w0: 1 }, { w1: 1 }, { w2: 1 }]
    expect(() => semiMarkovScore(model, w, x, [{ label: 0, start: 0, end: 2 }])).toThrow()
    expect(() => semiMarkovScore(model, w, x, [{ label: 0, start: 0, end: 3 }])).toThrow()
  })
})

describe('learning a planted segmentation task', () => {
  const labels = ['O', 'E']
  const model = semiMarkovCrf(labels, words, 3, { boundary: true })
  const layout = semiMarkovLayout(model)
  const truth = randomWeights(21, layout.size, 1.5)
  const oracle = semiMarkovOracle(model)
  const label = (x: SegmentInput): { x: SegmentInput; y: Segmentation } => ({
    x,
    y: semiMarkovDecode(model, truth, x).y,
  })
  const train = inputs(31, 40, 4, 8).map(label)
  const test = inputs(41, 30, 4, 8).map(label)

  const objective = (w: ArrayLike<number>): { value: number; gradient: Float64Array } => {
    const gradient = new Float64Array(layout.size)
    let value = 0
    for (const { x, y } of train) {
      const e = oracle.expectations(w, x)
      const phi = oracle.features(x, y)
      value += e.logPartition - phi.reduce((s, v, q) => s + v * (w[q] ?? 0), 0)
      for (let q = 0; q < gradient.length; q++) gradient[q] = (gradient[q] ?? 0) + (e.counts[q] ?? 0) - (phi[q] ?? 0)
    }
    for (let q = 0; q < gradient.length; q++) {
      value += 0.01 * (w[q] ?? 0) ** 2
      gradient[q] = (gradient[q] ?? 0) + 0.02 * (w[q] ?? 0)
    }
    return { value, gradient }
  }

  it('the objective gradient matches finite differences', () => {
    const w = randomWeights(2, layout.size, 0.5)
    const g = objective(w).gradient
    for (const q of [
      0,
      3,
      layout.begin + 2,
      layout.end + 1,
      layout.length + 2,
      layout.transition + 1,
      layout.size - 1,
    ]) {
      const h = 1e-6
      const up = Float64Array.from(w)
      const down = Float64Array.from(w)
      up[q] = (up[q] ?? 0) + h
      down[q] = (down[q] ?? 0) - h
      expect((objective(up).value - objective(down).value) / (2 * h)).toBeCloseTo(g[q] ?? Number.NaN, 5)
    }
  })

  it('L-BFGS recovers the segmentations of held-out sequences', () => {
    const r = lbfgs(
      analyticObjective({
        value: (w) => objective(Float64Array.from(w)).value,
        gradient: (w) => objective(Float64Array.from(w)).gradient,
      }),
      new Float64Array(layout.size),
      { gradTol: 1e-6, maxIterations: 300 },
    )
    const w = Float64Array.from(toFlat(r.x as Value))
    const exact = (data: typeof test): number =>
      data.filter(({ x, y }) => oracle.loss(y, semiMarkovDecode(model, w, x).y) === 0).length / data.length
    expect(exact(train)).toBeGreaterThan(0.95)
    expect(exact(test)).toBeGreaterThan(0.8)
  })
})
