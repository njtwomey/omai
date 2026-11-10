import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { adam } from '../../optim/index.ts'
import { key, uniform } from '../../random/index.ts'
import { toArray, type Value } from '../../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../../trace/index.ts'
import type { NeuralCrfAllowed } from './neural-crf.ts'
import {
  evaluateTagger,
  fitTagger,
  padTagging,
  SequenceTagger,
  type TaggedSequence,
  taggerAlgorithm,
} from './sequence-tagger.ts'

// Tags O, B-X, I-X cycle O, B, I, O, B, I, ... from a random start (O or B), so entities are B I pairs and the tag
// sequence has exactly the structure a CRF's transition scores can hold (I only after B). Each token is the tag's
// signature (O: 1 or 2, B: 3, I: 4) with probability 0.7 and otherwise one of the other three signatures: a token
// alone is a poor witness, a sentence's worth of them fixes the phase of the cycle.
const TAGS = ['O', 'B-X', 'I-X']
const SIGNATURE = [[1, 2], [3], [4]]
const NOISE = 0.3

function planted(seed: number, count: number): TaggedSequence[] {
  const u = uniform(key(seed), [count * 40]).data
  let at = 0
  const next = (): number => u[at++] ?? 0
  return Array.from({ length: count }, (): TaggedSequence => {
    const length = 6 + Math.floor(next() * 5)
    const phase = Math.floor(next() * 2)
    const tags = Array.from({ length }, (_, t) => (phase + t) % 3)
    const tokens = tags.map((tag) => {
      const own = SIGNATURE[tag] as number[]
      if (next() >= NOISE) return own[Math.floor(next() * own.length)] ?? 1
      const others = [1, 2, 3, 4].filter((v) => !own.includes(v))
      return others[Math.floor(next() * others.length)] ?? 1
    })
    return { tokens, tags }
  })
}

/** The BIO scheme: I-X may only follow B-X or I-X, and may not start a sequence. */
const allowedBio: NeuralCrfAllowed = {
  transitions: TAGS.map((a) => TAGS.map((b) => !(b.startsWith('I-') && !(a === 'B-X' || a === 'I-X')))),
  start: TAGS.map((t) => !t.startsWith('I-')),
}

const options = { vocabSize: 5, numTags: 3, embeddingDim: 4, hiddenSize: 4, paddingIdx: 0 }

describe('padTagging', () => {
  it('pads tokens and tags and masks the padding', () => {
    const b = padTagging(
      [
        { tokens: [3, 4, 5], tags: [0, 1, 1] },
        { tokens: [6], tags: [2] },
      ],
      9,
    )
    expect(b.tokens).toEqual([
      [3, 4, 5],
      [6, 9, 9],
    ])
    expect(b.tags).toEqual([
      [0, 1, 1],
      [2, 0, 0],
    ])
    expect(b.mask).toEqual([
      [1, 1, 1],
      [1, 0, 0],
    ])
    expect(() => padTagging([])).toThrow()
    expect(() => padTagging([{ tokens: [1], tags: [1, 2] }])).toThrow(/same length/)
  })
})

describe('SequenceTagger', () => {
  const tagger = new SequenceTagger({ ...options, key: key(3) })
  const sequences = planted(1, 3)
  const batch = padTagging(sequences)

  it('has the encoder, the classifier and the CRF as parameters', () => {
    const names = Object.keys(tagger.stateDict())
    expect(names).toContain('embedding.weight')
    expect(names).toContain('lstm.weight_ih_l0_reverse')
    expect(names).toContain('classifier.weight')
    expect(names).toContain('crf.transitions')
    const softmax = new SequenceTagger({ ...options, crf: false })
    expect(Object.keys(softmax.stateDict()).some((n) => n.startsWith('crf'))).toBe(false)
  })

  it('emissions are [B, T, K] and padding does not change the scores of the real tokens', () => {
    const e = tagger.apply(tagger.params, batch, { train: false })
    expect(e.shape).toEqual([3, batch.tokens[0]?.length ?? 0, 3])
    const shortest = sequences.reduce((a, s) => (s.tokens.length < a.tokens.length ? s : a))
    const index = sequences.indexOf(shortest)
    const alone = toArray(tagger.apply(tagger.params, padTagging([{ tokens: shortest.tokens }]), { train: false }))
    const padded = (toArray(e) as number[][][])[index] ?? []
    for (const [t, row] of ((alone as number[][][])[0] ?? []).entries())
      for (const [k, v] of row.entries()) expect(padded[t]?.[k]).toBeCloseTo(v, 12)
  })

  it('the loss is differentiable in the classifier and in the CRF transitions (finite differences)', () => {
    const small = new SequenceTagger({ vocabSize: 5, numTags: 3, embeddingDim: 3, hiddenSize: 3, key: key(5) })
    const b = padTagging(planted(2, 2).map((s) => ({ tokens: s.tokens.slice(0, 3), tags: s.tags.slice(0, 3) })))
    const head = small.params['classifier'] as Record<string, Value>
    const forClassifier = checkPrimitive(
      (weight: Value) => small.loss(b, { ...small.params, classifier: { ...head, weight } }),
      [head['weight'] as Value],
      { vmap: false, rtol: 1e-6, atol: 1e-7 },
    )
    expect(forClassifier.failures).toEqual([])
    const crf = small.params['crf'] as Record<string, Value>
    const forCrf = checkPrimitive(
      (transitions: Value) => small.loss(b, { ...small.params, crf: { ...crf, transitions } }),
      [crf['transitions'] as Value],
      { vmap: false, rtol: 1e-6, atol: 1e-7 },
    )
    expect(forCrf.failures).toEqual([])
  })

  it('predicts as many tags as tokens, and respects allowed transitions', () => {
    const paths = tagger.predict(batch, { allowed: allowedBio })
    expect(paths.map((p) => p.length)).toEqual(batch.mask.map((r) => r.reduce((a, v) => a + v, 0)))
    for (const p of paths) {
      expect(p[0]).not.toBe(2)
      for (let t = 1; t < p.length; t++) if (p[t] === 2) expect(p[t - 1]).not.toBe(0)
    }
  })

  it('refuses a loss without tags', () => {
    expect(() => tagger.loss(padTagging([{ tokens: [1, 2] }]))).toThrow(/no tags/)
  })
})

describe('training a BiLSTM-CRF tagger end to end', () => {
  const train = planted(11, 100)
  const test = planted(12, 200)
  const fit = (crf: boolean, seed: number): SequenceTagger => {
    const tagger = new SequenceTagger({ ...options, crf, key: key(seed) })
    fitTagger(tagger, train, { optimizer: adam({ learningRate: 0.05 }), epochs: 20, batchSize: 10, key: key(1) })
    return tagger
  }

  it('learns the planted task, and learns the BIO structure into the transition scores', () => {
    const tagger = fit(true, 7)
    // Unconstrained Viterbi already never breaks the scheme, and the constrained decoding agrees with it.
    const free = evaluateTagger(tagger, test, { tagNames: TAGS })
    const constrained = evaluateTagger(tagger, test, { tagNames: TAGS, allowed: allowedBio })
    expect(free.tokenAccuracy).toBeGreaterThan(0.9)
    expect(free.entityF1).toBeGreaterThan(0.9)
    expect(free.violations).toBe(0)
    expect(constrained.tokenAccuracy).toBeGreaterThan(0.9)
    expect(constrained.violations).toBe(0)
    // O to I (and O to O, which the cycle never does) are scored far below O to B; B to I far above B to B.
    const trans = toArray((tagger.params['crf'] as Record<string, Value>)['transitions'] as Value) as number[][]
    const at = (i: number, j: number): number => trans[i]?.[j] ?? 0
    expect(at(0, 2)).toBeLessThan(0)
    expect(at(0, 2)).toBeLessThan(at(0, 1) - 2)
    expect(at(1, 2)).toBeGreaterThan(at(1, 1) + 2)
  })

  it('beats the same encoder with a per-token softmax on entity F1, which also breaks the scheme', () => {
    for (const seed of [7, 9]) {
      const withCrf = evaluateTagger(fit(true, seed), test, { tagNames: TAGS, allowed: allowedBio })
      const withSoftmax = evaluateTagger(fit(false, seed), test, { tagNames: TAGS })
      expect(withCrf.entityF1 ?? 0).toBeGreaterThan(withSoftmax.entityF1 ?? 1)
      expect(withCrf.violations).toBe(0)
      expect(withSoftmax.violations ?? 0).toBeGreaterThan(0)
    }
  })

  it('is a traced algorithm: loss, gradient norm and per-epoch scores on the tape, a serialisable state', () => {
    const tagger = new SequenceTagger({ ...options, key: key(7) })
    const alg = taggerAlgorithm(tagger, train.slice(0, 24), {
      optimizer: adam({ learningRate: 0.05 }),
      epochs: 3,
      batchSize: 12,
      tagNames: TAGS,
    })
    const rec = recorder()
    const run = runAlgorithm(alg, { maxSteps: 100, tracer: rec })
    expect(run.status).toBe('completed')
    expect(run.state.step).toBe(6)
    expect(metricSeries(rec.run(), 'loss').values).toHaveLength(6)
    expect(metricSeries(rec.run(), 'gradNorm').values).toHaveLength(6)
    expect(metricSeries(rec.run(), 'tokenAccuracy').steps).toEqual([2, 4, 6])
    expect(metricSeries(rec.run(), 'entityF1').values).toHaveLength(3)
    const epochLoss = metricSeries(rec.run(), 'epochLoss').values
    expect(epochLoss[2] ?? 0).toBeLessThan(epochLoss[0] ?? 0)
    // A stored state resumes to the same place.
    const half = runAlgorithm(alg, { maxSteps: 3 })
    const json = alg.serialise?.(half.state)
    expect(json).toBeDefined()
    const resumed = runAlgorithm(alg, { maxSteps: 100, start: { state: json, step: 3, serialised: true } })
    expect(resumed.state.step).toBe(6)
    expect(resumed.state.loss).toBeCloseTo(run.state.loss, 10)
  })
})
