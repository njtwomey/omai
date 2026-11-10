import { describe, expect, it } from 'vitest'
import { bioViolations, entitySpans, type TagScheme } from '../../metrics/sequence-labelling.ts'
import { key, normal } from '../../random/index.ts'
import { crfDecode, crfMarginals, crfScore, type ItemFeatures, type LinearChainCrf, linearChainCrf } from './crf.ts'
import {
  allowedTransitions,
  crfDecodeConstrained,
  crfNBest,
  crfPosteriorDecode,
  crfSample,
  type TaggingScheme,
} from './decoding.ts'

const T_ = true
const F_ = false

describe('allowedTransitions', () => {
  it('BIO and IOB2 (every entity starts with B)', () => {
    const labels = ['O', 'B-A', 'I-A', 'B-B', 'I-B']
    const expected = {
      //          O    B-A  I-A  B-B  I-B
      transitions: [
        [T_, T_, F_, T_, F_], // O
        [T_, T_, T_, T_, F_], // B-A
        [T_, T_, T_, T_, F_], // I-A
        [T_, T_, F_, T_, T_], // B-B
        [T_, T_, F_, T_, T_], // I-B
      ],
      start: [T_, T_, F_, T_, F_],
      end: [T_, T_, T_, T_, T_],
    }
    expect(allowedTransitions(labels, 'BIO')).toEqual(expected)
    expect(allowedTransitions(labels, 'IOB2')).toEqual(expected)
  })

  it('IOB1 (I starts an entity, B only between two entities of one type)', () => {
    const labels = ['O', 'I-A', 'B-A', 'I-B']
    expect(allowedTransitions(labels, 'IOB1')).toEqual({
      transitions: [
        //  O    I-A  B-A  I-B
        [T_, T_, F_, T_], // O
        [T_, T_, T_, T_], // I-A
        [T_, T_, T_, T_], // B-A
        [T_, T_, F_, T_], // I-B
      ],
      start: [T_, T_, F_, T_],
      end: [T_, T_, T_, T_],
    })
  })

  it('BIOES (and BILOU with L and U)', () => {
    const labels = ['O', 'B-A', 'I-A', 'E-A', 'S-A', 'B-B', 'E-B']
    expect(allowedTransitions(labels, 'BIOES')).toEqual({
      transitions: [
        //  O    B-A  I-A  E-A  S-A  B-B  E-B
        [T_, T_, F_, F_, T_, T_, F_], // O
        [F_, F_, T_, T_, F_, F_, F_], // B-A
        [F_, F_, T_, T_, F_, F_, F_], // I-A
        [T_, T_, F_, F_, T_, T_, F_], // E-A
        [T_, T_, F_, F_, T_, T_, F_], // S-A
        [F_, F_, F_, F_, F_, F_, T_], // B-B
        [T_, T_, F_, F_, T_, T_, F_], // E-B
      ],
      start: [T_, T_, F_, F_, T_, T_, F_],
      end: [T_, F_, F_, T_, T_, F_, T_],
    })
    const bilou = allowedTransitions(['O', 'B-A', 'I-A', 'L-A', 'U-A', 'B-B', 'L-B'], 'BILOU')
    expect(bilou).toEqual(allowedTransitions(labels, 'BIOES'))
  })

  it('BMES (no O)', () => {
    const labels = ['B-A', 'M-A', 'E-A', 'S-A', 'B-B', 'E-B']
    expect(allowedTransitions(labels, 'BMES')).toEqual({
      transitions: [
        //  B-A  M-A  E-A  S-A  B-B  E-B
        [F_, T_, T_, F_, F_, F_], // B-A
        [F_, T_, T_, F_, F_, F_], // M-A
        [T_, F_, F_, T_, T_, F_], // E-A
        [T_, F_, F_, T_, T_, F_], // S-A
        [F_, F_, F_, F_, F_, T_], // B-B
        [T_, F_, F_, T_, T_, F_], // E-B
      ],
      start: [T_, F_, F_, T_, T_, F_],
      end: [F_, F_, T_, T_, F_, T_],
    })
    expect(() => allowedTransitions(['O', 'B-A'], 'BMES')).toThrow(/not a BMES label/)
  })

  it('rejects labels outside the scheme', () => {
    expect(() => allowedTransitions(['O', 'E-A'], 'BIO')).toThrow(/not a BIO label/)
    expect(() => allowedTransitions(['O', 'B'], 'BIO')).toThrow(/not a BIO label/)
    expect(() => allowedTransitions(['B-A', 'O-A'], 'BIO')).toThrow(/not a BIO label/)
    expect(() => allowedTransitions(['U-A'], 'BIOES')).toThrow(/not a BIOES label/)
  })

  it('keeps a hyphen inside a type name', () => {
    expect(allowedTransitions(['B-X-1', 'I-X-1', 'I-X'], 'BIO').transitions[0]).toEqual([T_, T_, F_])
  })
})

/** A CRF with Gaussian weights on two attributes. */
function randomCrf(labels: readonly string[], seed: number, scale = 1.5): LinearChainCrf {
  const K = labels.length
  const w = (k: number, n: number): number[] => Array.from(normal(key(seed * 7 + k), [n]).data, (v) => scale * v)
  return linearChainCrf(labels, ['a', 'b'], { state: w(1, 2 * K), transition: w(2, K * K) })
}

const items = (n: number, seed: number): { a?: number; b?: number }[] =>
  Array.from({ length: n }, (_, t) =>
    (t + seed) % 3 === 0 ? { a: 1 } : (t + seed) % 3 === 1 ? { b: 1 } : { a: 1, b: 1 },
  )

/** Every labelling of T positions with its crfScore, best first. */
function allLabellings(crf: LinearChainCrf, x: readonly ItemFeatures[]): { labels: string[]; score: number }[] {
  const out: { labels: string[]; score: number }[] = []
  const walk = (labels: string[]): void => {
    if (labels.length === x.length) {
      out.push({ labels, score: crfScore(crf, x, labels) })
      return
    }
    for (const l of crf.labels) walk([...labels, l])
  }
  walk([])
  return out.sort((a, b) => b.score - a.score)
}

const bioLabels = ['O', 'B-P', 'I-P', 'B-L', 'I-L']

/** Whether a tag sequence is valid in a scheme, judged only by `entitySpans` (the metrics' reading of it). */
function validByMetrics(tags: readonly string[], scheme: TaggingScheme): boolean {
  const kind: TagScheme = scheme === 'BILOU' ? 'bilou' : 'bio'
  const head = (t: string | undefined): string => (t ?? 'O').split('-')[0] ?? 'O'
  const spans = entitySpans(tags, kind)
  const covered = new Set<number>()
  for (const s of spans) for (let i = s.start; i < s.end; i++) covered.add(i)
  // every tagged token lies in an entity, and entities have the scheme's shape
  if (tags.some((t, i) => t !== 'O' && !covered.has(i))) return false
  return spans.every((s, n) => {
    const first = head(tags[s.start])
    const last = head(tags[s.end - 1])
    const inner = tags.slice(s.start + 1, s.end - 1).map(head)
    const types = new Set(tags.slice(s.start, s.end).map((t) => t.slice(t.indexOf('-') + 1)))
    if (types.size !== 1) return false
    if (scheme === 'BIO' || scheme === 'IOB2')
      return first === 'B' && inner.every((h) => h === 'I') && (s.end - s.start === 1 || last === 'I')
    if (scheme === 'IOB1') {
      const prev = spans[n - 1]
      const adjacent = prev !== undefined && prev.end === s.start && prev.type === s.type
      return first === (adjacent ? 'B' : 'I') && tags.slice(s.start + 1, s.end).every((t) => head(t) === 'I')
    }
    const [one, begin, middle, end] =
      scheme === 'BILOU' ? ['U', 'B', 'I', 'L'] : scheme === 'BMES' ? ['S', 'B', 'M', 'E'] : ['S', 'B', 'I', 'E']
    if (s.end - s.start === 1) return first === one
    return first === begin && last === end && inner.every((h) => h === middle)
  })
}

describe('constrained CRF decoding', () => {
  it('never emits a BIO violation, where plain Viterbi does', () => {
    let plainViolations = 0
    for (let seed = 1; seed <= 40; seed++) {
      const crf = randomCrf(bioLabels, seed)
      const x = items(3 + (seed % 5), seed)
      const plain = crfDecode(crf, x)
      if (bioViolations(plain.labels).length > 0) plainViolations++
      const fixed = crfDecodeConstrained(crf, x, 'BIO')
      expect(bioViolations(fixed.labels)).toEqual([])
      expect(validByMetrics(fixed.labels, 'BIO')).toBe(true)
      expect(fixed.score).toBeCloseTo(crfScore(crf, x, fixed.labels), 10)
      expect(fixed.score).toBeLessThanOrEqual(plain.score + 1e-12)
      // a valid plain decode is also the constrained one
      if (bioViolations(plain.labels).length === 0) expect(fixed.labels).toEqual(plain.labels)
    }
    expect(plainViolations).toBeGreaterThan(3)
  })

  const schemes: [TaggingScheme, string[]][] = [
    ['BIO', bioLabels],
    ['IOB2', bioLabels],
    ['IOB1', ['O', 'I-P', 'B-P', 'I-L', 'B-L']],
    ['BIOES', ['O', 'B-P', 'I-P', 'E-P', 'S-P', 'B-L', 'I-L', 'E-L', 'S-L']],
    ['BILOU', ['O', 'B-P', 'I-P', 'L-P', 'U-P', 'B-L', 'I-L', 'L-L', 'U-L']],
    ['BMES', ['B-P', 'M-P', 'E-P', 'S-P', 'B-L', 'M-L', 'E-L', 'S-L']],
  ]
  for (const [scheme, labels] of schemes) {
    it(`${scheme}: decoded, n-best and sampled sequences are valid by the metrics' reading`, () => {
      let plainInvalid = 0
      for (let seed = 1; seed <= 12; seed++) {
        const crf = randomCrf(labels, seed, 2)
        const x = items(4 + (seed % 3), seed)
        if (!validByMetrics(crfDecode(crf, x).labels, scheme)) plainInvalid++
        expect(validByMetrics(crfDecodeConstrained(crf, x, scheme).labels, scheme)).toBe(true)
        for (const c of crfNBest(crf, x, 6, scheme)) expect(validByMetrics(c.labels, scheme)).toBe(true)
        for (const s of crfSample(crf, x, key(seed), 10, scheme)) expect(validByMetrics(s, scheme)).toBe(true)
        expect(validByMetrics(crfPosteriorDecode(crf, x, scheme).labels, scheme)).toBe(true)
      }
      expect(plainInvalid).toBeGreaterThan(0)
    })
  }

  it('matches the best valid labelling found by enumeration', () => {
    for (let seed = 1; seed <= 10; seed++) {
      const crf = randomCrf(bioLabels, seed)
      const x = items(4, seed)
      const best = allLabellings(crf, x).find((l) => bioViolations(l.labels).length === 0)
      expect(crfDecodeConstrained(crf, x, 'BIO').score).toBeCloseTo(best?.score ?? Number.NaN, 10)
    }
  })

  it('takes constraints given directly, and an empty sequence', () => {
    const crf = linearChainCrf(['A', 'B'], ['a'], { state: [1, 0] })
    const x = [{ a: 1 }, { a: 1 }]
    expect(crfDecodeConstrained(crf, x, { end: [false, true] }).labels).toEqual(['A', 'B'])
    expect(crfDecodeConstrained(crf, [], 'BIO')).toEqual({ labels: [], score: 0 })
    expect(() => crfDecodeConstrained(crf, x, { start: [false, false] })).toThrow(/zero probability/)
  })
})

describe('crfNBest', () => {
  it('lists the labellings in the order of enumeration, with their probabilities', () => {
    for (let seed = 1; seed <= 8; seed++) {
      const crf = randomCrf(['A', 'B', 'C'], seed)
      const x = items(4, seed)
      const all = allLabellings(crf, x)
      const logZ = Math.log(all.reduce((s, l) => s + Math.exp(l.score), 0))
      const got = crfNBest(crf, x, 9)
      expect(got).toHaveLength(9)
      for (const [i, c] of got.entries()) {
        expect(c.score).toBeCloseTo(all[i]?.score ?? 0, 10)
        expect(crfScore(crf, x, c.labels)).toBeCloseTo(c.score, 10)
        expect(c.logProbability).toBeCloseTo(c.score - logZ, 10)
      }
      expect(got[0]?.labels).toEqual(crfDecode(crf, x).labels)
    }
  })

  it('all labellings together have probability one', () => {
    const crf = randomCrf(['A', 'B'], 3)
    const x = items(4, 1)
    const all = crfNBest(crf, x, 100)
    expect(all).toHaveLength(16)
    expect(all.reduce((s, c) => s + Math.exp(c.logProbability), 0)).toBeCloseTo(1, 10)
  })

  it('under a scheme it lists only valid labellings, normalised over them, and the first is the constrained decode', () => {
    for (let seed = 1; seed <= 8; seed++) {
      const crf = randomCrf(bioLabels.slice(0, 3), seed)
      const x = items(4, seed)
      const valid = allLabellings(crf, x).filter((l) => bioViolations(l.labels).length === 0)
      const got = crfNBest(crf, x, 100, 'BIO')
      expect(got).toHaveLength(valid.length)
      for (const [i, c] of got.entries()) expect(c.score).toBeCloseTo(valid[i]?.score ?? 0, 10)
      expect(got.reduce((s, c) => s + Math.exp(c.logProbability), 0)).toBeCloseTo(1, 10)
      expect(got[0]?.labels).toEqual(crfDecodeConstrained(crf, x, 'BIO').labels)
    }
  })

  it('handles an empty sequence and a bad n', () => {
    const crf = randomCrf(['A', 'B'], 1)
    expect(crfNBest(crf, [], 3)).toEqual([{ labels: [], score: 0, logProbability: 0 }])
    expect(() => crfNBest(crf, items(2, 0), 0)).toThrow(/positive/)
  })
})

describe('crfPosteriorDecode', () => {
  it('is the per-position argmax of the marginals when unconstrained', () => {
    for (let seed = 1; seed <= 6; seed++) {
      const crf = randomCrf(['A', 'B', 'C'], seed)
      const x = items(5, seed)
      const m = crfMarginals(crf, x)
      const r = crfPosteriorDecode(crf, x)
      expect(r.labels).toEqual(m.map((row) => crf.labels[row.indexOf(Math.max(...row))]))
      expect(r.expectedCorrect).toBeCloseTo(
        m.reduce((s, row) => s + Math.max(...row), 0),
        10,
      )
    }
  })

  it('under BIO has the largest expected accuracy of the valid labellings', () => {
    for (let seed = 1; seed <= 10; seed++) {
      const crf = randomCrf(bioLabels, seed)
      const x = items(4, seed)
      const r = crfPosteriorDecode(crf, x, 'BIO')
      expect(bioViolations(r.labels)).toEqual([])
      const valid = allLabellings(crf, x).filter((l) => bioViolations(l.labels).length === 0)
      const z = valid.reduce((s, l) => s + Math.exp(l.score), 0)
      const expected = (labels: readonly string[]): number =>
        valid.reduce((s, l) => s + (Math.exp(l.score) / z) * labels.filter((y, t) => y === l.labels[t]).length, 0)
      expect(expected(r.labels)).toBeCloseTo(Math.max(...valid.map((l) => expected(l.labels))), 9)
      expect(r.expectedCorrect).toBeCloseTo(expected(r.labels), 9)
    }
  })
})

describe('crfSample', () => {
  const frequencies = (samples: string[][], crf: LinearChainCrf): number[][] => {
    const f = samples[0]?.map(() => new Array<number>(crf.labels.length).fill(0)) ?? []
    for (const s of samples)
      for (const [t, l] of s.entries())
        (f[t] as number[])[crf.labels.indexOf(l)] = ((f[t] as number[])[crf.labels.indexOf(l)] ?? 0) + 1
    return f.map((row) => row.map((c) => c / samples.length))
  }

  const expectFrequencies = (f: number[][], marginals: number[][]): void => {
    for (const [t, row] of marginals.entries())
      for (const [k, p] of row.entries()) expect(Math.abs((f[t]?.[k] ?? 0) - p)).toBeLessThan(0.04)
  }

  it('draws from the posterior: label frequencies match the marginals', () => {
    const crf = randomCrf(['A', 'B', 'C'], 4)
    const x = items(4, 2)
    const f = frequencies(crfSample(crf, x, key(11), 4000), crf)
    expectFrequencies(f, crfMarginals(crf, x))
  })

  it('under a scheme the frequencies match the constrained marginals and no sample is invalid', () => {
    const crf = randomCrf(bioLabels, 6)
    const x = items(4, 1)
    const samples = crfSample(crf, x, key(5), 4000, 'BIO')
    for (const s of samples) expect(bioViolations(s)).toEqual([])
    const f = frequencies(samples, crf)
    expectFrequencies(f, crfPosteriorDecode(crf, x, 'BIO').marginals)
  })

  it('is reproducible for a key and handles an empty sequence', () => {
    const crf = randomCrf(['A', 'B'], 1)
    const x = items(3, 0)
    expect(crfSample(crf, x, key(3), 5)).toEqual(crfSample(crf, x, key(3), 5))
    expect(crfSample(crf, [], key(3), 2)).toEqual([[], []])
  })
})
