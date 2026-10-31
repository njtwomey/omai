import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import type { SceneObject } from './captions.ts'
import { captionedShapes, inContextRegression, plantedTransactions, taggedCorpus, typoPairs } from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length)

describe('taggedCorpus', () => {
  const { data, truth } = generate(taggedCorpus, {
    key: key(1),
    knobs: { sentences: 3000, tags: 4, vocabulary: 40, ambiguity: 0.25 },
  })
  const initial = (truth.extra ?? {})['initial'] as number[]
  const transition = (truth.extra ?? {})['transition'] as number[][]
  const emission = (truth.extra ?? {})['emission'] as number[][]
  const vocab = (truth.extra ?? {})['vocabulary'] as string[]

  it('has aligned tokens and tags, and counts that estimate the hidden tables', () => {
    expect(data.tokens.length).toBe(3000)
    data.tokens.forEach((s, i) => {
      expect(s.length).toBe(data.tags[i]?.length)
    })
    const counts = Array.from({ length: 4 }, () => new Array<number>(4).fill(0))
    for (const t of data.tags)
      for (let i = 1; i < t.length; i++)
        (counts[t[i - 1] ?? 0] as number[])[t[i] ?? 0] = ((counts[t[i - 1] ?? 0] as number[])[t[i] ?? 0] ?? 0) + 1
    counts.forEach((row, a) => {
      const z = row.reduce((x, y) => x + y, 0)
      row.forEach((c, b) => {
        expect(Math.abs(c / z - (transition[a]?.[b] ?? 0))).toBeLessThan(0.05)
      })
    })
    const starts = data.tags.map((t) => t[0] ?? 0)
    initial.forEach((p, t) => {
      expect(Math.abs(mean(starts.map((s) => (s === t ? 1 : 0))) - p)).toBeLessThan(0.04)
    })
  })

  it('only emits a word from tags that can emit it, and Viterbi under the true tables beats the best per-word tag', () => {
    for (const [i, s] of data.tokens.entries())
      for (const [j, w] of s.entries())
        expect(emission[data.tags[i]?.[j] ?? 0]?.[vocab.indexOf(w)] ?? 0).toBeGreaterThan(0)
    const viterbi = (words: readonly string[]): number[] => {
      const T = 4
      const lg = (p: number): number => Math.log(Math.max(p, 1e-300))
      let score = Array.from(
        { length: T },
        (_, t) => lg(initial[t] ?? 0) + lg(emission[t]?.[vocab.indexOf(words[0] ?? '')] ?? 0),
      )
      const back: number[][] = []
      for (let i = 1; i < words.length; i++) {
        const next = new Array<number>(T).fill(0)
        const bp = new Array<number>(T).fill(0)
        for (let b = 0; b < T; b++) {
          let best = Number.NEGATIVE_INFINITY
          for (let a = 0; a < T; a++) {
            const v = (score[a] ?? 0) + lg(transition[a]?.[b] ?? 0)
            if (v > best) {
              best = v
              bp[b] = a
            }
          }
          next[b] = best + lg(emission[b]?.[vocab.indexOf(words[i] ?? '')] ?? 0)
        }
        back.push(bp)
        score = next
      }
      let state = score.indexOf(Math.max(...score))
      const path = [state]
      for (let i = back.length - 1; i >= 0; i--) {
        state = back[i]?.[state] ?? 0
        path.unshift(state)
      }
      return path
    }
    // The most frequent tag of each word, from the corpus itself (an upper bound on a context-free tagger).
    const byWord = new Map<string, number[]>()
    for (const [i, s] of data.tokens.entries())
      for (const [j, w] of s.entries()) byWord.set(w, [...(byWord.get(w) ?? []), data.tags[i]?.[j] ?? 0])
    const mode = (xs: readonly number[]): number =>
      [0, 1, 2, 3].reduce((b, t) => (xs.filter((x) => x === t).length > xs.filter((x) => x === b).length ? t : b), 0)
    let vit = 0
    let base = 0
    let total = 0
    for (let i = 0; i < 400; i++) {
      const path = viterbi(data.tokens[i] ?? [])
      data.tokens[i]?.forEach((w, j) => {
        total++
        if (path[j] === data.tags[i]?.[j]) vit++
        if (mode(byWord.get(w) ?? []) === data.tags[i]?.[j]) base++
      })
    }
    expect(vit / total).toBeGreaterThan(base / total)
    expect(vit / total).toBeGreaterThan(0.8)
  })
})

describe('typoPairs', () => {
  it('has no typos at rate zero, and edits about the stated share of characters otherwise', () => {
    const clean = generate(typoPairs, { key: key(2), knobs: { n: 200, errorRate: 0 } }).data
    expect(clean.source).toEqual(clean.target)
    const noisy = generate(typoPairs, { key: key(2), knobs: { n: 4000, errorRate: 0.1 } })
    const same = mean(noisy.data.source.map((s, i) => (s === noisy.data.target[i] ? 1 : 0)))
    const meanLen = mean(noisy.data.target.map((w) => w.length))
    // A word survives when none of its characters is edited; a transposition of equal letters is invisible, so allow slack.
    expect(Math.abs(same - (1 - 0.1) ** meanLen)).toBeLessThan(0.08)
    expect(noisy.data.source.some((s, i) => s.length !== noisy.data.target[i]?.length)).toBe(true)
  })

  it('draws words with Zipfian frequency: the first word is the most common and the prior sums to one', () => {
    const { data, truth } = generate(typoPairs, { key: key(3), knobs: { n: 5000, errorRate: 0 } })
    const vocab = (truth.extra ?? {})['vocabulary'] as string[]
    const probs = (truth.extra ?? {})['wordProbabilities'] as number[]
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const count = (w: string): number => data.target.filter((t) => t === w).length
    expect(count(vocab[0] ?? '')).toBeGreaterThan(count(vocab[50] ?? ''))
    expect(new Set(vocab).size).toBe(vocab.length)
  })
})

describe('plantedTransactions', () => {
  it('has the expected supports for the planted itemsets and an exact empty-set support of one', () => {
    const { data, truth } = generate(plantedTransactions, {
      key: key(4),
      knobs: { transactions: 20000, patterns: 5, items: 30 },
    })
    const patterns = (truth.extra ?? {})['patterns'] as number[][]
    const expected = (truth.extra ?? {})['expectedSupport'] as (s: readonly number[]) => number
    const count = (truth.extra ?? {})['supportCount'] as (s: readonly number[]) => number
    expect(expected([])).toBeCloseTo(1, 12)
    expect(data.tokens?.length).toBe(20000)
    for (const p of patterns) {
      expect(Math.abs(count(p) / 20000 - expected(p))).toBeLessThan(0.015)
      expect(expected(p)).toBeGreaterThan(0.09)
      // Anti-monotone: a subset is at least as frequent as the itemset.
      expect(expected(p.slice(1))).toBeGreaterThanOrEqual(expected(p) - 1e-12)
    }
    expect(expected([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).toBeLessThan(0.01)
  })

  it('is monotone in the itemset for any itemset', () => {
    const { truth } = generate(plantedTransactions, {
      key: key(5),
      knobs: { transactions: 10, patterns: 6, items: 12 },
    })
    const expected = (truth.extra ?? {})['expectedSupport'] as (s: readonly number[]) => number
    fc.assert(
      fc.property(
        fc.uniqueArray(fc.integer({ min: 0, max: 11 }), { maxLength: 6 }),
        fc.integer({ min: 0, max: 11 }),
        (set, extra) => {
          expect(expected([...set, extra])).toBeLessThanOrEqual(expected(set) + 1e-12)
        },
      ),
      { numRuns: 60 },
    )
  })
})

describe('captionedShapes', () => {
  const { data, truth } = generate(captionedShapes, {
    key: key(6),
    knobs: { n: 60, size: 32, maxObjects: 3, noise: 0 },
  })
  const scenes = (truth.extra ?? {})['scenes'] as SceneObject[][]
  const captions = (truth.extra ?? {})['captions'] as string[]
  const qa = (truth.extra ?? {})['questions'] as { question: string; answer: string }[][]
  const rgb = { red: [1, 0.1, 0.1], green: [0.1, 0.8, 0.2], blue: [0.15, 0.25, 1], yellow: [1, 0.9, 0.1] }

  it('draws each object in its colour at the centre of its cell', () => {
    expect(data.images.shape).toEqual([60, 32, 32, 3])
    scenes.forEach((objects, i) => {
      for (const o of objects) {
        const cx = (o.cell % 2) * 16 + 8
        const cy = Math.floor(o.cell / 2) * 16 + 8
        const px = (c: number): number => data.images.data[((i * 32 + cy) * 32 + cx) * 3 + c] ?? 0
        const want = rgb[o.colour]
        for (let c = 0; c < 3; c++) expect(px(c)).toBeCloseTo(want[c] ?? 0, 9)
      }
    })
  })

  it('has captions that name every object and boxes with the shape classes', () => {
    scenes.forEach((objects, i) => {
      for (const o of objects) expect(captions[i]).toContain(`${o.colour} ${o.shape}`)
      expect(data.boxes?.[i]?.shape[0]).toBe(objects.length)
    })
    expect(data.classes).toEqual(['circle', 'square', 'triangle'])
  })

  it('has answers that follow from the scene', () => {
    scenes.forEach((objects, i) => {
      for (const { question, answer } of qa[i] ?? []) {
        if (question === 'how many shapes are there') expect(answer).toBe(String(objects.length))
        if (question.startsWith('what is at')) {
          const cell = ['top left', 'top right', 'bottom left', 'bottom right'].indexOf(
            question.slice('what is at '.length),
          )
          const o = objects.find((x) => x.cell === cell)
          expect(answer).toBe(o === undefined ? 'nothing' : `${o.colour} ${o.shape}`)
        }
      }
    })
  })
})

describe('inContextRegression', () => {
  it('writes x y pairs and a query as tokens, with the quantised noiseless value as the target', () => {
    const { data, truth } = generate(inContextRegression, {
      key: key(7),
      knobs: { n: 50, context: 6, features: 2, noise: 0, levels: 512 },
    })
    expect(data.inputs.shape).toEqual([50, 6 * 3 + 2])
    const decode = (truth.extra ?? {})['decode'] as (t: number) => number
    const target = (truth.extra ?? {})['noiselessTarget'] as number[]
    const clipped = (v: number): number => Math.max(-4, Math.min(4, v))
    for (let i = 0; i < 50; i++)
      expect(Math.abs(decode(data.targets.data[i] ?? 0) - clipped(target[i] ?? 0))).toBeLessThan(0.02)
  })

  it('least squares from the context recovers the target when there is no noise, and the decoded tokens solve to nearly the same', () => {
    const { data, truth } = generate(inContextRegression, {
      key: key(8),
      knobs: { n: 40, context: 10, features: 1, noise: 0, levels: 2048, range: 6 },
    })
    const ls = (truth.extra ?? {})['lsPrediction'] as number[]
    const target = (truth.extra ?? {})['noiselessTarget'] as number[]
    ls.forEach((v, i) => {
      expect(v).toBeCloseTo(target[i] ?? 0, 6)
    })
    const solve = (truth.extra ?? {})['solve'] as (s: readonly number[]) => number[]
    const decode = (truth.extra ?? {})['decode'] as (t: number) => number
    const width = (truth.extra ?? {})['binWidth'] as number
    let close = 0
    for (let i = 0; i < 40; i++) {
      const row = Array.from(toFlat(data.inputs as Tensor).slice(i * 21, i * 21 + 21))
      const got = decode(solve(row)[0] ?? 0)
      if (Math.abs(got - (target[i] ?? 0)) < 40 * width) close++
    }
    expect(close).toBeGreaterThan(36)
  })

  it('the Bayes (ridge) prediction has a lower squared error than least squares on short noisy contexts', () => {
    const { truth } = generate(inContextRegression, {
      key: key(9),
      knobs: { n: 3000, context: 2, features: 2, noise: 1 },
    })
    const ls = (truth.extra ?? {})['lsPrediction'] as number[]
    const bayes = (truth.extra ?? {})['bayesPrediction'] as number[]
    const target = (truth.extra ?? {})['noiselessTarget'] as number[]
    const mse = (p: number[]): number => mean(p.map((v, i) => (v - (target[i] ?? 0)) ** 2))
    expect(mse(bayes)).toBeLessThan(mse(ls))
  })
})
