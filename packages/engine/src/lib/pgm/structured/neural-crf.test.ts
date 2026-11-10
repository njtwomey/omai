import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { item, sum, tensor, toArray, toFlat, type Value } from '../../tensor/index.ts'
import { forwardBackward } from '../inference/chain.ts'
import {
  CRF,
  type NeuralCrfParameters,
  neuralCrfDecode,
  neuralCrfLogLikelihood,
  neuralCrfMarginals,
  neuralCrfNegLogLikelihood,
  neuralCrfParameters,
  neuralCrfPartition,
  neuralCrfScore,
} from './neural-crf.ts'

const K = 3
const B = 3
const T = 5
const draw = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const params: NeuralCrfParameters = {
  transitions: draw(1, [K, K]),
  startTransitions: draw(2, [K]),
  endTransitions: draw(3, [K]),
}
const emissions = draw(4, [B, T, K])
const tags = [
  [0, 2, 1, 1, 0],
  [2, 1, 0, 0, 0],
  [1, 0, 0, 0, 0],
]
const mask = [
  [true, true, true, true, true],
  [true, true, true, false, false],
  [true, false, false, false, false],
]
const opts = { batchFirst: true, mask }

/** Every tag path of a length-n sequence with its score, for brute-force checks. */
function paths(n: number): number[][] {
  let all: number[][] = [[]]
  for (let t = 0; t < n; t++) all = all.flatMap((p) => Array.from({ length: K }, (_, k) => [...p, k]))
  return all
}

describe('the CRF layer', () => {
  it('log Z is the log-sum of the scores of every path, per masked length', () => {
    const logZ = toFlat(neuralCrfPartition(emissions, params, opts))
    ;[5, 3, 1].forEach((length, b) => {
      const e = toArray(emissions) as number[][][]
      const scores = paths(length).map((p) => {
        const row = e[b] as number[][]
        let s = toFlat(params.startTransitions)[p[0] ?? 0] ?? 0
        s += toFlat(params.endTransitions)[p[length - 1] ?? 0] ?? 0
        p.forEach((k, t) => {
          s += (row[t] as number[])[k] ?? 0
          if (t > 0) s += toFlat(params.transitions)[(p[t - 1] ?? 0) * K + k] ?? 0
        })
        return s
      })
      const brute = Math.log(scores.reduce((a, s) => a + Math.exp(s), 0))
      expect(logZ[b]).toBeCloseTo(brute, 12)
    })
  })

  it('padding is invisible: values at masked positions change nothing, in either layout', () => {
    const base = toFlat(neuralCrfLogLikelihood(emissions, tags, params, { ...opts, reduction: 'none' }))
    const changed = (toArray(emissions) as number[][][]).map((row, b) =>
      row.map((r, t) => ((mask[b]?.[t] ?? false) ? r : r.map(() => 1e3))),
    )
    const wrongTags = tags.map((r, b) => r.map((k, t) => ((mask[b]?.[t] ?? false) ? k : 2)))
    expect(toFlat(neuralCrfLogLikelihood(changed, wrongTags, params, { ...opts, reduction: 'none' }))).toEqual(base)
    // The same batch time-major.
    const tm = (g: unknown[][]): unknown[][] => (g[0] ?? []).map((_, t) => g.map((r) => r[t]))
    const timeMajor = toFlat(
      neuralCrfLogLikelihood(tm(toArray(emissions) as unknown[][]) as never, tm(tags), params, {
        mask: tm(mask),
        reduction: 'none',
      }),
    )
    for (const [i, v] of timeMajor.entries()) expect(v).toBeCloseTo(base[i] ?? 0, 12)
  })

  it('reductions: sum, mean and token_mean of the per-sequence log-likelihoods', () => {
    const none = toFlat(neuralCrfLogLikelihood(emissions, tags, params, { ...opts, reduction: 'none' }))
    const total = none.reduce((a, v) => a + v, 0)
    expect(item(neuralCrfLogLikelihood(emissions, tags, params, opts))).toBeCloseTo(total, 12)
    expect(item(neuralCrfLogLikelihood(emissions, tags, params, { ...opts, reduction: 'mean' }))).toBeCloseTo(
      total / 3,
      12,
    )
    expect(item(neuralCrfLogLikelihood(emissions, tags, params, { ...opts, reduction: 'token_mean' }))).toBeCloseTo(
      total / 9,
      12,
    )
    expect(item(neuralCrfNegLogLikelihood(emissions, tags, params, opts))).toBeCloseTo(-total / 3, 12)
  })

  it('a probability: exp of the log-likelihoods of every tag path sums to one', () => {
    const e = draw(5, [1, 3, K])
    const all = paths(3)
    const p = all.reduce(
      (a, path) => a + Math.exp(item(neuralCrfLogLikelihood(e, [path], params, { batchFirst: true }))),
      0,
    )
    expect(p).toBeCloseTo(1, 12)
  })

  it('gradients of the loss match finite differences in the emissions and every parameter', () => {
    const report = checkPrimitive(
      (e: Value, trans: Value, start: Value, end: Value) =>
        neuralCrfNegLogLikelihood(
          e,
          tags,
          { transitions: trans, startTransitions: start, endTransitions: end },
          {
            ...opts,
            reduction: 'token_mean',
          },
        ),
      [emissions, params.transitions, params.startTransitions, params.endTransitions],
      { vmap: false, rtol: 1e-6, atol: 1e-7 },
    )
    expect(report.failures).toEqual([])
    expect(report.passed).toBe(true)
  })

  it('marginals are the posteriors of forward-backward on the same chain, zero at padding', () => {
    const marginals = toArray(neuralCrfMarginals(emissions, params, opts)) as number[][][]
    const A = toArray(params.transitions) as number[][]
    const start = toFlat(params.startTransitions)
    const end = toFlat(params.endTransitions)
    ;[5, 3, 1].forEach((length, b) => {
      const rows = (toArray(emissions) as number[][][])[b] as number[][]
      const logEmission = rows
        .slice(0, length)
        .map((r, t) => (t === length - 1 ? r.map((v, k) => v + (end[k] ?? 0)) : r))
      const fb = forwardBackward({ logInitial: start, logTransition: A, logEmission }, { method: 'log' })
      for (const [t, post] of fb.posteriors.entries()) {
        const row = marginals[b]?.[t] ?? []
        for (const [k, v] of post.entries()) expect(row[k]).toBeCloseTo(v, 10)
      }
      for (let t = length; t < T; t++) expect(marginals[b]?.[t]).toEqual([0, 0, 0])
    })
  })

  it('decoding is the highest-scoring path (brute force), with its score; allowed transitions constrain it', () => {
    const decoded = neuralCrfDecode(emissions, params, opts)
    ;[5, 3, 1].forEach((length, b) => {
      const e = (toArray(emissions) as number[][][])[b] as number[][]
      const score = (p: number[]): number => {
        let s =
          (toFlat(params.startTransitions)[p[0] ?? 0] ?? 0) + (toFlat(params.endTransitions)[p[length - 1] ?? 0] ?? 0)
        p.forEach((k, t) => {
          s += (e[t] as number[])[k] ?? 0
          if (t > 0) s += toFlat(params.transitions)[(p[t - 1] ?? 0) * K + k] ?? 0
        })
        return s
      }
      const best = paths(length).reduce((a, p) => (score(p) > score(a) ? p : a))
      expect(decoded.paths[b]).toEqual(best)
      expect(decoded.scores[b]).toBeCloseTo(score(best), 12)
    })
    // Forbid tag 2 after anything, tag 1 as a start and tag 0 as an end.
    const allowed = {
      transitions: [
        [true, true, false],
        [true, true, false],
        [true, true, false],
      ],
      start: [true, false, true],
      end: [false, true, true],
    }
    const constrained = neuralCrfDecode(emissions, params, { ...opts, allowed })
    for (const [b, path] of constrained.paths.entries()) {
      expect(path[0]).not.toBe(1)
      expect(path[path.length - 1]).not.toBe(0)
      expect(path.slice(1)).not.toContain(2)
      expect(allowedScore(b)).toBeLessThanOrEqual(decoded.scores[b] ?? 0)
    }
    function allowedScore(b: number): number {
      return constrained.scores[b] ?? 0
    }
    expect(() =>
      neuralCrfDecode(emissions, params, {
        ...opts,
        allowed: {
          transitions: [
            [false, false, false],
            [false, false, false],
            [false, false, false],
          ],
        },
      }),
    ).toThrow(/zero probability/)
  })

  it('score of the gold tags: emission, transition, start and end terms', () => {
    const p = {
      transitions: tensor([
        [0, 1],
        [2, 3],
      ]),
      startTransitions: tensor([10, 20]),
      endTransitions: tensor([100, 200]),
    }
    expect(
      toFlat(
        neuralCrfScore(
          [
            [
              [1, 2],
              [3, 4],
            ],
          ],
          [[1, 0]],
          p,
          { batchFirst: true },
        ),
      ),
    ).toEqual([127])
  })

  it('rejects a bad mask, a bad tag and a wrong number of tags', () => {
    expect(() =>
      neuralCrfPartition(emissions, params, {
        batchFirst: true,
        mask: [[true, false, true, true, true], ...mask.slice(1)],
      }),
    ).toThrow(/right-padded/)
    expect(() =>
      neuralCrfLogLikelihood(
        emissions,
        tags.map((r) => r.map(() => 7)),
        params,
        opts,
      ),
    ).toThrow(/outside/)
    expect(() => neuralCrfPartition(draw(6, [B, T, 2]), params, opts)).toThrow(/tags/)
  })
})

describe('the CRF module', () => {
  it('has torchcrf parameters, initialised uniformly on (-0.1, 0.1)', () => {
    const crf = new CRF(4, { key: key(9) })
    expect(Object.keys(crf.stateDict())).toEqual(['end_transitions', 'start_transitions', 'transitions'])
    expect(crf.parameterCount()).toBe(4 + 4 + 16)
    for (const v of crf.parameters()) for (const x of toFlat(v as Value)) expect(Math.abs(x)).toBeLessThan(0.1)
  })

  it('forward is the log-likelihood, decode the Viterbi path, and grad reaches the parameters', () => {
    const crf = new CRF(K, { batchFirst: true, key: key(10) })
    crf.loadStateDict({
      transitions: params.transitions,
      start_transitions: params.startTransitions,
      end_transitions: params.endTransitions,
    })
    const input = { emissions, tags, mask, reduction: 'none' as const }
    expect(toFlat(crf.forward(input))).toEqual(
      toFlat(neuralCrfLogLikelihood(emissions, tags, params, { ...opts, reduction: 'none' })),
    )
    expect(crf.decode(emissions, { mask }).paths).toEqual(neuralCrfDecode(emissions, params, opts).paths)
    expect(toFlat(crf.partition(emissions, mask))).toEqual(toFlat(neuralCrfPartition(emissions, params, opts)))
    expect(item(sum(crf.marginals(emissions, mask)))).toBeCloseTo(9, 10)
  })
})

describe('initial parameters', () => {
  it('draws all three uniformly', () => {
    const p = neuralCrfParameters(5, key(2))
    expect(p.transitions.shape).toEqual([5, 5])
    expect(Math.max(...toFlat(p.startTransitions).map(Math.abs))).toBeLessThan(0.1)
  })
})
