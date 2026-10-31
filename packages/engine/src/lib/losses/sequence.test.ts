import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { grad } from '../autodiff/index.ts'
import { item, logSoftmax, tensor, toArray, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const logits = tensor([
  [0.3, -1.2, 2, 0.1],
  [0.5, 0.1, -0.7, 0.9],
  [-0.4, 1.5, 0.2, 0.0],
  [1.1, 0.4, 0.2, -0.3],
  [0.2, -0.4, 1.1, 0.6],
])
const logProbs = logSoftmax(logits, -1)

/** log p(target) by enumerating every frame-level path of T frames over C classes. */
function bruteForce(lp: number[][], target: number[], blank = 0): number {
  const [frames, classes] = [lp.length, lp[0]?.length ?? 0]
  const collapse = (path: number[]): number[] => path.filter((c, i) => c !== blank && c !== path[i - 1])
  let total = 0
  const walk = (t: number, path: number[], logp: number): void => {
    if (t === frames) {
      const out = collapse(path)
      if (out.length === target.length && out.every((c, i) => c === target[i])) total += Math.exp(logp)
      return
    }
    for (let c = 0; c < classes; c++) walk(t + 1, [...path, c], logp + (lp[t]?.[c] ?? 0))
  }
  walk(0, [], 0)
  return Math.log(total)
}

describe('ctc', () => {
  const lp = toArray(logProbs) as number[][]

  it('sums the probability of every alignment (brute-force enumeration)', () => {
    for (const target of [[2], [1, 2], [2, 2], [3, 1, 2], []]) {
      const want = -bruteForce(lp, target)
      expect(item(L.ctc(logProbs, target, { reduction: 'sum' })), `[${target}]`).toBeCloseTo(want, 12)
      expect(L.ctcTrellis(logProbs, target).logLikelihood, `trellis [${target}]`).toBeCloseTo(-want, 12)
    }
  })

  it('has the gradient with respect to the log-probabilities equal to minus the label posterior', () => {
    const target = [1, 2, 2]
    const g = toArray(grad((x: Value) => L.ctc(x, target, { reduction: 'sum' }))(logProbs)) as number[][]
    const posterior = L.ctcTrellis(logProbs, target).labelPosterior
    for (const [t, row] of g.entries())
      for (const [c, v] of row.entries()) expect(v).toBeCloseTo(-(posterior[t]?.[c] ?? 0), 12)
    for (const row of posterior) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('passes the conformance check, batched with padding and input lengths', () => {
    const x = tensor(
      Array.from({ length: 5 }, (_, t) => [
        [0.3 * t, -0.2, 1.1, 0.4],
        [-0.5, 0.2 * t, 0.7, -1],
      ]),
    )
    expectConformant(
      (a) =>
        L.ctc(
          logSoftmax(a, -1),
          [
            [1, 2, 0],
            [2, 3, 1],
          ],
          { targetLengths: [2, 3], inputLengths: [5, 4] },
        ),
      [x],
    )
  })

  it('follows torch on reductions: mean divides by target length, and a batch equals its sequences', () => {
    const second = logSoftmax(
      tensor([
        [0.1, 0.4, -0.2, 0.3],
        [0.9, -0.1, 0.2, 0.0],
        [0.3, 0.3, 0.1, -0.5],
        [0.2, 0.8, 0.0, 0.1],
        [0.4, 0.1, 0.5, 0.2],
      ]),
      -1,
    )
    const batch = tensor(
      (toArray(logProbs) as number[][]).map((row, t) => [row, (toArray(second) as number[][])[t] ?? []]),
    )
    const targets = [
      [1, 2],
      [3, 1, 2],
    ]
    const [first, other] = [
      item(L.ctc(logProbs, [1, 2], { reduction: 'sum' })),
      item(L.ctc(second, [3, 1, 2], { reduction: 'sum' })),
    ]
    const each = [first, other]
    const padded = [
      [1, 2, 0],
      [3, 1, 2],
    ]
    const lengths = { targetLengths: targets.map((t) => t.length) }
    expect(toArray(L.ctc(batch, padded, { ...lengths, reduction: 'none' }))).toEqual(
      each.map((v) => expect.closeTo(v, 12)),
    )
    expect(item(L.ctc(batch, padded, { ...lengths, reduction: 'sum' }))).toBeCloseTo(first + other, 12)
    expect(item(L.ctc(batch, padded, lengths))).toBeCloseTo((first / 2 + other / 3) / 2, 12)
  })

  it('gives infinite loss to a target the frames cannot emit, or zero with zeroInfinity', () => {
    // [2, 2] needs three frames (a blank between the repeats): two frames cannot do it.
    const short = logProbs
    const two = tensor((toArray(short) as number[][]).slice(0, 2))
    expect(item(L.ctc(two, [2, 2], { reduction: 'sum' }))).toBe(Number.POSITIVE_INFINITY)
    expect(item(L.ctc(two, [2, 2], { reduction: 'sum', zeroInfinity: true }))).toBe(0)
    expect(item(L.ctc(logProbs, [2, 2], { reduction: 'sum' }))).toBeLessThan(Number.POSITIVE_INFINITY)
  })

  it('uses a blank other than zero', () => {
    const want = -bruteForce(toArray(logProbs) as number[][], [0, 1], 3)
    expect(item(L.ctc(logProbs, [0, 1], { blank: 3, reduction: 'sum' }))).toBeCloseTo(want, 12)
  })
})

describe('ctc trellis, alignment and decoding', () => {
  const peaked = tensor([
    [-0.1, -3, -3],
    [-3, -0.1, -3],
    [-3, -0.1, -3],
    [-0.1, -3, -3],
    [-3, -3, -0.1],
    [-0.1, -3, -3],
  ])

  it('aligns the target along the peaked path, which collapses back to it', () => {
    const { path, logProbability } = L.ctcForcedAlignment(peaked, [1, 2])
    expect(path).toEqual([0, 1, 1, 0, 2, 0])
    expect(logProbability).toBeCloseTo(-0.6, 10) // six frames of -0.1
    expect(L.ctcGreedyDecode(peaked)).toEqual([1, 2])
  })

  it('puts most of the alignment posterior on the peaked path and the posterior sums to one', () => {
    const t = L.ctcTrellis(peaked, [1, 2])
    expect(t.extended).toEqual([0, 1, 0, 2, 0])
    expect(t.labelPosterior[1]?.[1]).toBeGreaterThan(0.9)
    expect(t.labelPosterior[4]?.[2]).toBeGreaterThan(0.9)
    for (const row of t.statePosterior) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    // The forward and backward passes agree on the likelihood at every frame.
    for (let f = 0; f < 6; f++) {
      const total = t.alpha[f]?.reduce((acc, a, s) => acc + Math.exp(a + (t.beta[f]?.[s] ?? 0)), 0) ?? 0
      expect(Math.log(total)).toBeCloseTo(t.logLikelihood, 12)
    }
  })

  it('refuses to align a target the frames cannot emit', () => {
    expect(() => L.ctcForcedAlignment(peaked, [1, 1, 1, 1, 1, 1])).toThrow()
  })
})
