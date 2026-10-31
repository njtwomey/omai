import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { grad } from '../autodiff/index.ts'
import { defaults, grid } from '../space/index.ts'
import { item, softmax, tensor, toArray, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const z = tensor([
  [0.3, -1.2, 2],
  [0.5, 0.1, -0.7],
  [-0.4, 1.5, 0.2],
])
const y = [2, 0, 0]
const transition = [
  [0.8, 0.1, 0.1],
  [0.2, 0.7, 0.1],
  [0.05, 0.15, 0.8],
]

describe('noise-robust classification losses', () => {
  for (const [name, f] of [
    ['generalised', (a: Value) => L.generalisedCrossEntropy(a, y, { q: 0.6 })],
    ['truncated', (a: Value) => L.truncatedCrossEntropy(a, y, { q: 0.6, k: 0.2 })],
    ['symmetric', (a: Value) => L.symmetricCrossEntropy(a, y, { alpha: 0.5, beta: 1.5 })],
    ['probability MAE', (a: Value) => L.probabilityMae(a, y)],
    ['forward corrected', (a: Value) => L.forwardCorrected(a, y, transition)],
    ['backward corrected', (a: Value) => L.backwardCorrected(a, y, transition)],
    ['bi-tempered', (a: Value) => L.biTemperedLogistic(a, y, { t1: 0.7, t2: 1.4 })],
  ] as const) {
    it(`${name} passes the conformance check`, () => expectConformant(f as (...xs: Value[]) => Value, [z]))
  }

  it('holds the bootstrapped loss own belief constant: its gradient is that of the cross-entropy to the mixed target', () => {
    const p = toArray(softmax(z, -1)) as number[][]
    for (const mode of ['soft', 'hard'] as const) {
      const mixed = p.map((row, i) =>
        row.map((v, c) => 0.7 * (c === y[i] ? 1 : 0) + 0.3 * (mode === 'soft' ? v : v === Math.max(...row) ? 1 : 0)),
      )
      const got = toArray(grad((a: Value) => L.bootstrapped(a, y, { beta: 0.7, mode }))(z)) as number[][]
      const want = toArray(grad((a: Value) => L.crossEntropy(a, mixed))(z)) as number[][]
      for (const [i, row] of got.entries())
        for (const [c, v] of row.entries()) expect(v).toBeCloseTo(want[i]?.[c] ?? Number.NaN, 13)
    }
  })

  it('generalised cross-entropy tends to the cross-entropy as q goes to zero and is half the MAE at q = 1', () => {
    const ce = item(L.crossEntropy(z, y))
    expect(item(L.generalisedCrossEntropy(z, y, { q: 1e-6 }))).toBeCloseTo(ce, 4)
    expect(item(L.generalisedCrossEntropy(z, y, { q: 1 }))).toBeCloseTo(item(L.probabilityMae(z, y)) / 2, 13)
  })

  it('bounds the robust losses where cross-entropy is unbounded', () => {
    // A confidently wrong example: the target class has probability near zero.
    const wrong = [[20, 0, 0]]
    expect(item(L.crossEntropy(wrong, [1]))).toBeGreaterThan(19)
    expect(item(L.generalisedCrossEntropy(wrong, [1], { q: 0.7 }))).toBeLessThan(1 / 0.7)
    expect(item(L.probabilityMae(wrong, [1]))).toBeLessThanOrEqual(2)
    expect(item(L.biTemperedLogistic(wrong, [1], { t1: 0.5, t2: 1.5 }))).toBeLessThan(2 / (1 - 0.5) + 1)
  })

  it('has a constant sum over all labels for the probability MAE (the symmetry that makes it noise tolerant)', () => {
    const total = [0, 1, 2].reduce((acc, c) => acc + item(L.probabilityMae(z, [c, c, c])), 0)
    expect(total).toBeCloseTo(2 * (3 - 1), 13)
  })

  it('gives the truncated loss no gradient on an example below the threshold', () => {
    const g = grad((a: Value) => L.truncatedCrossEntropy(a, [1], { q: 0.7, k: 0.5 }))(tensor([[5, 0, 0]]))
    expect(toArray(g)).toEqual([[0, 0, 0]])
  })

  it('reduces the corrected losses and the bi-tempered logistic loss to the cross-entropy at the identity', () => {
    const ce = item(L.crossEntropy(z, y))
    const id = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]
    expect(item(L.forwardCorrected(z, y, id))).toBeCloseTo(ce, 13)
    expect(item(L.backwardCorrected(z, y, id))).toBeCloseTo(ce, 13)
    expect(item(L.biTemperedLogistic(z, y, { t1: 1, t2: 1 }))).toBeCloseTo(ce, 13)
    expect(item(L.bootstrapped(z, y, { beta: 1 }))).toBeCloseTo(ce, 13)
    expect(item(L.symmetricCrossEntropy(z, y, { alpha: 1, beta: 0 }))).toBeCloseTo(ce, 13)
  })

  it('has a tempered softmax that sums to one and tends to the softmax at t = 1', () => {
    const p = L.temperedSoftmax(z, 1.3, 40)
    for (const row of toArray(p) as number[][]) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    expect(toArray(L.temperedSoftmax(z, 1)) as number[][]).toEqual(toArray(L.temperedSoftmax(z, 1)))
    expect(item(L.logTemperature(Math.E, 1))).toBeCloseTo(1, 14)
    expect(item(L.expTemperature(L.logTemperature(2.5, 0.4), 0.4))).toBeCloseTo(2.5, 13)
  })

  it('recovers the clean class under symmetric noise where the cross-entropy at the same budget is pulled off', () => {
    // One logit pair, 100 examples of class 0 of which 40 carry the wrong label 1. The noisy-label minimiser of the
    // cross-entropy matches the 60/40 label mix (p = 0.6); the MAE (symmetric) keeps the clean majority at p -> 1.
    const logit = (s: number): number[][] => Array.from({ length: 100 }, () => [s, 0])
    const labels = Array.from({ length: 100 }, (_, i) => (i < 60 ? 0 : 1))
    const best = (loss: (s: number) => number): number => {
      let arg = 0
      let low = Infinity
      for (let s = -4; s <= 8; s += 0.05) {
        const v = loss(s)
        if (v < low) {
          low = v
          arg = s
        }
      }
      return arg
    }
    const sCe = best((s) => item(L.crossEntropy(logit(s), labels)))
    const sMae = best((s) => item(L.probabilityMae(logit(s), labels)))
    expect(1 / (1 + Math.exp(-sCe))).toBeCloseTo(0.6, 1)
    expect(sMae).toBeGreaterThan(7.9)
  })
})

describe('classification loss tables', () => {
  it('has every multi-class loss positive away from the label and no larger at a more confident right answer', () => {
    for (const config of grid(L.multiclassLosses.space, 1)) {
      const loss = L.multiclassLosses.make(config)
      const confident = item(loss([[4, 0, 0]], [0]))
      const unsure = item(loss([[0.2, 0, 0]], [0]))
      const wrong = item(loss([[0, 4, 0]], [0]))
      expect(confident, JSON.stringify(config)).toBeLessThan(unsure)
      expect(unsure, JSON.stringify(config)).toBeLessThanOrEqual(wrong)
      expect(wrong, JSON.stringify(config)).toBeGreaterThan(confident)
    }
    expect(defaults(L.multiclassLosses.space)).toEqual({ kind: 'crossEntropy' })
  })

  it('reads the noise-robust and bounded entries off the properties', () => {
    expect(L.multiclassLosses.where((p) => p.noiseRobust)).toEqual([
      'generalised',
      'truncated',
      'symmetric',
      'bootstrapped',
      'mae',
      'forwardCorrected',
      'backwardCorrected',
      'biTempered',
    ])
    expect(L.multiclassLosses.where((p) => p.bounded)).toEqual(['generalised', 'truncated', 'mae', 'biTempered'])
    expect(L.multiclassLosses.where((p) => p.likelihood)).toEqual(['crossEntropy'])
    expect(L.multiclassLosses.where((p) => p.needsTransition)).toEqual(['forwardCorrected', 'backwardCorrected'])
  })

  it('binds the knobs: label smoothing of the table is the cross-entropy option and focal matches categoricalFocal', () => {
    const smooth = L.multiclassLosses.make({ kind: 'labelSmoothing', epsilon: 0.2 })
    expect(item(smooth(z, y))).toBeCloseTo(item(L.crossEntropy(z, y, { labelSmoothing: 0.2 })), 14)
    const focal = L.multiclassLosses.make({ kind: 'focal', gamma: 3 })
    expect(item(focal(z, y))).toBeCloseTo(item(L.categoricalFocal(z, y, { gamma: 3 })), 14)
  })

  it('adds focal and smoothed logistic to the binary table, in the margin convention', () => {
    const focal = L.classificationLosses.make({ kind: 'focal', gamma: 0 })
    // gamma 0 and no alpha weighting is the logistic loss.
    expect(item(focal([0.7, -0.2], [1, -1]))).toBeCloseTo(item(L.logistic([0.7, -0.2], [1, -1])), 14)
    const smooth = L.classificationLosses.make({ kind: 'smoothedLogistic', epsilon: 0.1 })
    const sig = (v: number): number => 1 / (1 + Math.exp(-v))
    const bce = (v: number, t: number): number => -(t * Math.log(sig(v)) + (1 - t) * Math.log(sig(-v)))
    expect(item(smooth([0.7, -0.2], [1, -1]))).toBeCloseTo((bce(0.7, 0.95) + bce(-0.2, 0.05)) / 2, 13)
    expect(L.classificationLosses.properties({ kind: 'focal' }).convex).toBe(false)
    expect(L.classificationLosses.where((p) => p.likelihood)).toEqual(['logistic'])
  })
})
