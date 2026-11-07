/**
 * Tests for the degenerate inputs of the losses, as tables: gradients that must stay finite where the loss is defined
 * (all-zero embeddings, probabilities of exactly 0 or 1), and hyperparameters outside their domain, which must be
 * rejected with `MdmlError('value')` rather than give a finite wrong number.
 */

import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { item, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import { l2Normalise } from './_vectors.ts'
import * as L from './index.ts'

const zeros = tensor([
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
])
const some = tensor([
  [0.3, -1.2, 2],
  [0.5, 0.15, -0.7],
  [1.1, 0.4, 0.2],
  [-0.6, 0.9, 0.3],
])
const labels = [0, 1, 0, 1]
const centres = tensor([
  [1, 0.2, -0.3],
  [0.1, -1, 0.5],
])

describe('the gradients of the losses at degenerate points', () => {
  // The gradient of a norm at zero is infinite and the clamp that guards it passes a zero: infinity times zero is NaN.
  const degenerate = [
    ['l2Normalise', (x: Value) => sum(l2Normalise(x)), zeros],
    ['infoNce', (x: Value) => L.infoNce(x, some, { temperature: 0.5 }), zeros],
    ['infoNce keys', (x: Value) => L.infoNce(some, x, { temperature: 0.5 }), zeros],
    ['ntXent', (x: Value) => L.ntXent(x, labels, { temperature: 0.5 }), zeros],
    ['supervisedContrastive', (x: Value) => L.supervisedContrastive(x, labels, { temperature: 0.5 }), zeros],
    ['clipLoss', (x: Value) => L.clipLoss(x, some, { logitScale: 3 }), zeros],
    ['arcFace', (x: Value) => L.arcFace(x, centres, labels, { scale: 4, margin: 0.3 }), zeros],
    ['cosFace', (x: Value) => L.cosFace(x, centres, labels, { scale: 4 }), zeros],
    // A padded row among ordinary ones: its own gradient is zero and the others' are untouched.
    [
      'ntXent with one zero row',
      (x: Value) => L.ntXent(x, labels, { temperature: 0.5 }),
      tensor([
        [0, 0, 0],
        [0.5, 0.15, -0.7],
        [1.1, 0.4, 0.2],
        [-0.6, 0.9, 0.3],
      ]),
    ],
    // log(0) is clamped to -100 in the value, so the gradient there is 0, not infinity times 0.
    ['binaryCrossEntropy at 0 and 1', (p: Value) => L.binaryCrossEntropy(p, [0, 1, 1]), tensor([1, 0, 0.5])],
    ['binaryCrossEntropy at 1 and 0', (p: Value) => L.binaryCrossEntropy(p, [1, 0, 0]), tensor([1, 0, 0.5])],
  ] as const

  it('has a finite gradient (and value) at all-zero embeddings and at probabilities of exactly 0 and 1', () => {
    for (const [name, f, x] of degenerate) {
      const g = toFlat(grad(f as (t: Value) => Value)(x))
      expect(g.every(Number.isFinite), `${name}: ${g}`).toBe(true)
      expect(Number.isFinite(item(f(x))), `${name}: the value`).toBe(true)
    }
  })

  it('keeps the values and the ordinary gradients: the guard only acts where the norm vanishes', () => {
    // torch's clamp of each log at -100: three terms, two clamped, one at 0.5 (log 2).
    expect(item(L.binaryCrossEntropy([1, 0, 0.5], [0, 1, 1]))).toBeCloseTo((100 + 100 + Math.LN2) / 3, 12)
    const g = toFlat(grad((p: Value) => L.binaryCrossEntropy(p, [0, 1, 1]))(tensor([1, 0, 0.5])))
    expect(g[0]).toBe(0)
    expect(g[1]).toBe(0)
    expect(g[2]).toBeCloseTo(-2 / 3, 12)
    // All-zero embeddings are mapped to zero, not to NaN, and a zero row leaves the others' gradient at zero too.
    expect(toFlat(l2Normalise(zeros)).every((v) => v === 0)).toBe(true)
  })
})

describe('hyperparameters outside their domain', () => {
  const r = [1, 2, 3]
  const t = [1.5, 2, 2.5]
  const sc = [
    [0.3, -1.2, 2],
    [0.5, 0.15, -0.7],
  ]
  // [name, call, the same call with a valid value]: the first must throw a `value` error naming the loss, the second not.
  const rows: readonly (readonly [string, () => unknown, () => unknown])[] = [
    ['huber delta -1', () => L.huber(r, t, { delta: -1 }), () => L.huber(r, t, { delta: 1 })],
    ['huber delta NaN', () => L.huber(r, t, { delta: Number.NaN }), () => L.huber(r, t, { delta: 1 })],
    ['pinball tau 2', () => L.pinball(r, t, { tau: 2 }), () => L.pinball(r, t, { tau: 0.9 })],
    ['expectile tau -0.1', () => L.expectile(r, t, { tau: -0.1 }), () => L.expectile(r, t, { tau: 0 })],
    ['smoothL1 beta -1', () => L.smoothL1(r, t, { beta: -1 }), () => L.smoothL1(r, t, { beta: 0 })],
    [
      'crossEntropy labelSmoothing 2',
      () => L.crossEntropy(sc, [0, 1], { labelSmoothing: 2 }),
      () => L.crossEntropy(sc, [0, 1], { labelSmoothing: 0.1 }),
    ],
    [
      'crossEntropy negative class weight',
      () => L.crossEntropy(sc, [0, 1], { classWeight: [1, -1, 1] }),
      () => L.crossEntropy(sc, [0, 1], { classWeight: [1, 0, 1] }),
    ],
    [
      'binaryCrossEntropy labelSmoothing -1',
      () => L.binaryCrossEntropy([0.2, 0.7], [0, 1], { labelSmoothing: -1 }),
      () => L.binaryCrossEntropy([0.2, 0.7], [0, 1], { labelSmoothing: 0.2 }),
    ],
    ['focal gamma -1', () => L.focal([0.3], [1], { gamma: -1 }), () => L.focal([0.3], [1], { gamma: 0 })],
    [
      'categoricalFocal gamma -1',
      () => L.categoricalFocal(sc, [0, 1], { gamma: -1 }),
      () => L.categoricalFocal(sc, [0, 1], { gamma: 0 }),
    ],
    ['hinge margin -1', () => L.hinge([0.3], [1], { margin: -1 }), () => L.hinge([0.3], [1], { margin: 0 })],
    [
      'crammerSinger margin -1',
      () => L.crammerSinger(sc, [0, 1], { margin: -1 }),
      () => L.crammerSinger(sc, [0, 1], { margin: 0 }),
    ],
    [
      'infoNce temperature 0',
      () => L.infoNce(some, some, { temperature: 0 }),
      () => L.infoNce(some, some, { temperature: 0.1 }),
    ],
    [
      'infoNce temperature -1',
      () => L.infoNce(some, some, { temperature: -1 }),
      () => L.infoNce(some, some, { temperature: 1 }),
    ],
    [
      'ntXent temperature 0',
      () => L.ntXent(some, labels, { temperature: 0 }),
      () => L.ntXent(some, labels, { temperature: 0.5 }),
    ],
    [
      'supervisedContrastive baseTemperature 0',
      () => L.supervisedContrastive(some, labels, { baseTemperature: 0 }),
      () => L.supervisedContrastive(some, labels, { baseTemperature: 0.07 }),
    ],
    [
      'tripletMargin margin -1',
      () => L.tripletMargin(some, some, some, { margin: -1 }),
      () => L.tripletMargin(some, some, some, { margin: 1 }),
    ],
    [
      'contrastive margin -1',
      () => L.contrastive(some, some, [1, 0, 1, 0], { margin: -1 }),
      () => L.contrastive(some, some, [1, 0, 1, 0], { margin: 1 }),
    ],
    [
      'hingeEmbedding margin -1',
      () => L.hingeEmbedding([0.5], [-1], { margin: -1 }),
      () => L.hingeEmbedding([0.5], [-1], { margin: 1 }),
    ],
    [
      'arcFace scale 0',
      () => L.arcFace(some, centres, labels, { scale: 0 }),
      () => L.arcFace(some, centres, labels, { scale: 4 }),
    ],
    [
      'cosFace margin -1',
      () => L.cosFace(some, centres, labels, { margin: -1 }),
      () => L.cosFace(some, centres, labels, { margin: 0.2 }),
    ],
    ['alignment alpha 0', () => L.alignment(some, some, { alpha: 0 }), () => L.alignment(some, some, { alpha: 2 })],
    ['uniformity t -1', () => L.uniformity(some, { t: -1 }), () => L.uniformity(some, { t: 2 })],
    [
      'distillation temperature 0',
      () => L.distillation(sc, sc, { temperature: 0 }),
      () => L.distillation(sc, sc, { temperature: 2 }),
    ],
    [
      'sinkhorn epsilon 0',
      () => L.sinkhorn(some, some, { epsilon: 0 }),
      () => L.sinkhorn(some, some, { epsilon: 0.5 }),
    ],
    [
      'heteroscedasticGaussianNll beta 2',
      () => L.heteroscedasticGaussianNll([0], [0], [1], { beta: 2 }),
      () => L.heteroscedasticGaussianNll([0], [0], [1], { beta: 0.5 }),
    ],
    [
      'dpo beta -1',
      () =>
        L.dpo(
          { policyChosen: [-5], policyRejected: [-7], referenceChosen: [-5], referenceRejected: [-7] },
          { beta: -1 },
        ),
      () =>
        L.dpo(
          { policyChosen: [-5], policyRejected: [-7], referenceChosen: [-5], referenceRejected: [-7] },
          { beta: 0.1 },
        ),
    ],
  ]

  it('rejects each with a `value` error naming the loss, and accepts a valid one', () => {
    for (const [name, bad, good] of rows) {
      expect(bad, name).toThrow(MdmlError)
      expect(bad, name).toThrow(/^\w+: .* must be /)
      try {
        bad()
      } catch (e) {
        expect((e as MdmlError).code).toBe('value')
      }
      expect(good, name).not.toThrow()
    }
  })

  it('leaves the documented sentinels alone: a negative focal alpha disables it, a smoothL1 beta of 0 is the absolute error', () => {
    expect(() => L.focal([0.3], [1], { alpha: -1 })).not.toThrow()
    expect(item(L.smoothL1([0], [3], { beta: 0 }))).toBe(3)
  })
})
