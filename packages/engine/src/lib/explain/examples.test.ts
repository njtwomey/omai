import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { add, dot, fromStorage, matmul, square, sub, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  actionViolations,
  growingSpheres,
  pertinentNegative,
  pertinentPositive,
  projectActions,
  prototypeCounterfactual,
  recourseRobustness,
} from './counterfactuals.ts'
import { gradientSimilarity, mmdCritic, nearestExamples, trustScores } from './examples.ts'

const blobs = Array.from({ length: 30 }, (_, i) => {
  const c = i % 3
  return [c * 4 + Math.sin(i) * 0.3, (c === 1 ? 3 : 0) + Math.cos(i) * 0.3]
})

describe('example-based explanations', () => {
  it('MMD-critic covers the clusters, the MMD falling as prototypes are added', () => {
    const r = mmdCritic(blobs, { nPrototypes: 3, nCriticisms: 2, gamma: 0.5 })
    expect(new Set(r.prototypes.map((i) => i % 3)).size).toBe(3)
    for (let t = 1; t < r.mmd.length; t++) expect(r.mmd[t] ?? 0).toBeLessThan((r.mmd[t - 1] ?? 0) + 1e-12)
    for (const c of r.criticisms) expect(r.prototypes).not.toContain(c)
    expect(r.witness.length).toBe(30)
  })

  it('nearest examples are sorted and trust scores flag a disagreeing prediction', () => {
    const near = nearestExamples([4, 3], blobs, { k: 4 })
    expect(near.indices.every((i) => i % 3 === 1)).toBe(true)
    for (let t = 1; t < near.distances.length; t++)
      expect(near.distances[t] ?? 0).toBeGreaterThanOrEqual(near.distances[t - 1] ?? 0)
    const train = { x: blobs, y: blobs.map((_, i) => i % 3) }
    const r = trustScores(
      train,
      [
        [0, 0],
        [0, 0],
      ],
      [0, 2],
      { alpha: 0.1, kFilter: 3 },
    )
    expect(r.scores[0] ?? 0).toBeGreaterThan(1)
    expect(r.scores[1] ?? 2).toBeLessThan(1)
    expect(r.closestOther[1]).toBe(0)
  })

  it('gradient similarity ranks a training copy of the test point first', () => {
    const loss = (w: Value, x: Value, y: number): Value => square(sub(dot(w, x), y))
    const train = {
      x: [
        [1, 0],
        [0, 1],
        [2, 0.1],
      ],
      y: [3, -1, 5],
    }
    const r = gradientSimilarity(loss, [0.5, 0.5], train, { x: [[0, 1]], y: [-1] }, { k: 3 })
    expect(r.top[0]?.[0]).toBe(1)
    expect(r.scores[0]?.[1]).toBeCloseTo(1, 12)
    const dots = gradientSimilarity(loss, [0.5, 0.5], train, { x: [[0, 1]], y: [-1] }, { similarity: 'dot' })
    expect(dots.scores[0]?.[1] ?? 0).toBeGreaterThan(0)
  })
})

describe('contrastive and constrained counterfactuals', () => {
  const linear = (x: Tensor): Tensor => {
    const v = toFlat(x)
    const n = x.shape[0] ?? 0
    return fromStorage(
      Float64Array.from({ length: n }, (_, i) => (v[3 * i] ?? 0) + 2 * (v[3 * i + 1] ?? 0) - (v[3 * i + 2] ?? 0)),
      [n],
      'float64',
    )
  }

  it('projected points always satisfy the constraints', () => {
    const constraints = {
      immutable: [2],
      monotone: { 0: 'increase' as const },
      bounds: [
        [-1, 4],
        [-3, 3],
        [-9, 9],
      ] as [number, number][],
      integer: [1],
    }
    fc.assert(
      fc.property(
        fc.tuple(
          fc.double({ min: -1, max: 4, noNaN: true }),
          fc.integer({ min: -3, max: 3 }),
          fc.double({ min: -9, max: 9, noNaN: true }),
        ),
        fc.array(fc.double({ min: -5, max: 5, noNaN: true }), { minLength: 3, maxLength: 3 }),
        (x, cf) => {
          expect(actionViolations(x, projectActions(x, cf, constraints), constraints)).toEqual([])
        },
      ),
      { numRuns: 50 },
    )
    expect(actionViolations([1, 1, 1], [0, 1.5, 2], constraints).map((v) => v.reason)).toEqual([
      'increase only',
      'not integer',
      'immutable',
    ])
  })

  it('growing spheres finds a sparse, valid counterfactual that keeps immutable features', () => {
    const cf = growingSpheres(linear, [0, 0, 0], { accept: (p) => p > 2, constraints: { immutable: [1] }, key: key(3) })
    expect(cf?.valid).toBe(true)
    expect(cf?.x[1]).toBe(0)
    expect(cf?.x.filter((v) => v !== 0).length ?? 0).toBeLessThanOrEqual(2)
  })

  it('pertinent negatives change the class, positives keep it with less of x', () => {
    const scores = (v: Value): Value =>
      add(
        matmul(v, [
          [1, -1],
          [-1, 1],
          [0.5, -0.5],
        ]),
        [[0, 0.2]],
      )
    const pn = pertinentNegative(scores, [1, 0, 1], { beta: 0.01 })
    expect(pn?.predictedClass).toBe(1)
    const pp = pertinentPositive(scores, [1, 0, 1])
    expect(pp?.predictedClass).toBe(0)
    for (const [j, v] of (pp?.delta ?? []).entries())
      expect(Math.abs(v)).toBeLessThanOrEqual(Math.abs([1, 0, 1][j] ?? 0) + 1e-12)
    const train = [
      [2, 0, 0],
      [3, 0.5, 1],
      [0, 2, 0],
      [0.5, 3, 1],
    ]
    const proto = prototypeCounterfactual(scores, [1, 0, 1], train, [0, 0, 1, 1], { kPrototype: 2 })
    expect(proto?.targetClass).toBe(1)
    expect(proto?.predictedClass).toBe(1)
  })

  it('recourse robustness counts the models that keep the outcome', () => {
    const shift = (c: number) => (x: Tensor) => sub(linear(x), c) as Tensor
    const r = recourseRobustness([shift(0), shift(1), shift(5)], [1, 1, 0], (p) => p > 0)
    expect(r.accepted).toEqual([true, true, false])
  })
})
