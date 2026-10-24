import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

// Margins kept away from the hinges, where the losses are not differentiable.
const f = tensor([
  [0.3, -1.7, 2.2],
  [0.55, 0.1, -0.4],
])
const y = tensor([
  [1, 1, -1],
  [-1, 1, 1],
])
const scores = tensor([
  [0.3, -1.2, 2, 0.7],
  [0.5, 0.15, -0.7, 1.9],
])

describe('margin losses', () => {
  for (const [name, fn, at] of [
    ['hinge', (a: Value) => L.hinge(a, y), f],
    ['squared hinge', (a: Value) => L.squaredHinge(a, y), f],
    ['logistic', (a: Value) => L.logistic(a, y), f],
    ['exponential', (a: Value) => L.exponential(a, y), f],
    ['modified Huber', (a: Value) => L.modifiedHuber(a, y), f],
    ['perceptron', (a: Value) => L.perceptron(a, y), f],
    ['Crammer-Singer', (a: Value) => L.crammerSinger(a, [1, 0]), scores],
    ['Weston-Watkins', (a: Value) => L.westonWatkins(a, [1, 0], { p: 2 }), scores],
  ] as const) {
    it(`${name} passes the conformance check`, () => expectConformant(fn, [at]))
  }

  it('makes the hinge losses vanish beyond the margin', () => {
    expect(item(L.hinge([3, -2], [1, -1]))).toBe(0)
    expect(item(L.crammerSinger([[5, 1, 0]], [0]))).toBe(0)
    expect(item(L.westonWatkins([[5, 1, 0]], [0]))).toBe(0)
  })

  it('agrees between Crammer-Singer and Weston-Watkins for two classes, up to the 1/C factor', () => {
    const two = tensor([
      [0.2, 0.9],
      [1.5, -0.3],
    ])
    expect(item(L.westonWatkins(two, [0, 1])) * 2).toBeCloseTo(item(L.crammerSinger(two, [0, 1])), 14)
  })
})
