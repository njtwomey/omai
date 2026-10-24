import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const x = tensor([
  [0.3, -1.2, 2],
  [0.5, 0.1, -0.7],
])
const y = tensor([
  [1, 0, 0.4],
  [-0.2, 0.8, 0],
])

describe('regression losses', () => {
  for (const [name, f] of [
    ['mse', (a: Value) => L.mse(a, y)],
    ['mae', (a: Value) => L.mae(a, y)],
    ['huber', (a: Value) => L.huber(a, y, { delta: 0.5 })],
    ['smoothL1', (a: Value) => L.smoothL1(a, y, { beta: 0.5 })],
    ['logCosh', (a: Value) => L.logCosh(a, y)],
    ['pinball', (a: Value) => L.pinball(a, y, { tau: 0.3 })],
    ['expectile', (a: Value) => L.expectile(a, y, { tau: 0.3 })],
    [
      'poissonNll',
      (a: Value) =>
        L.poissonNll(
          a,
          tensor([
            [1, 0, 2],
            [0, 3, 1],
          ]),
          { full: true },
        ),
    ],
    [
      'gaussianNll',
      (a: Value) =>
        L.gaussianNll(
          a,
          y,
          tensor([
            [1, 2, 0.5],
            [0.3, 1, 2],
          ]),
        ),
    ],
    [
      'tweedie (log input)',
      (a: Value) =>
        L.tweedieDeviance(
          a,
          tensor([
            [1, 0, 2],
            [0.5, 3, 1],
          ]),
          { logInput: true },
        ),
    ],
  ] as const) {
    it(`${name} passes the conformance check`, () => expectConformant(f, [x]))
  }

  it('gives smoothL1 = huber / beta and smoothL1 with beta 0 = mae', () => {
    expect(item(L.smoothL1(x, y, { beta: 0.4 }))).toBeCloseTo(item(L.huber(x, y, { delta: 0.4 })) / 0.4, 14)
    expect(item(L.smoothL1(x, y, { beta: 0 }))).toBeCloseTo(item(L.mae(x, y)), 14)
  })

  it('reduces the expectile loss at tau = 0.5 to half the squared error', () => {
    expect(item(L.expectile(x, y))).toBeCloseTo(item(L.mse(x, y)) / 2, 14)
  })

  it('rejects Tweedie powers in (0, 1)', () => {
    expect(() => L.tweedieDeviance([1], [1], { power: 0.5 })).toThrow()
  })
})
