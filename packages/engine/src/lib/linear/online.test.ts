import { describe, expect, it } from 'vitest'
import { oneVsRest } from '../learn/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { kernelPerceptron, passiveAggressive, passiveAggressiveRegressor, perceptron } from './online.ts'

const separable = {
  x: [
    [0, 0],
    [1, 0.5],
    [0.5, 1],
    [4, 4],
    [5, 4.5],
    [4.5, 5],
  ],
  y: [0, 0, 0, 1, 1, 1],
}

describe('the perceptron family', () => {
  it('learns a separable problem within the Novikoff bound, one example per step', () => {
    const est = perceptron()
    const s = est.session(separable, { key: key(0), logs: 'all' })
    s.advance()
    const mistakes = metricSeries(s.run(), 'solver/mistakes').values
    // The count never falls, and the run stops once a pass is clean.
    expect(mistakes.every((m, i) => i === 0 || m >= (mistakes[i - 1] ?? 0))).toBe(true)
    expect(s.status).toBe('converged')
    const last = mistakes[mistakes.length - 1] ?? 0
    const radius = Math.max(...separable.x.map((r) => Math.hypot(...r, 1)))
    expect(last).toBeLessThanOrEqual((radius / 0.3) ** 2)
    expect(toFlat(s.fittedAt(s.t).predict(separable.x))).toEqual(separable.y)
    // After the first example the weights have moved by it alone.
    expect(s.fittedAt(1).params.mistakes).toBe(1)
    expect(s.fittedAt(1).params.examples).toBe(1)
  })

  it('orders passes by the key when it shuffles, reproducibly', () => {
    const a = perceptron({ shuffle: true, maxEpochs: 2 }).fit(separable, { key: key(1) }).params.coef
    const b = perceptron({ shuffle: true, maxEpochs: 2 }).fit(separable, { key: key(1) }).params.coef
    expect(a).toEqual(b)
  })

  it('keeps the best line on noisy data with the pocket, and a vote of vectors with the voted', () => {
    const x = Array.from({ length: 40 }, (_, i) => [Math.sin(i * 1.3) * 2, Math.cos(i * 2.1) * 2])
    const y = x.map(([a = 0, b = 0], i) => (a + b + (i % 7 === 0 ? -3 : 0) > 0 ? 1 : 0))
    const plain = perceptron({ maxEpochs: 5 }).fit({ x, y })
    const pocket = perceptron({ variant: 'pocket', maxEpochs: 5 }).fit({ x, y })
    const voted = perceptron({ variant: 'voted', maxEpochs: 5 }).fit({ x, y })
    expect(pocket.score({ x, y })).toBeGreaterThanOrEqual(plain.score({ x, y }) - 0.15)
    expect(pocket.score({ x, y })).toBeGreaterThan(0.8)
    expect(voted.params.votes.length).toBeGreaterThan(1)
    expect(voted.params.votes.reduce((s, v) => s + v.count, 0)).toBe(5 * 40)
    expect(
      Math.abs(toFlat(voted.decisionFunction(x)).reduce((a, b) => Math.max(a, Math.abs(b)), 0)),
    ).toBeLessThanOrEqual(1)
  })

  it('refuses more than two classes, which oneVsRest handles', () => {
    const three = {
      x: [
        [0, 0],
        [1, 0.5],
        [6, 0],
        [7, 0.5],
        [0, 6],
        [1, 6.5],
      ],
      y: [0, 0, 1, 1, 2, 2],
    }
    expect(() => perceptron().fit(three)).toThrow(/oneVsRest/)
    const f = oneVsRest(perceptron()).fit(three)
    expect(
      toFlat(
        f.predict([
          [0.5, 0.2],
          [6.5, 0.2],
          [0.5, 6.2],
        ]),
      ),
    ).toEqual([0, 1, 2])
  })
})

describe('the kernel perceptron', () => {
  it('learns XOR, which no line can, and logs its support', () => {
    const x = [
      [0, 0],
      [1, 1],
      [0, 1],
      [1, 0],
    ]
    const y = [0, 0, 1, 1]
    const est = kernelPerceptron({ kernelOptions: { gamma: 2 } })
    const s = est.session({ x, y }, { key: key(0), logs: 'all' })
    s.advance()
    expect(metricSeries(s.run(), 'solver/support').values.at(-1)).toBeGreaterThan(1)
    expect(toFlat(s.fittedAt(s.t).predict(x))).toEqual(y)
    expect(perceptron({ maxEpochs: 50 }).fit({ x, y }).score({ x, y })).toBeLessThan(1)
  })
})

describe('passive-aggressive learners', () => {
  const pair = {
    x: [
      [2, 0],
      [-2, 0],
    ],
    y: [1, 0],
  }

  it('leaves a satisfied margin alone and corrects a violated one exactly (PA)', () => {
    const f = passiveAggressive({ variant: 'pa', maxEpochs: 1, fitIntercept: false }).fit(pair)
    // The first example has loss 1 and |x|^2 = 4, so w = (0.5, 0) puts its margin at exactly 1; the second already
    // has margin 1 and is left alone.
    expect(f.params.coef).toEqual([0.5, 0])
    expect(f.params.mistakes).toBe(1)
  })

  it('caps the step with C in PA-I, and learns a line as a regressor', () => {
    const capped = passiveAggressive({ variant: 'pa1', c: 0.1, maxEpochs: 1, fitIntercept: false }).fit(pair)
    // Each of the two examples moves the weights by C |x| = 0.2, not by the 0.5 and 0.15 that would satisfy its margin.
    expect(capped.params.coef[0]).toBeCloseTo(0.4, 12)
    const r = passiveAggressiveRegressor({ epsilon: 0.01, maxEpochs: 60 }).fit({
      x: [[0], [1], [2], [3]],
      y: [1, 3, 5, 7],
    })
    expect(r.score({ x: [[0], [1], [2], [3]], y: [1, 3, 5, 7] })).toBeGreaterThan(0.99)
  })
})
