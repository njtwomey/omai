import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { item, tensor, toArray, type Value } from '../tensor/index.ts'
import * as I from './index.ts'

const p = tensor([0.1, 0.2, 0.3, 0.4])
const q = tensor([0.25, 0.25, 0.3, 0.2])
const joint = tensor([
  [0.1, 0.2, 0.05],
  [0.15, 0.05, 0.45],
])

describe('discrete entropies and divergences', () => {
  it('entropy matches the definition and handles bases, zeros and batches', () => {
    const h = -[0.1, 0.2, 0.3, 0.4].reduce((s, v) => s + v * Math.log(v), 0)
    expect(item(I.entropy(p))).toBeCloseTo(h, 14)
    expect(item(I.entropy(p, { base: 2 }))).toBeCloseTo(h / Math.LN2, 14)
    expect(item(I.entropy([0, 1, 0]))).toBe(0)
    expect(
      toArray(
        I.entropy(
          [
            [1, 1],
            [1, 0],
          ],
          { base: 2 },
        ),
      ),
    ).toEqual([1, 0])
    expect(
      toArray(
        I.entropy(
          [
            [1, 1],
            [1, 0],
          ],
          { base: 2, axis: 0 },
        ),
      ),
    ).toEqual([1, 0])
    expect(item(I.entropy([2, 2], { normalise: false }))).toBeCloseTo(-4 * Math.log(2), 14)
    expect(() => I.entropy(p, { base: 1 })).toThrow()
  })

  it('cross-entropy is entropy plus KL', () => {
    expect(item(I.crossEntropy(p, q))).toBeCloseTo(item(I.entropy(p)) + item(I.relativeEntropy(p, q)), 14)
    expect(item(I.crossEntropy([1, 0], [0, 1]))).toBe(Number.POSITIVE_INFINITY)
  })

  it('KL is zero on equal inputs and infinite off the support', () => {
    expect(item(I.relativeEntropy(p, p))).toBe(0)
    expect(item(I.relativeEntropy([0.5, 0.5], [1, 0]))).toBe(Number.POSITIVE_INFINITY)
    expect(item(I.relativeEntropy([1, 0], [0.5, 0.5]))).toBeCloseTo(Math.LN2, 14)
  })

  it('Jensen-Shannon is symmetric, bounded and weighted', () => {
    expect(item(I.jensenShannonDivergence(p, q))).toBeCloseTo(item(I.jensenShannonDivergence(q, p)), 14)
    expect(item(I.jensenShannonDivergence([1, 0], [0, 1]))).toBeCloseTo(Math.LN2, 14)
    // With weights (1, 0) the mixture is p itself.
    expect(item(I.jensenShannonDivergence(p, q, { weights: [1, 0] }))).toBeCloseTo(0, 14)
    expect(item(I.jensenShannonDistance(p, q)) ** 2).toBeCloseTo(item(I.jensenShannonDivergence(p, q)), 14)
  })

  it('mutual information, conditional entropy and joint entropy are consistent', () => {
    const hx = item(I.entropy([0.35, 0.65]))
    const hy = item(I.entropy([0.25, 0.25, 0.5]))
    const hxy = item(I.jointEntropy(joint))
    expect(item(I.mutualInformation(joint))).toBeCloseTo(hx + hy - hxy, 13)
    expect(item(I.conditionalEntropy(joint, 1))).toBeCloseTo(hxy - hy, 13)
    expect(item(I.conditionalEntropy(joint, 0))).toBeCloseTo(hxy - hx, 13)
    expect(item(I.conditionalEntropy(joint, [0, 1]))).toBeCloseTo(0, 14)
    expect(item(I.interactionInformation(joint))).toBeCloseTo(item(I.mutualInformation(joint)), 13)
    expect(() => I.mutualInformation([1, 2])).toThrow()
    expect(() => I.conditionalEntropy(joint, 3)).toThrow()
    expect(() => I.interactionInformation([1, 2])).toThrow()
  })

  it('conditional mutual information of a Markov chain X -> Y -> Z given Y is zero', () => {
    // p(x, y, z) = p(x) p(y | x) p(z | y), arranged as [x, z, y] so that I(X; Z | Y) is the conditional MI.
    const px = [0.3, 0.7]
    const pyx = [
      [0.8, 0.2],
      [0.1, 0.9],
    ]
    const pzy = [
      [0.6, 0.4],
      [0.3, 0.7],
    ]
    const t = px.map((a, x) => [0, 1].map((z) => [0, 1].map((y) => a * (pyx[x]?.[y] ?? 0) * (pzy[y]?.[z] ?? 0))))
    expect(item(I.conditionalMutualInformation(t))).toBeCloseTo(0, 14)
  })

  it('PMI and NPMI follow their definitions', () => {
    const pmi = toArray(I.pointwiseMutualInformation(joint)) as number[][]
    expect(pmi[0]?.[0]).toBeCloseTo(Math.log(0.1 / (0.35 * 0.25)), 14)
    const npmi = toArray(I.normalisedPointwiseMutualInformation(joint)) as number[][]
    expect(npmi[1]?.[2]).toBeCloseTo(Math.log(0.45 / (0.65 * 0.5)) / -Math.log(0.45), 14)
    expect(
      toArray(
        I.normalisedPointwiseMutualInformation([
          [1, 1],
          [1, 1],
        ]),
      ),
    ).toEqual([
      [0, 0],
      [0, 0],
    ])
  })

  it('information gain is the mutual information of the split table; gain ratio is 0 for no split', () => {
    const counts = [
      [3, 1],
      [1, 5],
    ]
    expect(item(I.informationGain(counts))).toBeCloseTo(item(I.mutualInformation(counts)), 14)
    expect(item(I.gainRatio([[3, 5]]))).toBe(0)
    expect(item(I.binaryEntropy(0.5, { base: 2 }))).toBe(1)
  })

  it('total variation is half the L1 distance', () => {
    expect(item(I.totalVariation(p, q))).toBeCloseTo(0.5 * (0.15 + 0.05 + 0 + 0.2), 14)
  })

  for (const [name, f, inputs] of [
    ['entropy', (a: Value) => I.entropy(a), [p]],
    ['crossEntropy', (a: Value, b: Value) => I.crossEntropy(a, b), [p, q]],
    ['relativeEntropy', (a: Value, b: Value) => I.relativeEntropy(a, b), [p, q]],
    ['jensenShannonDivergence', (a: Value, b: Value) => I.jensenShannonDivergence(a, b), [p, q]],
    ['totalVariation', (a: Value, b: Value) => I.totalVariation(a, b), [p, tensor([0.3, 0.1, 0.25, 0.35])]],
    ['mutualInformation', (a: Value) => I.mutualInformation(a), [joint]],
    ['conditionalEntropy', (a: Value) => I.conditionalEntropy(a), [joint]],
    [
      'interactionInformation',
      (a: Value) => I.interactionInformation(a),
      [
        tensor([
          [
            [0.1, 0.2],
            [0.05, 0.15],
          ],
          [
            [0.2, 0.1],
            [0.12, 0.08],
          ],
        ]),
      ],
    ],
    [
      'conditionalMutualInformation',
      (a: Value) => I.conditionalMutualInformation(a),
      [
        tensor([
          [
            [0.1, 0.2],
            [0.05, 0.15],
          ],
          [
            [0.2, 0.1],
            [0.12, 0.08],
          ],
        ]),
      ],
    ],
    ['pointwiseMutualInformation', (a: Value) => I.pointwiseMutualInformation(a), [joint]],
    ['binaryEntropy', (a: Value) => I.binaryEntropy(a), [tensor([0.2, 0.7])]],
  ] as const) {
    it(`${name} passes the derivative conformance check`, () => {
      expect(checkPrimitive(f as (...xs: Value[]) => Value, inputs, { vmap: false }).failures).toEqual([])
    })
  }
})
