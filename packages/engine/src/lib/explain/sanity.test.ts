import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { matmul, type Tensor, tanh, toFlat, type Value } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  dataRandomisationTest,
  inputXGradient,
  modelRandomisationTest,
  randomisationTestAlgorithm,
  saliency,
} from './index.ts'

// A three-layer tanh network on 6 inputs; the explained input is fixed.
const x = [[0.9, -0.4, 0.2, 0.7, -0.8, 0.3]]
const trained: Tensor[] = [
  normal(key(1), [6, 8]) as Tensor,
  normal(key(2), [8, 8]) as Tensor,
  normal(key(3), [8, 1]) as Tensor,
]
const network =
  (params: readonly Tensor[]) =>
  (inp: Value): Value =>
    matmul(tanh(matmul(tanh(matmul(inp, params[0] as Tensor)), params[1] as Tensor)), params[2] as Tensor)

const gradientExplainer = (params: readonly Tensor[]): number[] => Array.from(toFlat(saliency(network(params), x)))
const productExplainer = (params: readonly Tensor[]): number[] => Array.from(toFlat(inputXGradient(network(params), x)))
// An "explainer" that only looks at the input: it must fail the check.
const inputOnly = (): number[] => (x[0] as number[]).map(Math.abs)

describe('model randomisation test', () => {
  it('a gradient explainer changes as layers are randomised; an input-only one does not', () => {
    const good = modelRandomisationTest(gradientExplainer, trained, { key: key(7) })
    expect(good.layers).toEqual([2, 1, 0])
    expect(good.similarity[2] as number).toBeLessThan(0.8)
    const bad = modelRandomisationTest(inputOnly, trained, { key: key(7) })
    expect(bad.similarity).toEqual([1, 1, 1])
    // Input times gradient also depends on the model.
    expect(modelRandomisationTest(productExplainer, trained, { key: key(7) }).similarity[2] as number).toBeLessThan(0.9)
  })
  it('the independent order randomises each layer with the others trained', () => {
    const r = modelRandomisationTest(gradientExplainer, trained, { key: key(7), order: 'independent' })
    expect(r.layers).toEqual([2, 1, 0])
    // Randomising only the first layer leaves the later ones: a different map from the cascade's last step.
    const cascade = modelRandomisationTest(gradientExplainer, trained, { key: key(7) })
    expect(r.similarity[2]).not.toBe(cascade.similarity[2])
  })
  it('is an algorithm with a serialisable state, one layer per step', () => {
    const alg = randomisationTestAlgorithm(gradientExplainer, trained, { key: key(7) })
    const r = runAlgorithm(alg, { maxSteps: 2 })
    expect(r.state.randomised).toEqual([2, 1])
    const back = alg.deserialise?.(alg.serialise?.(r.state) as never)
    expect(back?.similarity).toEqual(r.state.similarity)
    expect(runAlgorithm(alg, { maxSteps: 10 }).status).toBe('completed')
  })
})

describe('data randomisation test', () => {
  it('a learned relation is destroyed by permuted labels', () => {
    // "Training" is a least-squares slope per feature on a design where feature j predicts the label with weight j.
    const design = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(2 * i), Math.sin(3 * i + 1)])
    const labels = design.map((r) => 3 * (r[0] as number) + 1 * (r[1] as number) + 0.2 * (r[2] as number))
    const fit = (y: readonly number[]): number[] =>
      [0, 1, 2].map(
        (j) =>
          design.reduce((a, r, i) => a + (r[j] as number) * (y[i] as number), 0) /
          design.reduce((a, r) => a + (r[j] as number) ** 2, 0),
      )
    const sim = dataRandomisationTest(fit, labels, { key: key(5), metric: 'pearson' })
    expect(sim).toBeLessThan(0.5)
    expect(dataRandomisationTest(() => [1, 2, 3], labels, { key: key(5) })).toBe(1)
  })
})
