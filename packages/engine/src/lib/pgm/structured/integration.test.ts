import { describe, expect, it } from 'vitest'
import { hyphenationLabels, hyphenationTables, hyphenationWords } from '../../datasets/index.ts'
import { accuracyScore } from '../../metrics/classification.ts'
import { sequenceAccuracyScore } from '../../metrics/sequence-labelling.ts'
import { key, uniform } from '../../random/index.ts'
import { chainOracle } from './chain-oracle.ts'
import { compileChainData } from './compiled.ts'
import type { ItemFeatures } from './crf.ts'
import { attributeExtractor, crfModelFor, labelledSequences } from './features.ts'
import { type GraphCrf, type GraphInput, type GraphLabelling, graphParameterNames } from './graph-crf.ts'
import { graphOracle } from './graph-oracle.ts'
import { gridCrf, gridInput, treeCrf, treeInput } from './graph-structures.ts'
import type { StructuredExample, StructuredOracle } from './oracle.ts'
import { trainStructured } from './solvers.ts'

/** Integration of the structured pieces: graph oracles and a chain oracle, trained through the one solver table. */

const LABELS = ['a', 'b']
const NOISE = 4

/** A reproducible stream of uniforms. */
function stream(seed: number, n: number): () => number {
  const u = uniform(key(seed), [n]).data
  let i = 0
  return () => u[i++] ?? 0
}

type Example = StructuredExample<GraphInput, GraphLabelling>

/** A random reading of each node: `obs=a` or `obs=b`, and each noise attribute on with probability a half. */
function readings(n: number, next: () => number): ItemFeatures[] {
  return Array.from({ length: n }, () => {
    const item: Record<string, number> = { [next() < 0.5 ? 'obs=a' : 'obs=b']: 1 }
    for (let q = 0; q < NOISE; q++) if (next() < 0.5) item[`noise${q}`] = 1
    return item
  })
}

/**
 * Planted data: the graphs are labelled by the decoding of a model with known weights, a unary pull towards the node's
 * reading and a pairwise pull towards agreeing with the neighbours (the noise attributes carry no weight). The labels
 * are therefore the reading with the isolated disagreements smoothed away, and exactly representable.
 */
function planted(model: GraphCrf, inputs: readonly GraphInput[]): Example[] {
  const names = graphParameterNames(model)
  const truth = names.map((name) => {
    const [kind, ...rest] = name.split('|')
    const label = rest.at(-1)
    if (name.startsWith('node:obs=')) return name.slice(9, 10) === label ? 1 : 0
    return kind?.startsWith('pair:') === true && rest.at(-2) === label ? 0.7 : 0
  })
  const oracle = graphOracle(model)
  return inputs.map((x) => ({ x, y: oracle.decode(truth, x).y }))
}

const attributes = ['obs=a', 'obs=b', ...Array.from({ length: NOISE }, (_, q) => `noise${q}`)]

function trees(seed: number, count: number, size: number): GraphInput[] {
  const next = stream(seed, count * size * (NOISE + 3))
  return Array.from({ length: count }, () =>
    treeInput(readings(size, next), {
      parents: Array.from({ length: size }, (_, i) => (i === 0 ? -1 : Math.floor(next() * i))),
    }),
  )
}

function grids(seed: number, count: number, rows: number, cols: number): GraphInput[] {
  const next = stream(seed, count * rows * cols * (NOISE + 2))
  return Array.from({ length: count }, () => gridInput(readings(rows * cols, next), rows, cols))
}

/** Share of nodes decoded right, at the weights. */
function nodeAccuracy(
  oracle: StructuredOracle<GraphInput, GraphLabelling>,
  w: ArrayLike<number>,
  data: readonly Example[],
): number {
  const truth = data.flatMap((d) => d.y.map(String))
  const predicted = data.flatMap((d) => oracle.decode(w, d.x).y.map(String))
  return accuracyScore(truth, predicted).estimate.value
}

/** The weights of the noise attributes (all zero when the sparsity found them). */
function noiseWeights(names: readonly string[], w: ArrayLike<number>): number[] {
  return names.flatMap((n, q) => (n.includes('noise') ? [w[q] ?? 0] : []))
}

/** The accuracy of reading the observation off as the label, with no model at all. */
function readingAccuracy(data: readonly Example[]): number {
  const truth = data.flatMap((d) => d.y.map(String))
  const read = data.flatMap((d) => d.x.nodes.map((n) => (n['obs=a'] === 1 ? 'a' : 'b')))
  return accuracyScore(truth, read).estimate.value
}

const treeModel = treeCrf(LABELS, attributes)
const gridModel = gridCrf(LABELS, attributes)
const structures = [
  {
    name: 'a tree CRF',
    model: treeModel,
    train: planted(treeModel, trees(1, 30, 10)),
    test: planted(treeModel, trees(2, 30, 10)),
  },
  {
    name: 'a grid CRF',
    model: gridModel,
    train: planted(gridModel, grids(3, 24, 3, 4)),
    test: planted(gridModel, grids(4, 24, 3, 4)),
  },
]

describe('graph CRFs trained through the structured solver table', () => {
  for (const s of structures) {
    const oracle = graphOracle(s.model)
    const names = graphParameterNames(s.model)
    const baseline = readingAccuracy(s.test)

    it(`${s.name}: L-BFGS (L2) beats reading the observations off, on held-out graphs`, () => {
      const fit = trainStructured(
        { oracle, data: s.train, penalty: { c2: 0.5 } },
        { solver: { kind: 'lbfgs' } as never },
        100,
      )
      expect(fit.converged).toBe(true)
      expect(Number.isFinite(fit.objective)).toBe(true)
      // the model uses the neighbours: better than the observation alone, on data the weights have not seen
      expect(nodeAccuracy(oracle, fit.weights, s.test)).toBeGreaterThan(baseline + 0.05)
      expect(nodeAccuracy(oracle, fit.weights, s.train)).toBeGreaterThanOrEqual(0.9)
    })

    it(`${s.name}: OWL-QN (c1 > 0) gives exact zeros on the noise attributes and keeps the accuracy`, () => {
      const dense = trainStructured(
        { oracle, data: s.train, penalty: { c2: 0.5 } },
        { solver: { kind: 'lbfgs' } as never },
        100,
      )
      const fit = trainStructured(
        { oracle, data: s.train, penalty: { c2: 0.01, c1: 6 } },
        { solver: { kind: 'owlqn' } as never },
        60,
      )
      const zeros = Array.from(fit.weights).filter((v) => v === 0).length
      expect(zeros).toBeGreaterThan(0)
      expect(fit.nonzeros).toBe(oracle.size - zeros)
      expect(fit.nonzeros).toBeLessThan(dense.nonzeros)
      // the noise attributes are exactly zero, the observation attributes are not
      expect(noiseWeights(names, fit.weights).every((v) => v === 0)).toBe(true)
      expect(Array.from(fit.weights).some((v, q) => v !== 0 && (names[q] ?? '').includes('obs='))).toBe(true)
      expect(nodeAccuracy(oracle, fit.weights, s.test)).toBeGreaterThan(baseline + 0.03)
    })

    it(`${s.name}: the averaged structured perceptron beats reading the observations off, on held-out graphs`, () => {
      const fit = trainStructured({ oracle, data: s.train }, { solver: { kind: 'averaged' } as never, key: key(5) }, 25)
      expect(nodeAccuracy(oracle, fit.weights, s.train)).toBeGreaterThan(readingAccuracy(s.train))
      expect(nodeAccuracy(oracle, fit.weights, s.test)).toBeGreaterThan(baseline + 0.03)
    })
  }
})

describe('a chain CRF with OWL-QN on hyphenation templates', () => {
  const data = hyphenationWords()
  const words = data.tokens.slice(0, 420)
  const labels = hyphenationLabels(data.tags.slice(0, 420))
  const train = { words: words.slice(0, 280), labels: labels.slice(0, 280) }
  const test = { words: words.slice(280), labels: labels.slice(280) }
  // letters c, classes k: the letter and its neighbours, bigrams, trigrams and the vowel/consonant pattern
  const extractor = attributeExtractor(
    [
      'c[0]',
      'c[-1]',
      'c[1]',
      'c[-1]|c[0]',
      'c[0]|c[1]',
      'c[-2]|c[-1]|c[0]',
      'c[-1]|c[0]|c[1]',
      'c[0]|c[1]|c[2]',
      'k[-2]|k[-1]|k[0]|k[1]|k[2]',
      'k[-1]|k[0]|k[1]',
    ].join('\n'),
    { columns: ['c', 'k'], bias: true },
  )
  const tables = hyphenationTables(train.words)
  const crf = crfModelFor(extractor, ['O', 'B'], tables, { minFrequency: 2 })
  const oracle = chainOracle(crf)
  const trainSequences = labelledSequences(extractor, tables, train.labels)
  const compiled = compileChainData(crf, trainSequences)
  const testSequences = labelledSequences(extractor, hyphenationTables(test.words), test.labels)
  const fit = trainStructured(
    { oracle, data: compiled, penalty: { c2: 0.01, c1: 0.5 } },
    { solver: { kind: 'owlqn' } as never },
    150,
  )
  const predict = (sequences: typeof testSequences): string[][] =>
    sequences.map((s) => oracle.decode(fit.weights, s.x).y.map(String))

  it('is sparse: most of the template features are exactly zero', () => {
    expect(oracle.size).toBeGreaterThan(1000)
    const zeros = Array.from(fit.weights).filter((v) => v === 0).length
    expect(fit.nonzeros).toBe(oracle.size - zeros)
    expect(zeros / oracle.size).toBeGreaterThan(0.5)
  })

  it('hyphenates held-out words better than never hyphenating, with a positive F1 on the hyphens', () => {
    const truth = test.labels
    const predicted = predict(testSequences)
    const token = accuracyScore(truth.flat(), predicted.flat()).estimate.value
    const never = accuracyScore(
      truth.flat(),
      truth.flat().map(() => 'O'),
    ).estimate.value
    expect(token).toBeGreaterThan(never)
    let tp = 0
    let fp = 0
    let fn = 0
    for (const [i, t] of truth.entries())
      for (const [j, label] of t.entries()) {
        const p = predicted[i]?.[j]
        if (label === 'B' && p === 'B') tp++
        else if (p === 'B') fp++
        else if (label === 'B') fn++
      }
    const f1 = (2 * tp) / (2 * tp + fp + fn)
    expect(f1).toBeGreaterThan(0.5)
    // sequence accuracy is stricter than token accuracy
    expect(sequenceAccuracyScore(truth, predicted).estimate.value).toBeLessThan(token)
    // and the model fits the training words better than the held-out ones
    expect(sequenceAccuracyScore(train.labels, predict(trainSequences)).estimate.value).toBeGreaterThanOrEqual(
      sequenceAccuracyScore(truth, predicted).estimate.value,
    )
  })
})
