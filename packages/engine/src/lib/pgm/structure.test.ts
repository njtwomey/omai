import { describe, expect, it } from 'vitest'
import { isStructure } from '../scene/index.ts'
import * as P from './index.ts'

describe('every model type describes itself as a structure', () => {
  const bn = P.bayesNet([
    { name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] },
    { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.2, 0.8] },
  ])
  const cases: [string, P.PgmModel, (s: ReturnType<typeof P.toStructure>) => void][] = [
    ['bayes net', bn, (s) => expect(s.edges).toEqual([{ from: 'a', to: 'b', directed: true, kind: 'dependency' }])],
    ['dag', P.dag([['x', 'y']]), (s) => expect(s.type).toBe('dag')],
    [
      'markov network',
      P.isingModel({ rows: 2, cols: 2 }),
      (s) => expect(s.edges.every((e) => e.directed === false)).toBe(true),
    ],
    ['factor graph', P.toFactorGraph(bn), (s) => expect(s.nodes.filter((n) => n.role === 'factor')).toHaveLength(2)],
    [
      'linear gaussian',
      P.linearGaussianBayesNet([
        { name: 'x', parents: [], weights: [], intercept: 0, variance: 1 },
        { name: 'y', parents: ['x'], weights: [2], intercept: 0, variance: 1 },
      ]),
      (s) => expect(s.edges[0]?.label).toBe('2'),
    ],
    [
      'gmrf',
      P.gmrf(
        ['a', 'b', 'c'],
        [0, 0, 0],
        [
          [2, -1, 0],
          [-1, 2, -1],
          [0, -1, 2],
        ],
      ),
      (s) => expect(s.edges).toHaveLength(2),
    ],
    [
      'dbn',
      P.dynamicBayesNet(
        [{ name: 'z', states: ['a', 'b'], parents: [], cpt: [0.5, 0.5] }],
        [{ name: 'z', states: ['a', 'b'], parents: [], previous: ['z'], cpt: [0.9, 0.1, 0.2, 0.8] }],
      ),
      (s) => expect(s.edges).toEqual([{ from: 'z_t-1', to: 'z_t', directed: true, kind: 'dependency' }]),
    ],
    [
      'hmm',
      P.hiddenMarkovModel({
        states: ['a', 'b'],
        initial: [0.5, 0.5],
        transition: [
          [0.9, 0.1],
          [0.1, 0.9],
        ],
        emission: { kind: 'categorical', symbols: ['x'], probabilities: [[1], [1]] },
      }),
      (s) => expect(s.groups).toHaveLength(2),
    ],
    [
      'lds',
      P.linearDynamicalSystem({
        transition: [[1]],
        observation: [[1]],
        transitionCovariance: [[1]],
        observationCovariance: [[1]],
        initialMean: [0],
        initialCovariance: [[1]],
      }),
      (s) => expect(s.nodes.some((n) => n.role === 'observed')).toBe(true),
    ],
    ['linear-chain crf', P.linearChainCrf(['B', 'I', 'O'], ['w']), (s) => expect(s.nodes).toHaveLength(6)],
    ['general crf', P.generalCrf([{ name: 'y', states: ['0', '1'] }], ['f']), (s) => expect(s.nodes).toHaveLength(1)],
    [
      'plate model',
      P.plateModel([
        { name: 'skill', states: ['lo', 'hi'], plates: ['student'], parents: [], cpt: [0.5, 0.5] },
        {
          name: 'pass',
          states: ['no', 'yes'],
          plates: ['student', 'exam'],
          parents: ['skill'],
          cpt: [0.6, 0.4, 0.1, 0.9],
        },
      ]),
      (s) => expect(s.groups.map((g) => g.kind)).toEqual(['plate', 'plate']),
    ],
    [
      'influence diagram',
      P.influenceDiagram(
        [{ name: 'w', states: ['dry', 'rain'], parents: [], cpt: [0.7, 0.3] }],
        [{ name: 'd', states: ['a', 'b'], parents: ['w'] }],
        [{ name: 'u', parents: ['w', 'd'], values: [1, 2, 3, 4] }],
      ),
      (s) => expect(s.nodes.map((n) => n.role)).toEqual(['latent', 'decision', 'utility']),
    ],
  ]
  for (const [name, model, check] of cases) {
    it(name, () => {
      const s = P.toStructure(model)
      expect(isStructure(s)).toBe(true)
      const ids = new Set(s.nodes.map((n) => n.id))
      for (const e of s.edges) {
        expect(ids.has(e.from)).toBe(true)
        expect(ids.has(e.to)).toBe(true)
      }
      check(s)
    })
  }
})
