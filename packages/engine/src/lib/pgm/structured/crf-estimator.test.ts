import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { hyphenationLabels, hyphenationTables, hyphenationWords } from '../../datasets/index.ts'
import { parseCrfppTemplate, type TokenTable } from '../../text/index.ts'
import { metricSeries } from '../../trace/index.ts'
import * as P from '../index.ts'

// A toy problem with a known sklearn-crfsuite answer (python-crfsuite 0.9.12, epsilon and delta 1e-14).
const X = [
  [
    { a: 1, b: 1 },
    { a: 1, c: 1 },
  ],
  [{ a: 1, b: 1 }, { c: 1 }],
  [{ b: 1 }, { a: 1 }],
]
const Y = [
  ['P', 'Q'],
  ['P', 'Q'],
  ['Q', 'P'],
]
const tight = { c2: 0.1, epsilon: 1e-12, delta: 0 } as const

const toy = {
  default: {
    hyper: {},
    state: { 'a|P': 0.828512, 'a|Q': -0.828512, 'b|P': -0.222674, 'b|Q': 0.222674, 'c|Q': 1.671001 },
    trans: { 'P|Q': 1.505142, 'Q|P': 0.97536 },
    marginals: [
      [0.836114, 0.163886],
      [0.268714, 0.731286],
    ],
  },
  minFreq2: {
    hyper: { minFreq: 2 },
    state: { 'a|P': 1.562424, 'b|P': -0.767456, 'c|Q': 1.542695 },
    trans: { 'P|Q': 1.414318 },
    marginals: [
      [0.84912, 0.15088],
      [0.244884, 0.755116],
    ],
  },
  allPossibleStates: {
    hyper: { allPossibleStates: true },
    state: { 'a|P': 0.86872, 'a|Q': -0.86872, 'b|P': -0.207974, 'b|Q': 0.207974, 'c|P': -1.181443, 'c|Q': 1.181443 },
    trans: { 'P|Q': 1.278443, 'Q|P': 1.034144 },
    marginals: [
      [0.860636, 0.139364],
      [0.195324, 0.804676],
    ],
  },
  allPossibleTransitions: {
    hyper: { allPossibleTransitions: true },
    state: { 'a|P': 0.887871, 'a|Q': -0.887871, 'b|P': -0.097711, 'b|Q': 0.097711, 'c|Q': 1.467843 },
    trans: { 'P|P': -1.15228, 'P|Q': 1.054569, 'Q|P': 0.630713, 'Q|Q': -0.533002 },
    marginals: [
      [0.836053, 0.163947],
      [0.242216, 0.757784],
    ],
  },
} as const

const flat = (t: P.CrfWeightTable): Record<string, number> =>
  Object.fromEntries(Object.entries(t).flatMap(([a, row]) => Object.entries(row).map(([b, w]) => [`${a}|${b}`, w])))

function expectClose(got: Record<string, number>, want: Record<string, number>, tol: number): void {
  expect(Object.keys(got).sort()).toEqual(Object.keys(want).sort())
  for (const [k, w] of Object.entries(want)) expect(Math.abs((got[k] ?? 0) - w)).toBeLessThan(tol)
}

describe('the CRF estimator against sklearn-crfsuite', () => {
  for (const [name, c] of Object.entries(toy)) {
    it(`generates CRFsuite's features and reproduces its weights, marginals and accuracy: ${name}`, () => {
      const fitted = P.crf({ ...tight, ...c.hyper }).fit(P.crfData(X, Y))
      expectClose(flat(fitted.stateFeatures()), c.state, 2e-6)
      expectClose(flat(fitted.transitionFeatures()), c.trans, 2e-6)
      expect(fitted.classes).toEqual(['P', 'Q'])
      const m = fitted.predictMarginalsSingle(X[0] ?? [])
      for (const [t, row] of c.marginals.entries()) {
        expect(Math.abs((m[t]?.['P'] ?? 0) - (row[0] ?? 0))).toBeLessThan(1e-5)
        expect(Math.abs((m[t]?.['Q'] ?? 0) - (row[1] ?? 0))).toBeLessThan(1e-5)
      }
      expect(fitted.predict([[{ a: 1 }, { c: 1 }]])).toEqual([['P', 'Q']])
      expect(fitted.score(X, Y)).toBe(1)
    })
  }

  it('reproduces the sklearn-crfsuite fixture through the estimator, features given as dictionaries', () => {
    for (const c of loadFixture('pgm/structured/crf').cases.filter((k) => k.fn === 'crfsuite')) {
      const Xf = c.params['x'] as P.ItemFeatures[][]
      const Yf = c.params['y'] as string[][]
      const labels = c.params['labels'] as string[]
      const fitted = P.crf({
        c2: c.params['c2'] as number,
        labels,
        allPossibleStates: true,
        allPossibleTransitions: true,
        epsilon: 1e-12,
        delta: 0,
      }).fit(P.crfData(Xf, Yf))
      expectClose(flat(fitted.stateFeatures()), c.params['state'] as Record<string, number>, 2e-6)
      expectClose(flat(fitted.transitionFeatures()), c.params['transition'] as Record<string, number>, 2e-6)
      const marginals = c.params['marginals'] as number[][][]
      const got = fitted.predictMarginals(Xf)
      for (const [s, seq] of marginals.entries())
        for (const [t, row] of seq.entries())
          for (const [k, v] of row.entries())
            expect(Math.abs((got[s]?.[t]?.[labels[k] ?? ''] ?? 0) - v)).toBeLessThan(1e-5)
      const test = c.params['test'] as P.ItemFeatures[][]
      expect(fitted.predictSingle(test[0] ?? [])).toEqual(c.params['prediction'])
    }
  })

  it('reproduces the CRF++ cases through the estimator with the CRF++ extractor, bigram weights as transitions', () => {
    for (const c of loadFixture('pgm/structured/crf').cases.filter((k) => k.fn === 'crfpp')) {
      const tables = c.params['tables'] as TokenTable[]
      const labels = c.params['labels'] as string[]
      const fitted = P.crf<TokenTable>({
        features: P.crfppExtractor(parseCrfppTemplate(c.params['template'] as string)),
        c2: c.params['c2'] as number,
        labels,
        minAttributeFrequency: c.params['minFrequency'] as number,
        allPossibleStates: true,
        allPossibleTransitions: true,
        epsilon: 1e-12,
        delta: 0,
      }).fit(P.crfData(tables, c.params['y'] as string[][]))
      expect(fitted.model.bigramAttributes).toEqual(['B'])
      expect(fitted.transitionFeatures()).toEqual({})
      expectClose(flat(fitted.stateFeatures()), c.params['state'] as Record<string, number>, 2e-6)
      const edge = Object.fromEntries(
        Object.entries(fitted.bigramFeatures()['B'] ?? {}).map(([k, w]) => [k.replace('->', '|'), w]),
      )
      expectClose(edge, c.params['transition'] as Record<string, number>, 2e-6)
      expect(fitted.predictSingle(c.params['test'] as TokenTable)).toEqual(c.params['prediction'])
    }
  })
})

describe('the CRF estimator against the functional API', () => {
  it('gives the weights of trainCrf for the same penalty, by attribute and label', () => {
    const start = P.crfModelFromInputs(['P', 'Q'], X.map(P.crfSequence))
    const trained = P.trainCrf(
      start,
      X.map((x, i) => ({ x, y: Y[i] ?? [] })),
      { c2: 0.3, tol: 1e-10 },
      500,
    )
    const fitted = P.crf({
      c2: 0.3,
      epsilon: 1e-10,
      delta: 0,
      allPossibleStates: true,
      allPossibleTransitions: true,
    }).fit(P.crfData(X, Y))
    const K = 2
    for (const [i, a] of trained.crf.attributes.entries())
      for (const [k, label] of ['P', 'Q'].entries())
        expect(
          Math.abs((fitted.stateFeatures()[a]?.[label] ?? 0) - (trained.crf.stateWeights[i * K + k] ?? 0)),
        ).toBeLessThan(1e-5)
    for (const [i, from] of ['P', 'Q'].entries())
      for (const [k, to] of ['P', 'Q'].entries())
        expect(
          Math.abs((fitted.transitionFeatures()[from]?.[to] ?? 0) - (trained.crf.transitionWeights[i * K + k] ?? 0)),
        ).toBeLessThan(1e-5)
    expect(Math.abs(fitted.params.objective - trained.objective)).toBeLessThan(1e-8)
  })

  it('trains with the online algorithms and the structured SVMs over the same slot', () => {
    for (const algorithm of ['ap', 'pa', 'arow', 'perceptron', 'bcfw', 'one-slack'] as const) {
      const fitted = P.crf({ algorithm, c2: 0.1, maxIterations: 60 }).fit(P.crfData(X, Y))
      expect(fitted.predict(X), algorithm).toEqual(Y)
    }
    const sgd = P.crf({ algorithm: 'l2sgd', c2: 0.1, maxIterations: 400, delta: 0 }).fit(P.crfData(X, Y))
    expect(sgd.score(X, Y)).toBeGreaterThan(0.7)
  })

  it('keeps masked features at zero under every kind of solver', () => {
    for (const algorithm of ['lbfgs', 'bfgs', 'ap', 'bcfw', 'owlqn'] as const) {
      const fitted = P.crf({ algorithm, c2: 0.1, maxIterations: 30 }).fit(P.crfData(X, Y))
      expect(fitted.params.state.length).toBe(6)
      // c|P never occurs: CRFsuite does not generate it
      expect(fitted.params.state[4], algorithm).toBe(0)
      expect(fitted.params.transition[0], algorithm).toBe(0)
      expect(fitted.params.transition[3], algorithm).toBe(0)
    }
  })

  it('gives exact zeros with L1 (OWL-QN) where L2 alone gives none', () => {
    const l2 = P.crf({ ...tight, allPossibleStates: true }).fit(P.crfData(X, Y))
    const l1 = P.crf({ ...tight, c1: 0.4, allPossibleStates: true }).fit(P.crfData(X, Y))
    expect(l2.params.state.every((w) => w !== 0)).toBe(true)
    expect(l1.params.state.filter((w) => w === 0).length).toBeGreaterThan(0)
    expect(l1.params.nonzeros).toBeLessThan(l2.params.nonzeros)
    expect(Object.keys(flat(l1.stateFeatures())).length).toBeLessThan(Object.keys(flat(l2.stateFeatures())).length)
    // the same through the named algorithm and through the slot value
    const named = P.crf({ ...tight, c1: 0.4, algorithm: 'owlqn', allPossibleStates: true }).fit(P.crfData(X, Y))
    const slot = P.crf({ ...tight, c1: 0.4, solver: { kind: 'owlqn' }, allPossibleStates: true }).fit(P.crfData(X, Y))
    expect(named.params.state).toEqual(l1.params.state)
    expect(slot.params.state).toEqual(l1.params.state)
  })
})

describe('the CRF estimator as an estimator', () => {
  it('takes items as dictionaries, name lists, single names and string values, as python-crfsuite does', () => {
    expect(P.crfItem({ w: 'cat', n: 2.5, up: true, low: false, zero: 0 })).toEqual({ 'w:cat': 1, n: 2.5, up: 1 })
    expect(P.crfItem(['a', 'b', 'a'])).toEqual({ a: 2, b: 1 })
    expect(P.crfItem('a')).toEqual({ a: 1 })
    const asLists = P.crf({ c2: 0.5 }).fit(
      P.crfData(
        [
          [['f=a'], ['f=b']],
          [['f=b'], ['f=a']],
        ],
        [
          ['X', 'Y'],
          ['Y', 'X'],
        ],
      ),
    )
    const asStrings = P.crf({ c2: 0.5 }).fit(
      P.crfData(
        [
          ['f=a', 'f=b'],
          ['f=b', 'f=a'],
        ],
        [
          ['X', 'Y'],
          ['Y', 'X'],
        ],
      ),
    )
    const asDicts = P.crf({ c2: 0.5 }).fit(
      P.crfData(
        [
          [{ 'f=a': 1 }, { 'f=b': 1 }],
          [{ 'f=b': 1 }, { 'f=a': 1 }],
        ],
        [
          ['X', 'Y'],
          ['Y', 'X'],
        ],
      ),
    )
    expect(asStrings.params.state).toEqual(asLists.params.state)
    expect(asDicts.params.state).toEqual(asLists.params.state)
    expect(asLists.predictSingle(['f=a', 'f=b'])).toEqual(['X', 'Y'])
  })

  it('ignores attributes unseen in training, and handles empty sequences', () => {
    const fitted = P.crf(tight).fit(P.crfData([...X, []], [...Y, []]))
    expect(
      fitted.predictSingle([
        { a: 1, never: 5 },
        { c: 1, other: 1 },
      ]),
    ).toEqual(fitted.predictSingle([{ a: 1 }, { c: 1 }]))
    expect(fitted.predictSingle([{ zzz: 1 }])).toHaveLength(1)
    expect(fitted.predict([[], X[0] ?? []])[0]).toEqual([])
    expect(fitted.predictMarginalsSingle([])).toEqual([])
    expect(fitted.score([[]], [[]])).toBeNaN()
    expect(() => P.crf().fit(P.crfData([[]], [[]]))).toThrow(/no non-empty/)
    expect(() => P.crf().fit({ x: [0] } as unknown as P.CrfData)).toThrow(/needs sequences/)
    expect(() => P.crfData([[{ a: 1 }]], [])).toThrow(/label lists/)
    expect(() => P.crf().fit(P.crfData([[{ a: 1 }]], [['X', 'Y']]))).toThrow(/items and/)
  })

  it('serialises, revives, clones and keeps the original untouched', () => {
    const estimator = P.crf(tight)
    const fitted = estimator.fit(P.crfData(X, Y))
    const json = JSON.parse(JSON.stringify(fitted.serialise())) as { estimator: string; params: P.CrfParams }
    expect(json.estimator).toBe('pgm/crf')
    const revived = estimator.revive(json.params)
    expect(revived.params).toEqual(fitted.params)
    expect(revived.predict(X)).toEqual(fitted.predict(X))
    expect(revived.predictMarginals(X)).toEqual(fitted.predictMarginals(X))
    expect(revived.run.events).toEqual([])
    const stronger = estimator.with({ c2: 5 })
    expect(stronger.hyper.c2).toBe(5)
    expect(estimator.hyper.c2).toBe(0.1)
    expect(stronger.fit(P.crfData(X, Y)).params.nonzeros).toBeGreaterThan(0)
    expect(stronger.fit(P.crfData(X, Y)).params.state).not.toEqual(fitted.params.state)
    expect(estimator.hyperSpace?.spaces).toHaveProperty('c2')
  })

  it('scores as sklearn-crfsuite does: flat accuracy over the items, not whole sequences', () => {
    const fitted = P.crf({ c2: 0.1 }).fit(P.crfData(X, Y))
    const wrong = [
      ['P', 'P'],
      ['P', 'Q'],
      ['Q', 'P'],
    ]
    expect(fitted.score(X, wrong)).toBeCloseTo(5 / 6, 12)
    expect(() => fitted.score(X, [['P']])).toThrow(/label lists/)
  })

  it('exposes the sequence score, the log-likelihood and the explanation', () => {
    const fitted = P.crf(tight).fit(P.crfData(X, Y))
    const x = X[0] ?? []
    const best = fitted.scoreSequence(x, ['P', 'Q'])
    expect(best).toBeGreaterThan(fitted.scoreSequence(x, ['Q', 'P']))
    expect(fitted.logLikelihood(x, ['P', 'Q'])).toBeCloseTo(
      Math.log(fitted.predictMarginalsSingle(x)[0]?.['P'] ?? 0),
      -1,
    )
    const total = ['P', 'Q'].flatMap((a) => ['P', 'Q'].map((b) => Math.exp(fitted.logLikelihood(x, [a, b]))))
    expect(total.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 10)
    const top = fitted.topFeatures({ top: 2 })
    expect(top.labels.map((l) => l.label)).toEqual(['P', 'Q'])
    expect(top.labels[1]?.features.names[0]).toBe('c')
    expect(fitted.exposition.length).toBeGreaterThan(0)
    expect(fitted.attributes).toEqual(['a', 'b', 'c'])
    expect(() => fitted.templateWeights([])).toThrow(/no `features`/)
  })

  it('decodes through the decoder hook', () => {
    const fitted = P.crf({ ...tight, decoder: (_crf, x) => P.crfUnigrams(x).map(() => 'Q') }).fit(P.crfData(X, Y))
    expect(fitted.predict(X)).toEqual([
      ['Q', 'Q'],
      ['Q', 'Q'],
      ['Q', 'Q'],
    ])
  })

  it('wraps in the CRF class as sklearn-crfsuite uses it', () => {
    const model = new P.CRF({ ...tight, allPossibleTransitions: true })
    expect(() => model.predict(X)).toThrow(/call fit first/)
    model.fit(X, Y)
    expect(model.predict(X)).toEqual(Y)
    expect(model.predictSingle(X[2] ?? [])).toEqual(['Q', 'P'])
    expect(model.classes).toEqual(['P', 'Q'])
    expect(Object.keys(model.transitionFeatures)).toEqual(['P', 'Q'])
    expect(model.stateFeatures['a']?.['P']).toBeGreaterThan(0)
    expect(model.predictMarginals(X)[0]?.[0]).toHaveProperty('P')
    expect(model.score(X, Y)).toBe(1)
    expect(model.getParams().c2).toBe(0.1)
    model.setParams({ c2: 2 })
    expect(() => model.predict(X)).toThrow(/call fit first/)
  })
})

describe('the CRF estimator on the log contract', () => {
  it('keeps the training tape of every fit, with the objective falling and the sequence error reaching zero', () => {
    const fitted = P.crf({ c2: 0.1, delta: 0 }).fit(P.crfData(X, Y))
    const objective = metricSeries(fitted.run, 'solver/objective').values
    expect(objective.length).toBeGreaterThan(2)
    expect(objective[objective.length - 1]).toBeLessThan(objective[0] ?? 0)
    expect(metricSeries(fitted.run, 'solver/sequenceError').values.at(-1)).toBe(0)
    expect(metricSeries(fitted.run, 'solver/nonzeros').values.at(-1)).toBe(fitted.params.nonzeros)
    expect(metricSeries(fitted.run, 'solver/gradient').values.at(-1)).toBeLessThan(1e-4)
    const loud = P.crf({ c2: 0.1, verbose: true }).fit(P.crfData(X, Y))
    expect(loud.run.events.length).toBeGreaterThan(fitted.run.events.length)
  })

  it('runs as a live session with the model at any reached step', () => {
    const data = P.crfData(X, Y)
    const s = P.crf({ c2: 0.1, delta: 0 }).session(data, { mode: 'eager', steps: 30 })
    expect(s.t).toBeGreaterThan(1)
    const early = s.fittedAt(1)
    const late = s.fittedAt(s.t)
    expect(late.score(X, Y)).toBe(1)
    expect(early.params.objective).toBeGreaterThan(late.params.objective)
    expect(early.predict(X)).toHaveLength(3)
    expect(P.crf().method(data).name).toBe('pgm/crf')
  })

  it('stops by the CRFsuite criterion: relative improvement over `period` iterations below `delta`', () => {
    const data = P.crfData(X, Y)
    const eager = P.crf({ c2: 0.1, delta: 0, epsilon: 1e-14, maxIterations: 200 }).session(data, { mode: 'eager' })
    const stopped = P.crf({ c2: 0.1, delta: 1e-2, period: 2, epsilon: 1e-14, maxIterations: 200 }).session(data, {
      mode: 'eager',
    })
    expect(stopped.t).toBeLessThan(eager.t)
    expect(stopped.status).toBe('converged')
  })
})

describe('the CRF estimator on word hyphenation', () => {
  const words = hyphenationWords()
  const tables = hyphenationTables(words.tokens)
  const labels = hyphenationLabels(words.tags)
  const train = { tables: tables.slice(0, 300), labels: labels.slice(0, 300) }
  const test = { tables: tables.slice(300, 700), labels: labels.slice(300, 700) }
  // 300 words are few for hyphenation: the learned tagger beats the baseline that never hyphenates by a margin
  const never = test.labels.flat().filter((l) => l === 'O').length / test.labels.flat().length
  const template = [
    'U00:%x[-2,0]',
    'U01:%x[-1,0]',
    'U02:%x[0,0]',
    'U03:%x[1,0]',
    'U04:%x[2,0]',
    'U05:%x[-1,0]/%x[0,0]',
    'U06:%x[0,0]/%x[1,0]',
    'U07:%x[0,1]',
    'B',
  ].join('\n')

  it('learns to hyphenate from 300 words in seconds with CRF++ templates', () => {
    const started = Date.now()
    const fitted = P.crf<TokenTable>({ features: P.crfppExtractor(template), c2: 1 }).fit(
      P.crfData(train.tables, train.labels),
    )
    expect(Date.now() - started).toBeLessThan(20_000)
    expect(fitted.classes.slice().sort()).toEqual(['B', 'O'])
    expect(fitted.score(test.tables, test.labels)).toBeGreaterThan(never + 0.03)
    expect(fitted.score(train.tables, train.labels)).toBeGreaterThan(0.8)
    // the explanation: which templates carry the model
    const templates = fitted.templateWeights(train.tables.slice(0, 50))
    expect(templates.map((t) => t.template)).toContain('U07:%x[0,1]')
    expect(fitted.topFeatures({ top: 3 }).labels).toHaveLength(2)
  })

  it('learns the same task with CRFsuite attribute templates and label transitions', () => {
    const features = P.attributeExtractor(
      ['w[-2]', 'w[-1]', 'w[0]', 'w[1]', 'w[2]', 'w[-1]|w[0]', 'w[0]|w[1]', 'c[0]'].join('\n'),
      { columns: ['w', 'c'], bias: true },
    )
    const fitted = P.crf<TokenTable>({ features, c2: 1, minFreq: 2 }).fit(P.crfData(train.tables, train.labels))
    expect(fitted.model.transitionWeights.length).toBe(4)
    expect(fitted.score(test.tables, test.labels)).toBeGreaterThan(never + 0.03)
    const hashed = P.crf<TokenTable>({ features: P.hashedExtractor(features, { nFeatures: 4096 }), c2: 1 }).fit(
      P.crfData(train.tables, train.labels),
    )
    expect(hashed.score(test.tables, test.labels)).toBeGreaterThan(never + 0.02)
  })

  it('sparsifies with L1 and still hyphenates', () => {
    const features = P.crfppExtractor(template)
    const dense = P.crf<TokenTable>({ features, c2: 0.5 }).fit(P.crfData(train.tables, train.labels))
    const sparse = P.crf<TokenTable>({ features, c1: 0.5, c2: 0.1 }).fit(P.crfData(train.tables, train.labels))
    expect(sparse.params.nonzeros).toBeLessThan(dense.params.nonzeros / 2)
    expect(sparse.score(test.tables, test.labels)).toBeGreaterThan(never + 0.02)
  })
})
