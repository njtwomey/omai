import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { bpeDropouts, parsingStrategies, punktProfiles } from './algorithm-options.ts'
import {
  cyk,
  cykAlgorithm,
  cykResult,
  earley,
  earleyAlgorithm,
  earleyResult,
  grammar,
  parseContract,
  toChomskyNormalForm,
  treeToString,
} from './parsing.ts'
import { punktParametersOf, trainPunkt, trainPunktAlgorithm } from './sentences.ts'
import {
  bpeModel,
  bpeTokenize,
  bpeTokenizeAlgorithm,
  bpeTokens,
  trainBpe,
  wordPieceAlgorithm,
  wordPieceModel,
  wordPieceTokenize,
} from './subword/index.ts'
import { tokenizerContract } from './tokenizer-contract.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const metric = (rec: ReturnType<typeof recorder>, name: string): number[] =>
  rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
const logPolicy = { ...policies.minimal, metrics: true, detail: 'full' } as const

const cnf = grammar(
  "S -> NP VP\nVP -> V NP | VP PP\nPP -> P NP\nNP -> NP PP | 'she' | 'fish' | 'forks'\nV -> 'eats'\nP -> 'with'",
)
const sentence = 'she eats fish with forks'.split(' ')

describe('cykAlgorithm', () => {
  it('fills one span length per step and counts the derivations, as cyk does', () => {
    const run = runAlgorithm(cykAlgorithm(cnf, sentence), { maxSteps: 10 })
    expect([run.status, run.steps]).toEqual(['converged', sentence.length])
    const result = cykResult(cnf, sentence, run.state)
    expect(result.count).toBe(2)
    expect(result.trees.map(treeToString)).toEqual(cyk(cnf, sentence).trees.map(treeToString))
    // Every cell of length 1 holds the lexical categories.
    expect(Object.keys(run.state.cells[0]?.[1] ?? {})).toEqual(['NP'])
  })

  it('resumes from JSON, logs the span, and rejects grammars outside CNF', () => {
    const alg = cykAlgorithm(cnf, sentence)
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 10,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(rest.state.cells).toEqual(whole.state.cells)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy: logPolicy })
    expect(metric(rec, 'chart/span')).toEqual([1, 2, 3, 4, 5])
    expect(Object.keys(parseContract)).toHaveLength(7)
    expect(() => cykAlgorithm(grammar("S -> 'a' S 'b' | 'a' 'b'"), ['a', 'b'])).toThrow(/Chomsky/)
    expect(runAlgorithm(cykAlgorithm(cnf, []), { maxSteps: 3 }).steps).toBe(0)
  })
})

describe('earleyAlgorithm', () => {
  const g = grammar(
    "S -> NP VP\nNP -> Det N | NP PP | 'I'\nVP -> V NP | VP PP\nPP -> P NP\nDet -> 'the'\nN -> 'man' | 'telescope'\nV -> 'saw'\nP -> 'with'",
  )
  const words = 'I saw the man with the telescope'.split(' ')

  it('processes one item per step and finds both attachments', () => {
    const run = runAlgorithm(earleyAlgorithm(g, words), { maxSteps: 2000 })
    expect(run.status).toBe('converged')
    const total = run.state.chart.reduce((s, c) => s + c.length, 0)
    expect(run.steps).toBe(total)
    const result = earleyResult(g, words, run.state)
    expect(result.trees).toHaveLength(2)
    expect(earley(g, words)).toEqual(result)
    expect(result.items).toBe(total)
  })

  it('handles nullable symbols and unrecognised sentences, resumes and logs', () => {
    const nullable = grammar("S -> A B\nA -> 'a' |\nB -> 'b'")
    expect(earley(nullable, ['b']).recognised).toBe(true)
    expect(earley(g, ['saw', 'I']).recognised).toBe(false)
    const alg = earleyAlgorithm(g, words)
    const whole = runAlgorithm(alg, { maxSteps: 2000 })
    const part = runAlgorithm(alg, { maxSteps: 40 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 2000,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 40, serialised: true },
    })
    expect(rest.state.chart).toEqual(whole.state.chart)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 2000, tracer: rec, policy: logPolicy })
    const added = metric(rec, 'earley/predict').reduce(
      (s, v, i) => s + v + (metric(rec, 'earley/scan')[i] ?? 0) + (metric(rec, 'earley/complete')[i] ?? 0),
      0,
    )
    // Every item but the initial ones was added by one of the three operations.
    expect(added + g.productions.filter((p) => p.lhs === 'S').length).toBe(
      whole.state.chart.reduce((s, c) => s + c.length, 0),
    )
  })

  it('agrees with cyk on a grammar converted to Chomsky normal form', () => {
    const converted = toChomskyNormalForm(g)
    expect(cyk(converted, words).count).toBe(earley(g, words).trees.length)
  })
})

describe('parsingStrategies', () => {
  it('builds either parser', () => {
    expect(parsingStrategies.names).toEqual(['cyk', 'earley'])
    const alg = parsingStrategies.make({ kind: 'cyk', convert: true })(grammar("S -> 'a' S 'b' | 'a' 'b'"), [
      'a',
      'a',
      'b',
      'b',
    ])
    expect(runAlgorithm(alg, { maxSteps: 10 }).status).toBe('converged')
  })
})

describe('bpeTokenizeAlgorithm', () => {
  const model = bpeModel(
    new Map([
      ['a', 0],
      ['b', 1],
      ['c', 2],
      ['ab', 3],
      ['abc', 4],
      ['bc', 5],
    ]),
    [
      ['a', 'b'],
      ['ab', 'c'],
      ['b', 'c'],
    ],
  )

  it('applies one merge per step, lowest rank first, and equals bpeTokenize', () => {
    const run = runAlgorithm(bpeTokenizeAlgorithm(model, 'abcabc'), { maxSteps: 20 })
    expect(run.status).toBe('converged')
    expect(bpeTokens(model, run.state)).toEqual(bpeTokenize(model, 'abcabc'))
    const rec = recorder()
    runAlgorithm(bpeTokenizeAlgorithm(model, 'abcabc'), { maxSteps: 20, tracer: rec, policy: logPolicy })
    const ranks = metric(rec, 'tok/merge')
    expect(ranks).toEqual([...ranks].sort((a, b) => a - b))
    expect(ranks.length).toBeGreaterThan(0)
    expect(bpeTokenize(model, 'abcabc').map((t) => t.token)).toEqual(['abc', 'abc'])
    expect(Object.keys(tokenizerContract)).toHaveLength(5)
  })

  it('draws the same dropout coins after a resume and needs a key', () => {
    const trained = trainBpe(
      new Map([
        ['lowest', 7],
        ['newer', 6],
        ['wider', 3],
        ['low', 5],
      ]),
      { vocabSize: 40 },
    )
    const opts = { dropout: 0.4, key: key(2) }
    const alg = bpeTokenizeAlgorithm(trained, 'lowernewest', opts)
    const whole = runAlgorithm(alg, { maxSteps: 50 })
    for (let cut = 1; cut < whole.steps; cut++) {
      const part = runAlgorithm(alg, { maxSteps: cut }).state
      const rest = runAlgorithm(alg, {
        maxSteps: 50,
        start: { state: viaJson(alg.serialise?.(part) ?? null), step: cut, serialised: true },
      })
      expect(rest.state.symbols).toEqual(whole.state.symbols)
    }
    expect(bpeTokenize(trained, 'lowernewest', opts)).toEqual(bpeTokens(trained, whole.state))
    expect(() => bpeTokenize(trained, 'lowernewest', { dropout: 0.4 })).toThrow(/needs a key/)
    expect(bpeDropouts.make({ kind: 'heavy' }).dropout).toBe(0.5)
  })
})

describe('wordPieceAlgorithm', () => {
  const model = wordPieceModel(['[UNK]', 'un', '##aff', '##able', 'a'])

  it('takes one piece per step, the longest first', () => {
    const run = runAlgorithm(wordPieceAlgorithm(model, 'unaffable'), { maxSteps: 10 })
    expect([run.status, run.steps]).toEqual(['converged', 3])
    expect(run.state.pieces).toEqual(wordPieceTokenize(model, 'unaffable'))
    expect(run.state.pieces.map((p) => p.token)).toEqual(['un', '##aff', '##able'])
    const unknown = runAlgorithm(wordPieceAlgorithm(model, 'unx'), { maxSteps: 10 })
    expect(unknown.state.unknown).toBe(true)
    expect(wordPieceTokenize(model, 'unx').map((p) => p.token)).toEqual(['[UNK]'])
    expect(wordPieceTokenize(model, '')).toEqual([])
    const alg = wordPieceAlgorithm(model, 'unaffable')
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 10,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state).toEqual(run.state)
  })
})

describe('trainPunktAlgorithm', () => {
  const text = 'Mr. Smith arrived. He saw Mr. Jones. Mr. Brown left. Then Mr. Smith left too. '.repeat(3)

  it('scores one abbreviation candidate per step and finishes with the parameters of trainPunkt', () => {
    const run = runAlgorithm(trainPunktAlgorithm(text), { maxSteps: 500 })
    expect(run.status).toBe('converged')
    const params = punktParametersOf(run.state)
    expect(params.abbrevTypes.has('mr')).toBe(true)
    const direct = trainPunkt(text)
    expect([...direct.abbrevTypes].sort()).toEqual([...params.abbrevTypes].sort())
    expect([...direct.sentStarters].sort()).toEqual([...params.sentStarters].sort())
    expect([...direct.collocations].sort()).toEqual([...params.collocations].sort())
    expect(Array.from(direct.orthoContext).sort()).toEqual(Array.from(params.orthoContext).sort())
  })

  it('logs scores, resumes from JSON and builds profiles', () => {
    const alg = trainPunktAlgorithm(text)
    const whole = runAlgorithm(alg, { maxSteps: 500 })
    const part = runAlgorithm(alg, { maxSteps: 3 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 500,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 3, serialised: true },
    })
    expect(rest.state.result).toEqual(whole.state.result)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 500, tracer: rec, policy: logPolicy })
    expect(metric(rec, 'punkt/score').length).toBeGreaterThan(0)
    expect(() => punktParametersOf(alg.init(key(0)))).toThrow(/not finished/)
    const strict = trainPunkt(text, punktProfiles.make({ kind: 'strict', abbrev: 1000 }))
    expect(strict.abbrevTypes.has('mr')).toBe(false)
  })
})
