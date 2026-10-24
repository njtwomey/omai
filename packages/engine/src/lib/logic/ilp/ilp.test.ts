import { describe, expect, it } from 'vitest'
import { messages, policies, recorder, runAlgorithm, validateAgainstSchema } from '../../trace/index.ts'
import { clauseToString, parseClause, parseTerm } from '../index.ts'
import {
  candidateEliminationAlgorithm,
  canonicalClause,
  foil,
  foilAlgorithm,
  headConnected,
  learnFromFailures,
  learnFromFailuresAlgorithm,
  lggClauses,
  METARULES,
  metagol,
  metagolAlgorithm,
  parseMode,
  progol,
  progolAlgorithm,
  rlgg,
} from './index.ts'

const FAMILY = 'parent(ann, amy). parent(ann, andy). parent(amy, amelia). parent(andy, spongebob).'

describe('ILP', () => {
  it('rejects malformed input', () => {
    expect(() => lggClauses(parseClause('p(a).'), parseClause('q(a).'))).toThrow(/differ/)
    expect(() => rlgg([], [])).toThrow(/no examples/)
    expect(() =>
      rlgg([parseTerm('p(a)'), parseTerm('p(b)')], [parseTerm('q(a)'), parseTerm('q(b)')], { maxLiterals: 1 }),
    ).toThrow(/maxLiterals/)
    expect(() => parseMode('mode(1, p(+t))')).toThrow(/modeh/)
    expect(() => parseMode('modeh(1, p(t))')).toThrow(/modeh/)
    expect(() => foil({ target: 't', arity: 2, positive: [['a']], background: {} })).toThrow(/arity/)
    expect(() => metagol({ background: '', predicates: [], positive: [], negative: [], metarules: [] })).toThrow(
      /positive/,
    )
    expect(() =>
      metagolAlgorithm({ background: '', predicates: [], positive: ['p(a)'], negative: [], metarules: ['nope'] }),
    ).toThrow(/metarule/)
    expect(() => learnFromFailures({ background: '', head: 'p', body: [], positive: [], negative: [] })).toThrow(
      /name\/arity/,
    )
  })

  it('reports failure when the language cannot express the target', () => {
    const r = metagol(
      {
        background: FAMILY,
        predicates: ['parent/2'],
        positive: ['grandparent(ann, amelia)'],
        negative: [],
        metarules: ['ident'],
      },
      { maxClauses: 2 },
    )
    expect(r.status).toBe('failed')
    const f = foil({ target: 'p', arity: 1, positive: [['a']], negative: [['b']], background: { q: [['a'], ['b']] } })
    expect(f.status).toBe('incomplete')
    const l = learnFromFailures(
      { background: 'q(a). q(b).', head: 'p/1', body: ['q/1'], positive: ['p(a)'], negative: ['p(b)'] },
      { maxBody: 1, maxClauses: 1 },
    )
    expect(l.status).toBe('failed')
  })

  it('learns recursive theories and facts for exceptions', () => {
    const r = progol({
      background: 'q(a). q(b).',
      positive: ['p(a)', 'p(c)'],
      negative: ['p(b)'],
      modeh: 'modeh(1, p(+t))',
      modeb: ['modeb(1, q(+t))'],
    })
    // Nothing generalises p(a) without covering p(b), so the examples are kept as facts.
    expect(r.theory.map((c) => clauseToString(c))).toEqual(['p(a).', 'p(c).'])
    expect(clauseToString(canonicalClause(headConnected(parseClause('p(X) :- q(Y).'))))).toBe('p(A).')
  })

  it('traces every learner according to its schema', () => {
    const cases = [
      candidateEliminationAlgorithm([{ values: ['a'], positive: true }], [['a', 'b']]),
      foilAlgorithm({
        target: 'g',
        arity: 2,
        positive: [['ann', 'amelia']],
        background: {
          p: [
            ['ann', 'amy'],
            ['amy', 'amelia'],
          ],
        },
      }),
      progolAlgorithm({
        background: 'q(a).',
        positive: ['p(a)'],
        negative: [],
        modeh: 'modeh(1, p(+t))',
        modeb: ['modeb(1, q(+t))'],
      }),
      metagolAlgorithm({
        background: FAMILY,
        predicates: ['parent/2'],
        positive: ['g(ann, amelia)'],
        negative: [],
        metarules: [METARULES['chain'] ?? 'chain'],
      }),
      learnFromFailuresAlgorithm({
        background: FAMILY,
        head: 'g/2',
        body: ['parent/2'],
        positive: ['g(ann, amelia)'],
        negative: ['g(ann, amy)'],
      }),
    ]
    for (const a of cases) {
      const rec = recorder()
      const r = runAlgorithm(a as Parameters<typeof runAlgorithm>[0], {
        maxSteps: 5000,
        tracer: rec,
        policy: policies.teaching,
      })
      expect(r.status).toBe('converged')
      if (a.schema !== undefined) expect(validateAgainstSchema(rec.run().events, a.schema)).toEqual([])
      expect(rec.run().events.some((e) => e.type === 'message')).toBe(true)
    }
    const rec = recorder()
    learnFromFailures(
      {
        background: FAMILY,
        head: 'g/2',
        body: ['parent/2'],
        positive: ['g(ann, amelia)', 'g(ann, spongebob)'],
        negative: ['g(ann, amy)'],
      },
      { tracer: rec, policy: policies.teaching },
    )
    const outcomes = messages(rec.run(), 'test').map((m) => (m.payload as { outcome: string }).outcome)
    expect(outcomes[outcomes.length - 1]).toBe('solution')
    expect(outcomes.slice(0, -1).every((o) => o !== 'solution')).toBe(true)
  })
})
