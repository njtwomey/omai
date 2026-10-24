import { describe, expect, it } from 'vitest'
import { messages, policies, recorder, runAlgorithm, validateAgainstSchema } from '../trace/index.ts'
import {
  datalog,
  datalogAlgorithm,
  derivationTree,
  evaluateDatalog,
  groundProgram,
  groundRuleToString,
  leastModel,
  normalProgram,
  queryModel,
  reduct,
  stableModels,
} from './index.ts'

describe('Datalog', () => {
  it('rejects function symbols, unsafe rules and recursion through negation', () => {
    expect(() => datalog('p(f(X)) :- q(X).')).toThrow(/function symbols/)
    expect(() => datalog('p(X) :- not q(X).')).toThrow(/unsafe/)
    expect(() => datalog('p(X) :- q(X), X < Y.')).toThrow(/unsafe/)
    expect(() => datalog('p :- not p.')).toThrow(/stratify/)
    expect(() => datalog('p(X).')).toThrow(/unsafe/)
    const manual = {
      rules: [{ head: datalog('p(X) :- q(X).').rules[0]?.head ?? { kind: 'atom' as const, name: 'p' }, body: [] }],
      strata: [['p/1']],
      edb: [],
      idb: [],
    }
    expect(() => evaluateDatalog(manual)).toThrow(/not ground/)
    expect(() => evaluateDatalog('n(a). p(X) :- n(X), X < 3.')).toThrow(/compares numbers/)
  })

  it('evaluates in strata and answers queries', () => {
    const src =
      'e(1, 2). e(2, 3). e(3, 1). e(3, 4). r(X, Y) :- e(X, Y). r(X, Z) :- r(X, Y), e(Y, Z). cyc(X) :- r(X, X). acyc(X) :- e(X, _), not cyc(X). acyc(X) :- e(_, X), not cyc(X).'
    const r = evaluateDatalog(src)
    expect(r.program.strata.length).toBe(2)
    expect(queryModel(r.model, 'acyc(X)')).toEqual([{ X: '4' }])
    expect(queryModel(r.model, 'r(1, X), X > 2')).toEqual([{ X: '3' }, { X: '4' }])
    expect(queryModel(r.model, 'r(X, Y), not r(Y, X)').length).toBe(3)
    expect(leastModel('p.')).toEqual(['p'])
    const tree = derivationTree(r, 'cyc(1)')
    expect(tree.nodes[0]?.detail).toBe('cyc(X) :- r(X,X).')
    expect(() => derivationTree(r, 'cyc(4)')).toThrow(/not in the model/)
  })

  it('traces rounds', () => {
    const p = datalog('e(1, 2). e(2, 3). p(X, Y) :- e(X, Y). p(X, Z) :- p(X, Y), e(Y, Z).')
    const rec = recorder()
    const run = runAlgorithm(datalogAlgorithm(p), { maxSteps: 100, tracer: rec, policy: policies.teaching })
    expect(run.status).toBe('converged')
    const rounds = messages(rec.run(), 'round').map((m) => (m.payload as { derived: number }).derived)
    expect(rounds).toEqual([2, 1, 0])
    const schema = datalogAlgorithm(p).schema
    if (schema !== undefined) expect(validateAgainstSchema(rec.run().events, schema)).toEqual([])
  })
})

describe('answer sets', () => {
  it('grounds, reduces and checks stability', () => {
    const p = normalProgram('n(1). n(2). in(X) :- n(X), not out(X). out(X) :- n(X), not in(X). :- in(1), in(2).')
    const g = groundProgram(p)
    expect(g.map(groundRuleToString)).toContain('in(1) :- n(1), not out(1).')
    expect(g.filter((r) => r.head === null).map(groundRuleToString)).toEqual([':- in(1), in(2).'])
    expect(reduct(g, new Set(['in(1)', 'out(2)'])).map((r) => r.head)).toEqual([
      'n(1)',
      'n(2)',
      'in(1)',
      'out(2)',
      null,
    ])
    expect(stableModels(p).models.length).toBe(3)
    expect(() =>
      stableModels(
        'a(1). a(2). a(3). a(4). a(5). a(6). a(7). a(8). a(9). a(10). a(11). b(X) :- a(X), not c(X). c(X) :- a(X), not b(X).',
      ),
    ).toThrow(/maxGuessAtoms/)
    expect(stableModels('a :- not b. b :- not a.', { maxModels: 1 }).models.length).toBe(1)
  })
})
