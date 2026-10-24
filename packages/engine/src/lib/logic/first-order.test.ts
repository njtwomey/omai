import { describe, expect, it } from 'vitest'
import { messages, policies, recorder, validateAgainstSchema } from '../trace/index.ts'
import {
  checkProof,
  clausalForm,
  clauseText,
  folToString,
  hasFiniteModel,
  parseFol,
  proofStructure,
  proveFol,
  resolutionAlgorithm,
  subsumes,
} from './index.ts'

const cnfText = (s: string): string[] => clausalForm(parseFol(s)).map(clauseText)

describe('first-order syntax and clausal form', () => {
  it('parses quantifiers, connectives and equality', () => {
    expect(folToString(parseFol('all x y. (r(x, y) -> exists z. (r(y, z) & z != x))'))).toBe(
      '∀x ∀y (r(x,y) → ∃z (r(y,z) ∧ ¬z=x))',
    )
    expect(folToString(parseFol('∀x.∃y.loves(x, y)'))).toBe('∀x ∃y loves(x,y)')
    expect(folToString(parseFol('forall X. p(X) <-> not q(X) or true'))).toBe('∀X (p(X) ↔ (¬q(X) ∨ ⊤))')
    expect(() => parseFol('all . p')).toThrow(/variable/)
    expect(() => parseFol('all x. x')).toThrow(/not a formula/)
    expect(() => parseFol('p(a')).toThrow(/expected \)/)
    expect(() => parseFol('p $ q')).toThrow(/unexpected/)
  })

  it('Skolemises by the enclosing universals and standardises apart', () => {
    expect(cnfText('all x. exists y. all z. exists w. r(x, y, z, w)')).toEqual(['r(X1,sk1(X1),X2,sk2(X1,X2))'])
    expect(cnfText('(exists x. p(x)) & (exists x. q(x))')).toEqual(['p(sk1)', 'q(sk2)'])
    expect(cnfText('all x. (p(x) <-> q(x))')).toEqual(['¬p(X1) ∨ q(X1)', 'p(X1) ∨ ¬q(X1)'])
    expect(cnfText('p(X) | -p(X)')).toEqual([])
    expect(cnfText('false')).toEqual(['□'])
    expect(cnfText('p(sk1) & exists x. q(x)')).toEqual(['p(sk1)', 'q(sk2)'])
    expect(clausalForm(parseFol('exists x. p(x)'), { skolemPrefix: 'c', skolemStart: 7 }).map(clauseText)).toEqual([
      'p(c7)',
    ])
  })

  it('subsumes clauses', () => {
    const [general] = clausalForm(parseFol('p(X, Y) | q(Y)'))
    const [specific] = clausalForm(parseFol('p(a, b) | q(b) | r(c)'))
    const [other] = clausalForm(parseFol('p(a, b) | q(c)'))
    if (general === undefined || specific === undefined || other === undefined) throw new Error('clauses')
    expect(subsumes(general, specific)).not.toBeNull()
    expect(subsumes(general, other)).toBeNull()
    const [two] = clausalForm(parseFol('p(X) | p(Y)'))
    const [one] = clausalForm(parseFol('p(a)'))
    if (two === undefined || one === undefined) throw new Error('clauses')
    expect(subsumes(two, one)).not.toBeNull()
    expect(subsumes(two, one, { strict: true })).toBeNull()
  })
})

describe('resolution proving', () => {
  it('proves, saturates and runs out of budget', () => {
    const r = proveFol(['all x.(p(x) -> p(f(x)))', 'p(a)'], 'p(f(f(f(a))))')
    expect(r.status).toBe('theorem')
    expect(checkProof(r.proof)).toBe(true)
    const tree = proofStructure(r.proof)
    expect(tree.nodes.filter((n) => n.role === 'output').map((n) => n.label)).toEqual(['□'])
    // With a set of support only the negated conjecture is given: it resolves with nothing, so the set saturates.
    expect(proveFol(['all x.(p(x) -> p(f(x)))', 'p(a)'], 'q(a)').status).toBe('countersatisfiable')
    // Without it the axioms generate p(f(a)), p(f(f(a))), ... for ever.
    expect(proveFol(['all x.(p(x) -> p(f(x)))', 'p(a)'], 'q(a)', { setOfSupport: false, maxSteps: 50 }).status).toBe(
      'unknown',
    )
    expect(proveFol(['p', '-p'], null).status).toBe('theorem')
    expect(proveFol(['p'], null).status).toBe('countersatisfiable')
    expect(checkProof([])).toBe(false)
  })

  it('traces given clauses and kept resolvents', () => {
    const axioms = clausalForm([parseFol('all x.(man(x) -> mortal(x))'), parseFol('man(socrates)')])
    const goal = clausalForm(parseFol('-mortal(socrates)'))
    const rec = recorder()
    const r = proveFol(['all x.(man(x) -> mortal(x))', 'man(socrates)'], 'mortal(socrates)', {
      tracer: rec,
      policy: policies.teaching,
    })
    expect(r.status).toBe('theorem')
    const run = rec.run()
    expect(messages(run, 'given').map((m) => (m.payload as { clause: string }).clause)).toEqual([
      '¬mortal(socrates)',
      '¬man(socrates)',
    ])
    expect(messages(run, 'proof').length).toBe(1)
    const schema = resolutionAlgorithm(axioms, goal).schema
    if (schema !== undefined) expect(validateAgainstSchema(run.events, schema)).toEqual([])
  })

  it('checks finite models of saturated sets', () => {
    expect(hasFiniteModel(clausalForm([parseFol('p(a)'), parseFol('-p(a)')]), ['a'])).toBe(false)
    expect(() => hasFiniteModel(clausalForm(parseFol('p(f(a))')), ['a'])).toThrow(/function/)
  })
})
