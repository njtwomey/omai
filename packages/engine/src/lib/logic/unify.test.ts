import { describe, expect, it } from 'vitest'
import { messages, policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  applySubstitution,
  Bindings,
  compareTerms,
  composeSubstitutions,
  isVariant,
  martelliMontanari,
  matchTerm,
  mgu,
  parseTerm,
  renameApart,
  restrictSubstitution,
  type Substitution,
  substitutionToString,
  type Term,
  termVariables,
  unificationAlgorithm,
  unify,
} from './index.ts'

const t = (s: string): Term => parseTerm(s)
const same = (a: Term, b: Term): boolean => compareTerms(a, b) === 0

describe('unification', () => {
  it('finds most general unifiers', () => {
    expect(substitutionToString(mgu(t('f(X, g(Y, a))'), t('f(h(Z), g(Z, W))')))).toBe('{W/a, X/h(Z), Y/Z}')
    expect(mgu(t('f(X, X)'), t('f(a, b)'))).toBeNull()
    expect(mgu(t('X'), t('f(X)'))).toBeNull()
    expect(unify(t('X'), t('f(X)'))).not.toBeNull()
    expect(mgu(t('p(X, Y, Z)'), t('p(Y, Z, c)'))?.get('X')).toEqual(t('c'))
    expect(mgu(t('1'), t('1.0'))).toBeNull()
  })

  it('composes, restricts and renames', () => {
    const s1: Substitution = new Map([['X', t('f(Y)')]])
    const s2: Substitution = new Map([['Y', t('b')]])
    const c = composeSubstitutions(s1, s2)
    const term = t('g(X, Y, Z)')
    expect(same(applySubstitution(term, c), applySubstitution(applySubstitution(term, s1), s2))).toBe(true)
    expect(substitutionToString(restrictSubstitution(c, t('p(X)')))).toBe('{X/f(b)}')
    const r = renameApart({ head: t('p(X, Y)'), body: [t('q(X)')] }, 0)
    expect(isVariant(r.clause.head, t('p(X, Y)'))).toBe(true)
    expect(termVariables(r.clause.head).every((v) => v.name.startsWith('_'))).toBe(true)
    expect(matchTerm(t('f(X, X)'), t('f(a, b)'))).toBeNull()
  })

  it('binds and undoes with a trail', () => {
    const b = new Bindings()
    const m0 = b.mark()
    expect(b.unify(t('f(X, Y)'), t('f(Y, a)'))).toBe(true)
    expect(same(b.resolve(t('X')), t('a'))).toBe(true)
    let undone = false
    b.onUndo(() => {
      undone = true
    })
    b.undo(m0)
    expect([b.size, undone]).toEqual([0, true])
    expect(b.unify(t('X'), t('f(X)'), true)).toBe(false)
  })

  it('traces the Martelli-Montanari rules', () => {
    const r = martelliMontanari(t('f(X, b)'), t('f(a, Y)'))
    expect(r.steps.map((s) => s.rule)).toEqual(['decompose', 'eliminate', 'swap', 'eliminate', 'solved'])
    expect(martelliMontanari(t('f(a)'), t('g(a)')).status).toBe('clash')
    expect(martelliMontanari(t('f(X)'), t('f(X)')).steps.map((s) => s.rule)).toEqual(['decompose', 'delete', 'solved'])
    expect(() => martelliMontanari(t('a'))).toThrow()
    const rec = recorder()
    runAlgorithm(unificationAlgorithm([{ left: t('g(X)'), right: t('g(c)') }]), {
      maxSteps: 10,
      tracer: rec,
      policy: policies.teaching,
    })
    expect(messages(rec.run(), 'rule').map((m) => (m.payload as { rule: string }).rule)).toEqual([
      'decompose',
      'eliminate',
    ])
  })
})
