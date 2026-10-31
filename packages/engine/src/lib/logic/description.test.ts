import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import {
  alcTableauAlgorithm,
  type Concept,
  classifyConcepts,
  conceptSatisfiable,
  conceptSubsumes,
  conceptText,
  nnf,
  parseConcept,
} from './index.ts'

describe('concepts', () => {
  it('parse with the usual precedence and print back', () => {
    expect(conceptText(parseConcept('A or B and C'))).toBe('(A or (B and C))')
    expect(conceptText(parseConcept('some R A and all R B'))).toBe('(some R A and all R B)')
    expect(conceptText(parseConcept('not not A'))).toBe('not (not A)')
    expect(() => parseConcept('A and')).toThrow()
    expect(() => parseConcept('(A')).toThrow()
  })

  it('negation normal form pushes negation to the names by the dualities', () => {
    expect(conceptText(nnf(parseConcept('not (some R A and all R B)')))).toBe('(all R (not A) or some R (not B))')
    expect(conceptText(nnf(parseConcept('not not A')))).toBe('A')
    expect(conceptText(nnf(parseConcept('not top')))).toBe('bottom')
  })
})

describe('satisfiability', () => {
  const cases: [string, boolean][] = [
    ['A and not A', false],
    ['A or B', true],
    ['bottom', false],
    ['some R A and all R not A', false],
    ['some R A and all R B', true],
    ['some R A and some R B and all R (not A or not B)', true],
    ['some R A and some R B and all R (not A or not B) and all R (A or B) and all R (not A)', false],
    ['all R bottom', true],
    ['some R bottom', false],
    ['some R (A and not A)', false],
    ['(some R A or some R B) and all R not A and all R not B', false],
    ['some R (some S A) and all R (all S not A)', false],
    ['some R A and all S not A', true],
  ]
  it.each(cases)('%s is %s', (c, expected) => {
    expect(conceptSatisfiable(c)).toBe(expected)
  })

  it('a clash closes a branch and the other disjunct is tried (backtracking)', () => {
    const alg = alcTableauAlgorithm('(A or B) and not A and (C or D) and not C')
    const { state } = runAlgorithm(alg, { maxSteps: 100, seed: 0 })
    expect(state.status).toBe('satisfiable')
    expect(state.clashes).toBeGreaterThanOrEqual(2)
  })

  it('every rule application is one step, the state serialises and the run ends in a verdict', () => {
    const alg = alcTableauAlgorithm('some R A and all R not A')
    const { state, steps } = runAlgorithm(alg, { maxSteps: 100, seed: 0 })
    expect(state.status).toBe('unsatisfiable')
    expect(steps).toBeGreaterThan(3)
    expect(alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(state))))).toEqual(state)
  })
})

describe('subsumption', () => {
  it('the textbook cases', () => {
    expect(conceptSubsumes('some R A', 'some R (A and B)')).toBe(true)
    expect(conceptSubsumes('some R (A and B)', 'some R A')).toBe(false)
    expect(conceptSubsumes('all R (A and B)', 'all R A and all R B')).toBe(true)
    expect(conceptSubsumes('all R A and all R B', 'all R (A and B)')).toBe(true)
    expect(conceptSubsumes('some R A and some R B', 'some R (A and B)')).toBe(true)
    expect(conceptSubsumes('some R (A and B)', 'some R A and some R B')).toBe(false)
    expect(conceptSubsumes('A', 'A and B')).toBe(true)
    expect(conceptSubsumes('top', 'A')).toBe(true)
    expect(conceptSubsumes('A', 'bottom')).toBe(true)
  })
})

describe('with a TBox, blocking, and classification', () => {
  const tbox = [
    ['Man', 'Human'],
    ['Woman', 'Human'],
    ['Human', 'some hasParent Human'],
    ['Mother', 'Woman'],
    ['Mother', 'some hasChild Human'],
  ] as const

  it('a cyclic inclusion (every human has a human parent) terminates by blocking and is satisfiable', () => {
    const alg = alcTableauAlgorithm('Human', tbox)
    const { state } = runAlgorithm(alg, { maxSteps: 1000, seed: 0 })
    expect(state.status).toBe('satisfiable')
    expect(state.nodes.some((n) => n.blocked)).toBe(true)
    expect(state.nodes.length).toBeLessThan(10)
  })

  it('inclusions are used: an inclusion plus a universal restriction can clash', () => {
    expect(conceptSatisfiable('Human and all hasParent not Human', tbox)).toBe(false)
    expect(conceptSatisfiable('Human and all hasParent Human', tbox)).toBe(true)
    expect(conceptSubsumes('Human', 'Mother', tbox)).toBe(true)
    expect(conceptSubsumes('Mother', 'Human', tbox)).toBe(false)
  })

  it('classifies a small ontology into its hierarchy', () => {
    expect(classifyConcepts(['Human', 'Man', 'Woman', 'Mother'], tbox)).toEqual({
      Human: [],
      Man: ['Human'],
      Woman: ['Human'],
      Mother: ['Woman'],
    })
  })
})

describe('soundness against finite models', () => {
  // A random interpretation of two names and two roles on three elements; a concept true somewhere in it is satisfiable.
  const concept: fc.Arbitrary<Concept> = fc.letrec<{ c: Concept }>((tie) => ({
    c: fc.oneof(
      { depthSize: 'small', maxDepth: 3 },
      fc.constantFrom<Concept>({ k: 'name', n: 'A' }, { k: 'name', n: 'B' }, { k: 'top' }),
      tie('c').map((c): Concept => ({ k: 'not', c })),
      fc.tuple(tie('c'), tie('c')).map(([a, b]): Concept => ({ k: 'and', a, b })),
      fc.tuple(tie('c'), tie('c')).map(([a, b]): Concept => ({ k: 'or', a, b })),
      fc.tuple(fc.constantFrom('R', 'S'), tie('c')).map(([r, c]): Concept => ({ k: 'some', r, c })),
      fc.tuple(fc.constantFrom('R', 'S'), tie('c')).map(([r, c]): Concept => ({ k: 'all', r, c })),
    ),
  })).c
  const model = fc.record({
    A: fc.array(fc.boolean(), { minLength: 3, maxLength: 3 }),
    B: fc.array(fc.boolean(), { minLength: 3, maxLength: 3 }),
    R: fc.array(fc.array(fc.boolean(), { minLength: 3, maxLength: 3 }), { minLength: 3, maxLength: 3 }),
    S: fc.array(fc.array(fc.boolean(), { minLength: 3, maxLength: 3 }), { minLength: 3, maxLength: 3 }),
  })
  type Model = { A: boolean[]; B: boolean[]; R: boolean[][]; S: boolean[][] }
  const holds = (m: Model, c: Concept, x: number): boolean => {
    switch (c.k) {
      case 'top':
        return true
      case 'bottom':
        return false
      case 'name':
        return (c.n === 'A' ? m.A : m.B)[x] === true
      case 'not':
        return !holds(m, c.c, x)
      case 'and':
        return holds(m, c.a, x) && holds(m, c.b, x)
      case 'or':
        return holds(m, c.a, x) || holds(m, c.b, x)
      case 'some':
        return [0, 1, 2].some((y) => (c.r === 'R' ? m.R : m.S)[x]?.[y] === true && holds(m, c.c, y))
      case 'all':
        return [0, 1, 2].every((y) => (c.r === 'R' ? m.R : m.S)[x]?.[y] !== true || holds(m, c.c, y))
    }
  }

  it('a concept with a model is judged satisfiable, and nnf preserves meaning in every model', () => {
    fc.assert(
      fc.property(concept, model, (c, m) => {
        for (const x of [0, 1, 2]) expect(holds(m as Model, nnf(c), x)).toBe(holds(m as Model, c, x))
        if ([0, 1, 2].some((x) => holds(m as Model, c, x))) expect(conceptSatisfiable(c)).toBe(true)
      }),
      { numRuns: 300 },
    )
  })

  it('subsumption is reflexive and a concept is subsumed by top', () => {
    fc.assert(
      fc.property(concept, (c) => {
        const t = conceptText(c)
        expect(conceptSubsumes(t, t)).toBe(true)
        expect(conceptSubsumes('top', t)).toBe(true)
      }),
      { numRuns: 100 },
    )
  })
})
