import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { evaluateDatalog, groundProgram, isStableModel, normalProgram, stableModels } from './index.ts'

/** Random facts over e/2 and n/1 with constants 0..4, plus rules drawn from a pool (some recursive, some negated). */
const RULES = [
  'p(X, Y) :- e(X, Y).',
  'p(X, Z) :- p(X, Y), e(Y, Z).',
  'p(X, Z) :- p(X, Y), p(Y, Z).',
  'q(X) :- n(X), not r(X).',
  'r(X) :- e(X, X).',
  'r(Y) :- e(_, Y), n(Y).',
  's(X, Y) :- p(X, Y), not p(Y, X).',
  't(X) :- n(X), n(Y), X < Y.',
  'u(X) :- q(X), t(X).',
  'v(X, Y) :- e(X, Y), X \\= Y, not s(X, Y).',
]

const program = fc
  .record({
    edges: fc.array(fc.tuple(fc.integer({ min: 0, max: 4 }), fc.integer({ min: 0, max: 4 })), { maxLength: 9 }),
    nodes: fc.array(fc.integer({ min: 0, max: 4 }), { maxLength: 4 }),
    rules: fc.subarray(RULES, { minLength: 1 }),
  })
  .map(
    ({ edges, nodes, rules }) =>
      `${edges.map(([a, b]) => `e(${a}, ${b}).`).join(' ')} ${nodes.map((n) => `n(${n}).`).join(' ')} ${rules.join(' ')}`,
  )

const sortedModel = (src: string, method: 'naive' | 'semi-naive'): string[] => {
  const r = evaluateDatalog(src, { method })
  return [...r.model.values()].flatMap((fs) => fs.map((f) => JSON.stringify(f))).sort()
}

describe('Datalog properties', () => {
  it('semi-naive evaluation reaches the naive fixpoint in the same rounds', () => {
    fc.assert(
      fc.property(program, (src) => {
        expect(sortedModel(src, 'semi-naive')).toEqual(sortedModel(src, 'naive'))
        const a = evaluateDatalog(src, { method: 'naive' }).history
        const b = evaluateDatalog(src, { method: 'semi-naive' }).history
        expect(b).toEqual(a)
      }),
      { numRuns: 150 },
    )
  })

  it('the perfect model of a stratified program is its unique answer set', () => {
    fc.assert(
      fc.property(program, (src) => {
        const r = evaluateDatalog(src)
        const model = [...r.model.values()].flatMap((fs) => fs.map((f) => JSON.stringify(f)))
        const asp = stableModels(src, { maxGuessAtoms: 24 })
        expect(asp.models.length).toBe(1)
        expect(asp.models[0]?.length).toBe(model.length)
      }),
      { numRuns: 60 },
    )
  })
})

/** Small ground normal programs over atoms a..d. */
const groundRules = fc.array(
  fc.record({
    head: fc.constantFrom('a', 'b', 'c', 'd', ''),
    pos: fc.subarray(['a', 'b', 'c', 'd'], { maxLength: 2 }),
    neg: fc.subarray(['a', 'b', 'c', 'd'], { maxLength: 2 }),
  }),
  { minLength: 1, maxLength: 6 },
)

describe('answer set properties', () => {
  it('guess and check finds exactly the sets that pass the stability check', () => {
    fc.assert(
      fc.property(groundRules, (rules) => {
        const text = rules
          .map((r) => {
            const body = [...r.pos, ...r.neg.map((x) => `not ${x}`)].join(', ')
            if (r.head === '') return body === '' ? '' : `:- ${body}.`
            return body === '' ? `${r.head}.` : `${r.head} :- ${body}.`
          })
          .join(' ')
        const p = normalProgram(text)
        const found = stableModels(p).models.map((m) => m.join(','))
        const atoms = ['a', 'b', 'c', 'd']
        const brute: string[] = []
        for (let m = 0; m < 16; m++) {
          const set = atoms.filter((_, i) => (m >> i) & 1)
          if (isStableModel(p, set)) brute.push(set.join(','))
        }
        expect(found.sort()).toEqual(brute.sort())
        expect(groundProgram(p).length).toBeLessThanOrEqual(rules.length)
      }),
      { numRuns: 300 },
    )
  })
})
