import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  candidateElimination,
  classifyByVersionSpace,
  coversInstance,
  findS,
  type Hypothesis,
  type LabelledExample,
  moreGeneral,
  versionSpaceMembers,
} from './index.ts'

const DOMAINS = [
  ['a', 'b'],
  ['x', 'y', 'z'],
  ['u', 'v'],
]

const examples = fc.array(
  fc
    .tuple(fc.constantFrom('a', 'b'), fc.constantFrom('x', 'y', 'z'), fc.constantFrom('u', 'v'), fc.boolean())
    .map(([p, q, r, positive]): LabelledExample => ({ values: [p, q, r], positive })),
  { maxLength: 6 },
)

function allHypotheses(): Hypothesis[] {
  const out: Hypothesis[] = []
  for (const a of ['?', ...(DOMAINS[0] ?? [])])
    for (const b of ['?', ...(DOMAINS[1] ?? [])]) for (const c of ['?', ...(DOMAINS[2] ?? [])]) out.push([a, b, c])
  return out
}

describe('version space properties', () => {
  it('candidate elimination keeps exactly the consistent hypotheses', () => {
    fc.assert(
      fc.property(examples, (ex) => {
        const consistent = allHypotheses().filter((h) => ex.every((e) => coversInstance(h, e.values) === e.positive))
        const vs = candidateElimination(ex, DOMAINS)
        const members = versionSpaceMembers(vs).map((h) => JSON.stringify(h))
        expect(members.sort()).toEqual(consistent.map((h) => JSON.stringify(h)).sort())
        for (const s of vs.specific) for (const g of vs.general) expect(moreGeneral(g, s)).toBe(true)
      }),
      { numRuns: 300 },
    )
  })

  it('Find-S covers every positive and agrees with unanimous votes', () => {
    fc.assert(
      fc.property(examples, (ex) => {
        const h = findS(ex)
        for (const e of ex) if (e.positive) expect(coversInstance(h, e.values)).toBe(true)
        const vs = candidateElimination(ex, DOMAINS)
        if (vs.collapsed) return
        for (const e of ex) {
          expect(classifyByVersionSpace(vs, e.values).label).toBe(e.positive ? 'positive' : 'negative')
        }
      }),
      { numRuns: 200 },
    )
  })
})
