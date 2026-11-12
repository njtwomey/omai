// The knobs an atom opens on are a typed record of the EP knobs: a misspelled one is refused where the atom is declared and where a
// run is asked for, instead of running on the defaults and reporting success.
import { describe, expect, it } from 'vitest'
import { defineAtom } from './_define.ts'
import { epAtom, epAtoms } from './ep-atom-list.ts'
import { atomRun } from './ep-atoms.ts'

const gaussianAtom = epAtom('gaussian')

describe('the knobs of an atom', () => {
  const problem = gaussianAtom.problem(gaussianAtom.defaults)

  it('a run with a misspelled knob throws and names it', () => {
    expect(() => atomRun(problem, { dampng: 0.5 })).toThrow(/unknown EP knob 'dampng'/)
    expect(atomRun(problem, { damping: 0.5 }).status).toBe('ran')
  })

  it('an atom declared with a misspelled knob throws', () => {
    // @ts-expect-error `dampng` is not a knob: the typed record refuses it before the check at run time does
    expect(() => defineAtom({ ...gaussianAtom, ep: { dampng: 0.5 } })).toThrow(/unknown EP knob 'dampng'/)
  })

  it('every atom of the suite opens on knobs that exist', () => {
    for (const a of epAtoms) expect(() => defineAtom({ ...a, ep: a.ep ?? {} }), a.name).not.toThrow()
  })
})
