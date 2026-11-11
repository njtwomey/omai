// The registry test: every atom is run and its outcome compared with what it declares (it runs unless it says `planned`).
import { describe, expect, it } from 'vitest'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import { epAtoms } from './ep-atom-list.ts'
import { defineAtom, verifyAtom } from './ep-atoms.ts'
import { gaussianAtom } from './ep-atoms-conjugate.ts'

describe('the EP atoms', () => {
  it('have distinct names and say what they exercise', () => {
    expect(new Set(epAtoms.map((a) => a.name)).size).toBe(epAtoms.length)
    for (const a of epAtoms) expect(a.factors.length).toBeGreaterThan(0)
  })

  describe.each(epAtoms)('$name runs under EP and passes its check', (atom) => {
    it(atom.title, () => {
      const v = verifyAtom(atom)
      expect(v.run.status).toBe('ran')
      expect(v.check?.rows.filter((r) => !r.pass)).toEqual([])
      expect(v.check?.pass).toBe(true)
    }, 60_000)
  })

  it('has no atom planned today, and a planned atom needs its plan', () => {
    expect(epAtoms.filter((a) => a.expected === 'planned').map((a) => a.name)).toEqual([])
    expect(() => defineAtom({ ...gaussianAtom, expected: 'planned' })).toThrow(/says where its rule is planned/)
  })

  it('a planned atom is refused with a reason: a model EP does not take (a random scale of a Gaussian) is not run', () => {
    // The escape hatch for an atom the library refuses on purpose: built here from a model the compiler refuses, since none of the
    // suite's atoms is planned. It is refused with the library's reason, not run and not thrown on.
    const planned = defineAtom({
      ...gaussianAtom,
      name: 'planned-synthetic',
      expected: 'planned',
      plan: 'a random scale on a Gaussian is not in the precision idiom here',
      problem: (k) => ({
        ...gaussianAtom.problem(k),
        definition: defineModel('planned-synthetic', (b) => {
          b.observe('y', d.normal(0, b.latent('s', d.gamma(2, 2))), b.data('y', { dims: ['row'] }))
        }),
        data: { y: [1, 2] },
      }),
    })
    const run = verifyAtom(planned).run
    expect(run.status).toBe('refused')
    expect(run.status === 'refused' ? run.reason.length : 0).toBeGreaterThan(10)
  })
})
