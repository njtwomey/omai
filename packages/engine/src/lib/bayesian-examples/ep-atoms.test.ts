// The registry test: every atom is run (or refused) and its outcome compared with what it declares, so coverage only goes up.
import { describe, expect, it } from 'vitest'
import { epAtoms } from './ep-atom-list.ts'
import { verifyAtom } from './ep-atoms.ts'

describe('the EP atoms', () => {
  it('have distinct names and say what they exercise', () => {
    expect(new Set(epAtoms.map((a) => a.name)).size).toBe(epAtoms.length)
    for (const a of epAtoms) expect(a.factors.length).toBeGreaterThan(0)
  })

  describe.each(epAtoms.filter((a) => a.expected === 'runs'))('$name runs under EP and passes its check', (atom) => {
    it(atom.title, () => {
      const v = verifyAtom(atom)
      expect(v.run.status).toBe('ran')
      expect(v.check?.rows.filter((r) => !r.pass)).toEqual([])
      expect(v.check?.pass).toBe(true)
    })
  })

  describe.each(epAtoms.filter((a) => a.expected === 'planned'))('$name is refused with a reason', (atom) => {
    it(atom.title, () => {
      // Planned means EP does not run today: the compiler refuses (the verdict) or the run breaks (a gap); either way with a reason,
      // and when a rule lands this fails until the atom is flipped to `runs`.
      const run = verifyAtom(atom).run
      expect(run.status).not.toBe('ran')
      expect(run.status === 'ran' ? '' : run.reason.length).toBeGreaterThan(10)
      expect(atom.plan?.length).toBeGreaterThan(0)
    })
  })
})
