// The composed atoms: each runs under EP and agrees with its reference within its declared tolerance.
import { describe, expect, it } from 'vitest'
import { epAtoms } from '../ep-atom-list.ts'
import { verifyAtom } from '../ep-atoms.ts'

const composed = epAtoms.filter((a) => a.family === 'composed')

describe('the composed atoms', () => {
  it('are one: a softmax over many classes with a coefficient for each of two groups', () => {
    expect(composed.map((a) => a.name)).toEqual(['softmax-two-groups'])
  })

  describe.each(composed)('$name', (atom) => {
    it('has eight classes by default, past the six of the product grid, and asks about the two coefficients, the mean and the precision', () => {
      expect(atom.defaults).toMatchObject({ classes: 8 })
      expect(atom.problem(atom.defaults).quantities.map((q) => q.name)).toEqual(['beta_1', 'beta_2', 'mu', 'tau'])
    })

    it('runs under EP, agrees with its grid and holds the true coefficients inside two and a half posterior sds', () => {
      const v = verifyAtom(atom)
      expect(v.run.status).toBe('ran')
      if (v.run.status !== 'ran') return
      const table = v.check?.rows.map(
        (r) => `${r.name}: z ${r.meanZ.toFixed(3)}, sd ratio ${r.sdRatio.toFixed(3)}, ks ${r.ks.toFixed(3)}`,
      )
      expect(
        v.check?.rows.filter((r) => !r.pass),
        String(table),
      ).toEqual([])
      for (const q of atom.problem(atom.defaults).quantities) {
        if (q.truth === undefined) continue
        const b = v.run.beliefs[q.name]
        expect(Math.abs((b?.mean ?? Number.NaN) - q.truth) / (b?.sd ?? 1), q.name).toBeLessThan(2.5)
      }
    }, 120_000)
  })
})
