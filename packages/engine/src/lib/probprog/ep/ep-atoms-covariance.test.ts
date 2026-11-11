// The covariance atoms: the closed-form pieces of their references against direct computations, and each atom against its reference.
import { describe, expect, it } from 'vitest'
import { epAtoms } from './ep-atom-list.ts'
import { verifyAtom } from './ep-atoms.ts'

const covariance = epAtoms.filter((a) => a.family === 'covariance')

describe('the covariance atoms', () => {
  it('are six, three basic and three learned', () => {
    expect(covariance.map((a) => a.name)).toEqual([
      'mvn-wishart',
      'mvn-wishart-learned',
      'bivariate-correlation',
      'bivariate-correlation-learned',
      'random-intercepts-slopes',
      'random-intercepts-slopes-learned',
    ])
  })

  describe.each(covariance)('$name', (atom) => {
    it('runs under EP and agrees with its reference', () => {
      const v = verifyAtom(atom)
      expect(v.run.status).toBe('ran')
      const table = v.check?.rows.map((r) => `${r.name}: z ${r.meanZ.toFixed(3)}, sd ratio ${r.sdRatio.toFixed(3)}, ks ${r.ks.toFixed(3)}`)
      expect(v.check?.rows.filter((r) => !r.pass), String(table)).toEqual([])
    }, 120_000)
  })
})
