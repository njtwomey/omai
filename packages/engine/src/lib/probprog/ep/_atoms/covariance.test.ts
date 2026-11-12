// The covariance atoms: the closed-form pieces of their references against direct computations, and each atom against its reference.
import { describe, expect, it } from 'vitest'
import { epAtoms } from '../ep-atom-list.ts'
import { verifyAtom } from '../ep-atoms.ts'

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
      const table = v.check?.rows.map(
        (r) => `${r.name}: z ${r.meanZ.toFixed(3)}, sd ratio ${r.sdRatio.toFixed(3)}, ks ${r.ks.toFixed(3)}`,
      )
      expect(
        v.check?.rows.filter((r) => !r.pass),
        String(table),
      ).toEqual([])
    }, 120_000)
  })
})

describe('correlated random effects with few groups and the population mean unknown', () => {
  const learned = covariance.find((a) => a.name === 'random-intercepts-slopes-learned') as (typeof covariance)[number]

  // The group vectors, the shared mean and the matrix are one factor (`mvn-wishart-group`), so there is no loop through the mean and the matrix:
  // with a factor per vector the mean's spread was 0.77 to 0.89 of the exact at three or four groups and the matrix 1.05 to 1.25 times too wide.
  it.each([
    { seed: 10, n: 3, df: 2.2 },
    { seed: 11, n: 3, df: 2.2 },
    { seed: 10, n: 4, df: 2.2 },
    { seed: 2, n: 3, trueRho: 0.9, trueSd1: 0.5, trueSd2: 0.4 },
  ])(
    'carries the mean, the matrix and every group to the exact spread: %j',
    (knobs) => {
      const v = verifyAtom(learned, knobs)
      expect(v.run.status).toBe('ran')
      if (v.run.status !== 'ran') return
      expect(v.run.sweeps).toBeLessThanOrEqual(5)
      const rows = v.check?.rows ?? []
      const ratio = (prefix: string): number[] => rows.filter((r) => r.name.startsWith(prefix)).map((r) => r.sdRatio)
      // the reference grid's own spread for the mean is a few percent high at three groups (a graded integration of the same model gives 0.6887 where
      // the grid says 0.7068 and EP 0.6877, seed 10 of three groups), so the mean's ratio is held to 0.93 and the others to 3%
      for (const r of ratio('mu')) expect(r).toBeGreaterThan(0.93)
      for (const r of ratio('mu')) expect(r).toBeLessThan(1.05)
      for (const r of [...ratio('Lambda'), ...ratio('theta')]) {
        expect(r).toBeGreaterThan(0.97)
        expect(r).toBeLessThan(1.03)
      }
      expect(Math.max(...rows.map((r) => r.meanZ))).toBeLessThan(0.05)
      // the evidence is the exact one (the grid of the matrix with the mean and the effects integrated out, every constant of the model in it)
      const exact = v.reference.logEvidence as number
      expect(Number.isFinite(exact)).toBe(true)
      expect(Math.abs((v.run.evidence as number) - exact)).toBeLessThan(0.01)
    },
    120_000,
  )
})
