// The mixture whose groups have a mean and a precision of their own: the compiler fuses each pair into one normal-gamma variable
// (the atom runs closed-form messages), EP recovers the centres and precisions of the exact posterior (every assignment of the rows to
// the groups enumerated) and the probability that each row is in the second group, evidence included.
import { describe, expect, it } from 'vitest'
import { toFactorGraph } from '../../model/index.ts'
import { epAtom } from '../ep-atom-list.ts'
import { atomBound, atomReference, atomRun, verifyAtom } from '../ep-atoms.ts'

const atom = epAtom('gate-normal-gamma-pairs')

describe('the mixture of conjugate pairs', () => {
  const problem = atom.problem(atom.defaults)
  const n = atom.defaults['n'] as number

  it('compiles to a gate per row and direction over gaussian-gamma-precision branches; fused it is four normal-gamma variables', () => {
    const plain = toFactorGraph(atomBound(problem))
    expect(plain.issues).toEqual([])
    expect(plain.graph?.factors.filter((f) => f.operator === 'gate')).toHaveLength(2 * n)
    const fused = toFactorGraph(atomBound(problem), { conjugatePairs: true })
    expect(fused.graph?.variables.filter((v) => v.family.kind === 'normal-gamma')).toHaveLength(4)
    expect(fused.graph?.factors.some((f) => f.operator === 'gaussian-gamma-precision')).toBe(false)
  })

  it('is held to the enumeration over its seeds: centres and precisions within 0.3 reference sds, the groups of the rows within 0.3', () => {
    for (const seed of [1, 2, 3]) {
      const v = verifyAtom(atom, { seed })
      expect(v.run.status).toBe('ran')
      expect(v.check?.rows.filter((r) => !r.pass)).toEqual([])
      const rows = v.check?.rows ?? []
      const parameters = rows.filter((r) => !r.name.startsWith('which['))
      expect(parameters).toHaveLength(8)
      expect(Math.max(...parameters.map((r) => Math.abs(r.ep.mean - r.refMean) / r.refSd))).toBeLessThan(0.3)
      expect(Math.min(...parameters.map((r) => r.ep.sd / r.refSd))).toBeGreaterThan(0.65)
      expect(
        Math.max(...rows.filter((r) => r.name.startsWith('which[')).map((r) => Math.abs(r.ep.mean - r.refMean))),
      ).toBeLessThan(0.1)
    }
  }, 120_000)

  it('has an EP log evidence within a few hundredths of a nat of the exact one', () => {
    const run = atomRun(problem, atom.ep ?? {}, { light: true })
    expect(run.status).toBe('ran')
    const reference = atomReference(atom, problem)
    const gap = Math.abs(
      (run.status === 'ran' ? (run.evidence ?? Number.NaN) : Number.NaN) - (reference.logEvidence ?? 0),
    )
    expect(gap).toBeLessThan(0.1)
  }, 60_000)

  it('moves with the knobs: groups further apart are told apart with more certainty', () => {
    const sure = (gap: number): number => {
      const v = verifyAtom(atom, { gap, n: 10 })
      const rows = (v.check?.rows ?? []).filter((r) => r.name.startsWith('which['))
      return rows.reduce((s, r) => s + Math.abs(r.ep.mean - 0.5), 0) / rows.length
    }
    expect(sure(3)).toBeGreaterThan(sure(0.8))
  }, 60_000)
})
