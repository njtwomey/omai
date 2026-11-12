// The gate between a Poisson and a negative binomial (a switch per observation, a shared log-linear rate, a Beta prior on the
// switch, a learned dispersion): it compiles to one engine gate per observation, EP recovers the simulated values and says
// which observations are from the second branch, against the grid posterior with the switches summed out, evidence included.
import { describe, expect, it } from 'vitest'
import { toFactorGraph } from '../../model/index.ts'
import { epAtom } from '../ep-atom-list.ts'
import { atomBound, atomReference, atomRun, verifyAtom } from '../ep-atoms.ts'

const atom = epAtom('gate-poisson-negative-binomial')

describe('the gate between a Poisson and a negative binomial', () => {
  const problem = atom.problem(atom.defaults)
  const n = atom.defaults['n'] as number

  it('compiles to one gate per observation, with nothing refused', () => {
    const compiled = toFactorGraph(atomBound(problem))
    expect(compiled.issues).toEqual([])
    expect(compiled.graph?.factors.filter((f) => f.operator === 'gate')).toHaveLength(n)
  })

  it('is held to its reference: the parameters and the probability of each switch', () => {
    const v = verifyAtom(atom)
    expect(v.run.status).toBe('ran')
    expect(v.check?.rows.filter((r) => !r.pass)).toEqual([])
    const gaps = (v.check?.rows ?? [])
      .filter((r) => r.name.startsWith('z['))
      .map((r) => Math.abs(r.ep.mean - r.refMean))
    expect(gaps).toHaveLength(n)
    expect(Math.max(...gaps)).toBeLessThan(0.1)
  }, 60_000)

  it('recovers the simulated values, and gives the second-branch observations the higher probability', () => {
    const v = verifyAtom(atom)
    const rows = v.check?.rows ?? []
    for (const name of ['w', 'b0', 'b1', 'dispersion']) {
      const row = rows.find((r) => r.name === name)
      const truth = problem.quantities.find((q) => q.name === name)?.truth ?? Number.NaN
      // the truth is inside the reference's own interval of ±2 sds
      expect(Math.abs(truth - (row?.refMean ?? Number.NaN))).toBeLessThan(2 * (row?.refSd ?? 0))
    }
    // the observations whose count is far above the rate are the second branch: the switch probabilities rank the truth
    const probs = rows.filter((r) => r.name.startsWith('z[')).map((r) => r.ep.mean)
    expect(Math.max(...probs)).toBeGreaterThan(0.9)
    expect(Math.min(...probs)).toBeLessThan(0.15)
  }, 60_000)

  it('has an EP log evidence within a nat of the grid evidence (0.07 to 0.5 measured over five seeds)', () => {
    const run = atomRun(problem, atom.ep ?? {}, { light: true })
    expect(run.status).toBe('ran')
    const reference = atomReference(atom, problem)
    expect(
      Math.abs((run.status === 'ran' ? (run.evidence ?? Number.NaN) : Number.NaN) - (reference.logEvidence ?? 0)),
    ).toBeLessThan(1)
  }, 60_000)

  it('moves with the knobs: a larger true probability of the second branch gives a larger w', () => {
    const w = (trueShare: number): number => {
      const v = verifyAtom(atom, { trueShare, n: 24 })
      return v.run.status === 'ran' ? (v.run.beliefs['w']?.mean ?? Number.NaN) : Number.NaN
    }
    expect(w(0.5)).toBeGreaterThan(w(0.1))
  }, 120_000)
})
