// runEp on a bound model that is not an atom: the beliefs of every site, a misspelled knob refused, a model the library refuses
// reported with its reason, and the same run as the atom's own (`atomRun` is `runEp` with the atom's quantities).
import { describe, expect, it } from 'vitest'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { epAtom } from './ep-atom-list.ts'
import { atomBound, atomRun } from './ep-atoms.ts'
import { runEp } from './run.ts'

describe('runEp', () => {
  const mean = defineModel('mean', (m) => {
    const mu = m.latent('mu', d.normal(0, 2))
    m.observe('y', d.normal(mu, 1), m.data('y', { dims: ['i'], role: 'response' }))
  })

  it('reads the beliefs of every site of a model that no atom describes', () => {
    const run = runEp(mean.bind({ y: [1.2, 0.4, 1.9] }))
    expect(run.status).toBe('ran')
    if (run.status !== 'ran') return
    expect(Object.keys(run.beliefs)).toEqual(['mu'])
    // the conjugate posterior: precision 1 / 4 + 3, mean 3.5 / 3.25
    expect(run.beliefs['mu']?.mean).toBeCloseTo(3.5 / 3.25, 6)
    expect(run.beliefs['mu']?.proper).toBe(true)
    expect(run.history.length).toBeGreaterThan(1)
    expect(run.record().steps.length).toBeGreaterThan(0)
  })

  it('names the elements of a vector site', () => {
    const vector = defineModel('vector', (m) => {
      const w = m.latent('w', d.normal(0, 1), { dims: [m.range('k', { size: 2 })] })
      m.observe('y', d.normal(ops.sum(w), 1), m.data('y', { dims: [m.range('i')] }))
    })
    const run = runEp(vector.bind({ y: [0.5, 1.5] }), { light: true })
    expect(run.status === 'ran' ? Object.keys(run.beliefs) : run.status).toEqual(['w[0]', 'w[1]'])
  })

  it('refuses a misspelled knob instead of running on the defaults', () => {
    expect(() => runEp(mean.bind({ y: [1] }), { ep: { dampng: 0.5 } })).toThrow(/unknown EP knob 'dampng'/)
  })

  it('is what atomRun runs: the same beliefs and sweeps', () => {
    const atom = epAtom('gaussian')
    const problem = atom.problem(atom.defaults)
    const viaAtom = atomRun(problem)
    const direct = runEp(atomBound(problem), { quantities: problem.quantities })
    expect(viaAtom.status === 'ran' && direct.status === 'ran' ? [viaAtom.sweeps, viaAtom.beliefs] : []).toEqual(
      direct.status === 'ran' ? [direct.sweeps, direct.beliefs] : [],
    )
  })
})
