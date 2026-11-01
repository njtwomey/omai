import { describe, expect, it } from 'vitest'
import {
  nonlinearFilters,
  particleProposals,
  resamplers,
  simulateStateSpace,
  stateSpaceModels,
} from '../dynamics/filtering/index.ts'
import { completeOption } from '../learn/index.ts'
import { key } from '../random/index.ts'
import { nonlinearFilter, particleFilter, stateSpaceData } from './state-filters.ts'

const model = stateSpaceModels.make({ kind: 'pendulum' })
const { states, observations } = simulateStateSpace(model, 12, key(3))
const pend = completeOption(stateSpaceModels, { kind: 'pendulum' })
const data = stateSpaceData(observations, { states })

describe('nonlinearFilter', () => {
  it('fits every filter of the table and reports the error against the true states', () => {
    for (const name of nonlinearFilters.names) {
      const fitted = nonlinearFilter
        .with({ model: pend, filter: completeOption(nonlinearFilters, { kind: name }) })
        .fit(data)
      expect(fitted.params.steps).toBe(12)
      expect(fitted.params.filter).toBe(name)
      expect(fitted.params.means).toHaveLength(12)
      expect(fitted.params.covariances[0]).toHaveLength(4)
      expect(fitted.component(0)).toHaveLength(12)
      expect(fitted.error(states) ?? Number.NaN).toBeLessThan(1)
    }
  })

  it('steps live: fittedAt(t) is the filter after t observations, and a session logs the paths', () => {
    const est = nonlinearFilter.with({ model: pend, filter: completeOption(nonlinearFilters, { kind: 'unscented' }) })
    const s = est.session(data, {
      mode: 'eager',
      logs: { 'filter/mean': {}, 'filter/log-likelihood': {}, 'filter/sigma-points': {} } as never,
    })
    expect(s.t).toBe(12)
    expect(s.fittedAt(5).params.steps).toBe(5)
    expect(s.fittedAt(5).params.means).toEqual(est.fit(data).params.means.slice(0, 5))
  })

  it('takes the model from the data when it carries one, and uses the slot otherwise', () => {
    const own = { f: (x: never) => x, h: (x: never) => x, Q: [[0.1]], R: [[1]], x0: [0], P0: [[1]] }
    const fitted = nonlinearFilter.fit(stateSpaceData([1, 1.2, 0.8], { model: own as never }))
    expect(fitted.params.stateSize).toBe(1)
    expect(nonlinearFilter.fit(stateSpaceData([1, 1.2, 0.8])).params.stateSize).toBe(1)
  })
})

describe('particleFilter', () => {
  it('fits every proposal and resampling, with the ESS and the resampling steps', () => {
    for (const proposal of [
      completeOption(particleProposals, { kind: 'bootstrap', essThreshold: 0.8 }),
      completeOption(particleProposals, { kind: 'auxiliary' }),
    ]) {
      for (const kind of ['multinomial', 'systematic', 'stratified', 'residual']) {
        const fitted = particleFilter
          .with({ model: pend, proposal, resampling: completeOption(resamplers, { kind }), particles: 60 })
          .fit(data, { key: key(2) })
        expect(fitted.params.steps).toBe(12)
        expect(fitted.params.ess).toHaveLength(12)
        expect(Number.isFinite(fitted.params.logEvidence)).toBe(true)
        if ((proposal as { kind: string }).kind === 'auxiliary') expect(fitted.resamplingSteps()).toHaveLength(12)
      }
    }
  })

  it('steps live and can be read at any step', () => {
    const est = particleFilter.with({ particles: 40 })
    const walk = simulateStateSpace(stateSpaceModels.make({ kind: 'random-walk' }), 8, key(1))
    const s = est.session(stateSpaceData(walk.observations, { states: walk.states }), { mode: 'eager', key: key(4) })
    expect(s.fittedAt(3).params.steps).toBe(3)
    expect(s.fittedAt(8).error(walk.states) ?? Number.NaN).toBeLessThan(1.5)
  })
})
