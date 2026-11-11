// The route of a bound model through expectation propagation: plain, inside a grid over hyperparameters, or refused.
import { describe, expect, it } from 'vitest'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import { benchmarkModel } from '../models/suite.ts'
import { epGridMethod } from './ep-grid.ts'
import { epRoute } from './ep-route.ts'
import { analysisOf } from './model-input.ts'
import { epSpecMethod } from './mp.ts'

describe('epRoute', () => {
  it('runs a conjugate model as it is, with the method of plain EP and its verdict', () => {
    const bound = defineModel('route-plain', (b) => {
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), b.data('y', { dims: ['row'] }))
    }).bind({ y: [1, 2] })
    const route = epRoute(bound)
    expect(route.via).toBe('ep')
    expect(route.verdict.applies).toBe(true)
    expect(route.method?.name).toBe('ep')
  })

  it('puts a model with a random length scale inside a grid, the verdict being the grid method', () => {
    const bound = benchmarkModel('gaussian-process-regression').bound()
    expect(epSpecMethod(bound).check?.(analysisOf(bound)).applies).toBe(false)
    const route = epRoute(bound)
    expect(route.via).toBe('ep-grid')
    expect(route.method?.name).toBe('ep-grid')
    expect(route.verdict).toEqual(epGridMethod(bound).check?.(analysisOf(bound)))
  })

  it('refuses a model neither route takes, with the reason of plain EP and no method', () => {
    const bound = defineModel('route-refused', (b) => {
      b.observe('y', d.normal(0, b.latent('s', d.gamma(2, 2))), b.data('y', { dims: ['row'] }))
    }).bind({ y: [1, 2] })
    const route = epRoute(bound)
    expect(route.via).toBe('none')
    expect(route.method).toBeUndefined()
    expect(route.verdict).toEqual(epSpecMethod(bound).check?.(analysisOf(bound)))
    expect(route.verdict.applies).toBe(false)
  })
})
