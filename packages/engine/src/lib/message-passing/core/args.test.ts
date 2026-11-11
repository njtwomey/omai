import { describe, expect, it } from 'vitest'
import {
  allProper,
  etaOf,
  FLAT_CAVITY_PRECISION,
  gaussianMessage,
  gaussianOf,
  gaussianSiteMessage,
  ImproperMessageError,
  logGaussianIntegral,
  massOf,
  messageArg,
  numberParam,
  rawSum,
  scalarOf,
  siteMessage,
  valueArg,
  vectorOf,
  vectorParam,
} from './args.ts'
import { gaussian, gaussianEta } from './families.ts'

describe('how operators read their arguments', () => {
  it('reads messages and observed values', () => {
    const m = messageArg(gaussian, gaussianEta(1, 2))
    expect(etaOf(m)).toEqual([0.5, 0.5])
    expect(scalarOf(valueArg(gaussian, 3))).toBe(3)
    expect(vectorOf(valueArg(gaussian, [1, 2]))).toEqual([1, 2])
    expect(() => etaOf(valueArg(gaussian, 3))).toThrow('expected a message')
    expect(() => scalarOf(m)).toThrow('expected an observed number')
    expect(() => vectorOf(valueArg(gaussian, 3))).toThrow('expected an observed vector')
  })

  it('takes the moments of a Gaussian argument, an observed one with variance zero', () => {
    expect(gaussianOf(messageArg(gaussian, gaussianEta(1, 2)))).toEqual({ mean: 1, variance: 2 })
    expect(gaussianOf(valueArg(gaussian, 4))).toEqual({ mean: 4, variance: 0 })
    expect(() => gaussianOf(messageArg(gaussian, [0, -1]))).toThrow(ImproperMessageError)
  })

  it('says whether every message has a density, and the mass of one', () => {
    const proper = messageArg(gaussian, gaussianEta(0, 1))
    const flat = messageArg(gaussian, [0, 0])
    expect(allProper([proper, valueArg(gaussian, 1)])).toBe(true)
    expect(allProper([proper, flat])).toBe(false)
    expect(massOf(valueArg(gaussian, 1))).toBe(0)
    expect(massOf(proper)).toBeCloseTo(0.5 * Math.log(2 * Math.PI), 12)
  })

  it('reads parameters with a fallback, and says which one is wrong', () => {
    expect(numberParam({ a: 2 }, 'a')).toBe(2)
    expect(numberParam({}, 'a', 5)).toBe(5)
    expect(() => numberParam({}, 'a')).toThrow('the parameter a must be a number')
    expect(vectorParam({ v: [1, 2] }, 'v')).toEqual([1, 2])
    expect(() => vectorParam({ v: 'x' }, 'v')).toThrow('the parameter v must be an array of numbers')
  })

  it('forms Gaussian messages and the site that turns a cavity into a tilted Gaussian', () => {
    expect(gaussianMessage(1, 0.5)).toEqual([2, 2])
    expect(() => gaussianMessage(0, -1)).toThrow(ImproperMessageError)
    // cavity N(0, 1) times site = N(1, 0.5): natural parameters [2, 2] - [0, 1]
    expect(gaussianSiteMessage([0, 1], 1, 0.5)).toEqual([2, 1])
    expect(() => gaussianSiteMessage([0, 1], 0, 0)).toThrow(ImproperMessageError)
  })

  it('turns moments into the site of a family: the projection quotiented by the cavity', () => {
    // the moments of a Gaussian are E x and E x^2: N(1, 0.5) over a cavity N(0, 1)
    expect(siteMessage(gaussian, [0, 1], [1, 1.5])).toEqual(gaussianSiteMessage([0, 1], 1, 0.5))
  })

  it('adds messages without renormalising', () => {
    expect(rawSum([1, 2], [3, 4])).toEqual([4, 6])
  })

  it('integrates an unnormalised Gaussian in natural parameters', () => {
    // exp(h z - p z^2 / 2) integrates to sqrt(2 pi / p) exp(h^2 / (2 p))
    expect(logGaussianIntegral([1], [2], 1)).toBeCloseTo(0.5 * Math.log(Math.PI) + 0.25, 12)
    expect(logGaussianIntegral([1], [-2], 1)).toBeNaN()
  })

  it('names the precision a flat cavity is shown as', () => {
    expect(FLAT_CAVITY_PRECISION).toBe(1e-6)
  })
})
