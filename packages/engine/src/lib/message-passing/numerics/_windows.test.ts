import { describe, expect, it } from 'vitest'
import { centredMoments } from './_moments.ts'
import { gammaLogWindow, tiltGrid, windowProfiles } from './_windows.ts'

describe('tiltGrid', () => {
  it('finds a mass much narrower than its window: a spread of 0.01 inside the 40 standard deviations of a gamma cavity', () => {
    const window = gammaLogWindow(2, 1, windowProfiles.logTilt)
    expect(window.hi - window.lo).toBeGreaterThan(30)
    // A Gaussian bump of sd 0.01 at 0.5 in the log of the variable: one grid of 241 nodes over the window has a node every 0.15.
    const centre = 0.5
    const g = tiltGrid(window, 241, (t) => -0.5 * ((t - centre) / 0.01) ** 2)
    const m = centredMoments(g.weights, g.total, g.t)
    expect(m.mean).toBeCloseTo(centre, 6)
    expect(Math.sqrt(m.variance) / 0.01).toBeCloseTo(1, 4)
    expect(g.top + Math.log(g.total * g.step)).toBeCloseTo(Math.log(0.01 * Math.sqrt(2 * Math.PI)), 6)
  })

  it('keeps the whole window when the mass fills it', () => {
    const window = { lo: -1, hi: 1 }
    const g = tiltGrid(window, 101, () => 0)
    expect(g.t[0]).toBe(-1)
    expect(g.t[100]).toBe(1)
    expect(g.total).toBe(101)
  })

  it('has no top when nothing on the grid has weight, for the caller to report', () => {
    const g = tiltGrid({ lo: 0, hi: 1 }, 11, () => Number.NEGATIVE_INFINITY)
    expect(Number.isFinite(g.top)).toBe(false)
  })
})

describe('windowProfiles', () => {
  it('has one profile for every zoomed tilt, and the narrower ones of the panelled grid', () => {
    expect(windowProfiles.logTilt).toEqual({ below: 20, above: 20, clamp: 60 })
    expect(windowProfiles.gridGamma.below + windowProfiles.gridGamma.above).toBeLessThan(40)
  })
})
