import { describe, expect, it } from 'vitest'
import {
  checkNumber,
  clampNumber,
  formatFieldValue,
  fromFraction,
  numberLadder,
  numberStep,
  parseNumber,
  stepNumber,
  toFraction,
} from './number.ts'

describe('parsing and validation', () => {
  it('parses decimals, exponents and a typographic minus', () => {
    expect(parseNumber(' 1.5 ')).toBe(1.5)
    expect(parseNumber('−2')).toBe(-2)
    expect(parseNumber('3e-4')).toBe(3e-4)
    expect(parseNumber('.5')).toBe(0.5)
    expect(parseNumber('1.2.3')).toBeNull()
    expect(parseNumber('')).toBeNull()
  })

  it('refuses drafts that break the type or a bound', () => {
    expect(checkNumber({ integer: true }, '2.5')).toEqual({ error: 'must be a whole number' })
    expect(checkNumber({ min: 0 }, '-1')).toEqual({ error: 'must be ≥ 0' })
    expect(checkNumber({ max: 1 }, '2')).toEqual({ error: 'must be ≤ 1' })
    expect(checkNumber({ log: true }, '0')).toEqual({ error: 'must be > 0' })
    expect(checkNumber({ min: 0, max: 1 }, '0.25')).toEqual({ value: 0.25 })
  })

  it('clamps and rounds programmatic values', () => {
    expect(clampNumber({ min: 0, max: 10, integer: true }, 3.6)).toBe(4)
    expect(clampNumber({ min: 0, max: 10 }, 12)).toBe(10)
  })
})

describe('the ladder of 11 values', () => {
  it('spaces values evenly between linear bounds, ends exact', () => {
    expect(numberLadder({ min: 0, max: 1 })).toEqual([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    expect(numberLadder({ min: -5, max: 5 })).toHaveLength(11)
  })

  it('spaces values geometrically on a log scale, rounded to two figures', () => {
    const ladder = numberLadder({ min: 1e-4, max: 1, log: true })
    expect(ladder).toHaveLength(11)
    expect(ladder[0]).toBe(1e-4)
    expect(ladder[10]).toBe(1)
    expect(ladder[5]).toBe(0.01)
    for (let i = 1; i < ladder.length; i++) expect(ladder[i] as number).toBeGreaterThan(ladder[i - 1] as number)
  })

  it('gives whole numbers without repeats for integers', () => {
    expect(numberLadder({ min: 1, max: 5, integer: true })).toEqual([1, 2, 3, 4, 5])
    expect(numberLadder({ min: 0, max: 100, integer: true })).toEqual([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
  })

  it('is empty without a range', () => {
    expect(numberLadder({ min: 0 })).toEqual([])
    expect(numberLadder({ min: 0, max: 1, log: true })).toEqual([])
  })
})

describe('stepping', () => {
  it('steps linearly by a round hundredth of the range, ten with shift', () => {
    expect(numberStep({ min: 0, max: 1 })).toBe(0.01)
    expect(numberStep({ min: 0, max: 30 })).toBe(0.2)
    expect(stepNumber({ min: 0, max: 1 }, 0.5, 1)).toBe(0.51)
    expect(stepNumber({ min: 0, max: 1 }, 0.5, -1, true)).toBe(0.4)
  })

  it('stops at the bounds', () => {
    expect(stepNumber({ min: 0, max: 1 }, 1, 1)).toBeNull()
    expect(stepNumber({ min: 0, max: 1, step: 0.3 }, 0.9, 1)).toBe(1)
  })

  it('steps along 1, 2, 5 on a log scale, a decade with shift', () => {
    const spec = { min: 1e-4, max: 1, log: true }
    expect(stepNumber(spec, 0.01, 1)).toBe(0.02)
    expect(stepNumber(spec, 0.02, 1)).toBe(0.05)
    expect(stepNumber(spec, 0.03, -1)).toBe(0.02)
    expect(stepNumber(spec, 0.01, 1, true)).toBe(0.1)
  })

  it('always moves an integer on a log scale', () => {
    expect(stepNumber({ min: 1, max: 1000, integer: true, log: true }, 1, 1)).toBe(2)
  })
})

describe('the slider', () => {
  it('maps values to positions and back on the scale', () => {
    expect(toFraction({ min: 0, max: 10 }, 2.5)).toBe(0.25)
    expect(toFraction({ min: 1, max: 100, log: true }, 10)).toBeCloseTo(0.5, 12)
    expect(fromFraction({ min: 0, max: 10 }, 0.25)).toBe(2.5)
    expect(fromFraction({ min: 1, max: 100, log: true }, 0.5)).toBe(10)
    expect(fromFraction({ min: 0, max: 10, integer: true }, 0.33)).toBe(3)
    expect(fromFraction({ min: 0.5, max: 2 }, 1)).toBe(2)
  })
})

describe('display', () => {
  it('shows integers in full and extremes in exponent form', () => {
    expect(formatFieldValue({ integer: true }, 12000000)).toBe('12000000')
    expect(formatFieldValue({}, 0.00025)).toBe('2.5e-4')
    expect(formatFieldValue({}, 1 / 3)).toBe('0.3333')
    expect(formatFieldValue({}, 250000)).toBe('2.5e5')
  })
})

describe('the ladder on a step grid', () => {
  it('rounds interior values to the step', () => {
    const ladder = numberLadder({ min: 0, max: 0.95, step: 0.05 })
    expect(ladder).toHaveLength(11)
    for (const v of ladder) expect(Math.abs(v / 0.05 - Math.round(v / 0.05))).toBeLessThan(1e-9)
    for (let i = 1; i < ladder.length; i++) expect(ladder[i] as number).toBeGreaterThan(ladder[i - 1] as number)
  })
})
