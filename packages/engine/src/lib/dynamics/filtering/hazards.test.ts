import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { defaults } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { bocpd, constantHazard } from './changepoint.ts'
import { hazards, hazardValue, segmentModels, segmentModelValue, segmentModelValueOf } from './changepoint-options.ts'
import { gammaHazard, minimumSegmentHazard, negativeBinomialHazard, weibullHazard } from './hazards.ts'

/** The segment-length pmf P(T = r) implied by a hazard: h(r) times the chance of having survived to r. */
function pmfOf(h: (r: number) => number, upTo: number): number[] {
  const out: number[] = []
  let alive = 1
  for (let r = 1; r <= upTo; r++) {
    out.push(alive * h(r))
    alive *= 1 - h(r)
  }
  return out
}

const lgam = (x: number): number => {
  // Lanczos (g = 7): an independent log-gamma for the reference densities below.
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgam(1 - x)
  const z = x - 1
  let a = c[0] ?? 0
  const t = z + 7.5
  for (let i = 1; i < 9; i++) a += (c[i] ?? 0) / (z + i)
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a)
}

describe('hazards from priors on the segment length', () => {
  it('Weibull: the implied length law is the discretised Weibull', () => {
    const [k, scale] = [2.5, 12]
    const pmf = pmfOf(weibullHazard(k, scale), 60)
    for (const [i, p] of pmf.entries()) {
      const r = i + 1
      expect(p).toBeCloseTo(Math.exp(-(((r - 1) / scale) ** k)) - Math.exp(-((r / scale) ** k)), 12)
    }
  })

  it('negative binomial: the implied length law is one plus a negative binomial', () => {
    const [s, p] = [3.5, 0.15]
    const pmf = pmfOf(negativeBinomialHazard(s, p), 80)
    for (const [i, got] of pmf.entries()) {
      const x = i // T = r = i + 1 means X = r - 1 = i failures
      const want = Math.exp(lgam(x + s) - lgam(s) - lgam(x + 1) + s * Math.log(p) + x * Math.log(1 - p))
      expect(got).toBeCloseTo(want, 10)
    }
  })

  it('gamma: the implied length law integrates the gamma density over each unit', () => {
    const [shape, rate] = [4, 0.5]
    const density = (x: number): number =>
      x <= 0 ? 0 : Math.exp(shape * Math.log(rate) - lgam(shape) + (shape - 1) * Math.log(x) - rate * x)
    const pmf = pmfOf(gammaHazard(shape, rate), 40)
    for (const [i, got] of pmf.entries()) {
      const [a, b] = [i, i + 1]
      const steps = 400
      let area = 0
      for (let j = 0; j < steps; j++) {
        const [x0, x1] = [a + ((b - a) * j) / steps, a + ((b - a) * (j + 1)) / steps]
        area += ((x1 - x0) / 6) * (density(x0) + 4 * density((x0 + x1) / 2) + density(x1))
      }
      expect(got).toBeCloseTo(area, 8)
    }
  })

  it('their memoryless members are the constant hazard', () => {
    for (const r of [1, 4, 17]) {
      expect(weibullHazard(1, 10)(r)).toBeCloseTo(1 - Math.exp(-0.1), 12)
      expect(gammaHazard(1, 0.1)(r)).toBeCloseTo(1 - Math.exp(-0.1), 10)
      expect(negativeBinomialHazard(1, 0.1)(r)).toBeCloseTo(0.1, 10)
      expect(constantHazard(10)(r)).toBeCloseTo(0.1, 12)
    }
  })

  it('a minimum segment forbids short segments and is constant beyond', () => {
    const h = minimumSegmentHazard(0.2, 3)
    expect([1, 2, 3, 4, 50].map(h)).toEqual([0, 0, 0.2, 0.2, 0.2])
    expect(() => minimumSegmentHazard(1, 3)).toThrow()
    expect(() => minimumSegmentHazard(0.1, 0)).toThrow()
    expect(() => weibullHazard(0, 1)).toThrow()
    expect(() => gammaHazard(1, -1)).toThrow()
    expect(() => negativeBinomialHazard(2, 1)).toThrow()
  })

  it('a hazard that fits the regularity of the segments explains the data better', () => {
    // Means alternate between 0 and 5 every 20 steps, with small noise.
    const noise = toFlat(normal(key(3), [100]))
    const y = noise.map((v, i) => 0.3 * v + (Math.floor(i / 20) % 2) * 5)
    const model = { kind: 'normal-known-variance', sigma: 0.3, priorMean: 2.5, priorStd: 4 } as const
    const constant = bocpd(y, model, constantHazard(20)).logEvidence
    const regular = bocpd(y, model, gammaHazard(100, 5)).logEvidence
    expect(regular).toBeGreaterThan(constant + 1)
  })
})

describe('the option tables', () => {
  it('name the shifts, each with its own prior knobs, and build the library model', () => {
    expect(segmentModels.names).toEqual(['mean', 'variance', 'mean-variance', 'rate', 'bernoulli', 'ar', 'trend'])
    expect(segmentModels.make({ kind: 'mean', sigma: 0.5, priorMean: 1, priorStd: 2 })).toEqual({
      kind: 'normal-known-variance',
      sigma: 0.5,
      priorMean: 1,
      priorStd: 2,
    })
    // Knobs not given take the option's defaults.
    expect(segmentModels.make({ kind: 'rate' })).toEqual({ kind: 'poisson-gamma', shape: 2, rate: 0.4 })
    expect(segmentModels.make(undefined).kind).toBe('normal-known-variance')
    expect(segmentModels.where((p) => p.regression)).toEqual(['ar', 'trend'])
    expect(segmentModels.where((p) => p.discrete)).toEqual(['rate', 'bernoulli'])
  })

  it('give a working model and hazard from every default configuration', () => {
    const y = [0.2, -0.1, 0.4, 3.2, 3.5, 2.9, 3.1, 0.1, 0, 0.3, -0.2, 0.1]
    const counts = y.map((v) => Math.round(Math.abs(v) * 2))
    for (const name of segmentModels.names) {
      const model = segmentModels.make({ ...(defaults(segmentModels.space) as object), kind: name })
      const series = name === 'rate' ? counts : name === 'bernoulli' ? counts.map((v) => (v > 2 ? 1 : 0)) : y
      for (const h of hazards.names) {
        const hazard = hazards.make({ ...(defaults(hazards.space) as object), kind: h })
        const r = bocpd(series, model, hazard)
        expect(Number.isFinite(r.logEvidence), `${name} with ${h}`).toBe(true)
      }
    }
  })

  it('offer the hazards with their knobs and properties', () => {
    expect(hazards.names).toEqual(['constant', 'minimum-segment', 'weibull', 'gamma', 'negative-binomial'])
    expect(hazards.make({ kind: 'constant', rate: 0.25 })(9)).toBe(0.25)
    expect(hazards.make({ kind: 'minimum-segment', rate: 0.1, minSegment: 4 })(3)).toBe(0)
    expect(hazards.where((p) => !p.allowsShort)).toEqual(['minimum-segment'])
    expect(hazards.where((p) => p.constant)).toEqual(['constant'])
  })

  it('make complete values, and a model gives its own configuration back', () => {
    expect(segmentModelValue('rate', { shape: 4 })).toEqual({ kind: 'rate', shape: 4, rate: 0.4 })
    expect(hazardValue('constant')).toEqual({ kind: 'constant', rate: 0.01 })
    expect(() => segmentModelValue('nope')).toThrow(/not one of/)
    expect(() => hazardValue('constant', { shape: 1 })).toThrow(/no knob/)
    for (const name of segmentModels.names) {
      const model = segmentModels.make(segmentModelValue(name))
      expect(segmentModels.make(segmentModelValueOf(model))).toEqual(model)
    }
  })
})
