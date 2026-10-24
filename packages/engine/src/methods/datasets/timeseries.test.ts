import { describe, expect, it } from 'vitest'
import { generate, load, memoryCache, sha256Hex } from '../../lib/data/index.ts'
import { key } from '../../lib/random/index.ts'
import { type Tensor, tensor, toFlat } from '../../lib/tensor/index.ts'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import {
  airPassengers,
  arma,
  armaProcess,
  type BocpdResult,
  changepointSeries,
  coalMiningCounts,
  coalMiningDisasters,
  electrocardiogram,
  exactBocpd,
  hawkesProcess,
  inhomogeneousPoisson,
  multivariateHawkes,
  poissonProcess,
  type SegmentModel,
  seasonalSeries,
  sinusoids,
  spatialPoisson,
  subsampleSeries,
  sunspots,
} from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('ARMA processes against statsmodels', () => {
  for (const c of loadFixture('methods/datasets/timeseries').cases) {
    it(c.name, () => {
      const ar = toFlat(decode(c.inputs[0]) as Tensor)
      const ma = toFlat(decode(c.inputs[1]) as Tensor)
      const g = armaProcess(key(0), { ar, ma, n: 5, sigma: c.params['sigma'] as number })
      const acvf = g.truth.extra?.['acvf'] as (h: number) => number
      expectMatches(tensor(Array.from({ length: 10 }, (_, h) => acvf(h))), c.output, c.rtol, 1e-12, c.name)
    })
  }

  it('simulated autocovariances match the theory', () => {
    const g = armaProcess(key(1), { ar: [0.6, -0.2], ma: [0.4], n: 20000 })
    const x = toFlat(g.data.values)
    const m = mean(x)
    const acvf = g.truth.extra?.['acvf'] as (h: number) => number
    for (const h of [0, 1, 2]) {
      const sample = mean(x.slice(h).map((v, i) => (v - m) * ((x[i] ?? 0) - m)))
      expect(Math.abs(sample - acvf(h))).toBeLessThan(0.06)
    }
    expect(armaProcess(key(0), { ar: [1.1], n: 5 }).truth.extra?.['stationary']).toBe(false)
    expect(armaProcess(key(0), { ma: [2], n: 5 }).truth.extra?.['invertible']).toBe(false)
    // The spectral density integrates to the variance.
    const f = g.truth.extra?.['spectralDensity'] as (w: number) => number
    let area = 0
    for (let i = 0; i < 4000; i++) area += f(-Math.PI + ((i + 0.5) * 2 * Math.PI) / 4000) * ((2 * Math.PI) / 4000)
    expect(area).toBeCloseTo(acvf(0), 6)
    expect(generate(arma, { knobs: { n: 30, phi1: 0.3, theta1: 0.2 } }).truth.extra?.['ma']).toEqual([0.2])
  })
})

describe('time series generators and real series', () => {
  it('sinusoids: the clean signal plus the asked noise, evenly or unevenly sampled', () => {
    const { data, truth } = generate(sinusoids, { key: key(2), knobs: { n: 2000, noise: 0.3, uneven: true } })
    const t = toFlat(data.times)
    expect(t.every((v, i) => i === 0 || v >= (t[i - 1] ?? 0))).toBe(true)
    expect(data.samplingRate).toBeUndefined()
    const clean = toFlat(truth.regressionFunction?.(data.times) as Tensor)
    const r = toFlat(data.values).map((v, i) => v - (clean[i] ?? 0))
    expect(Math.sqrt(mean(r.map((v) => v * v)))).toBeCloseTo(0.3, 1)
  })
  it('seasonal series decompose into their components', () => {
    const { data, truth } = generate(seasonalSeries, { key: key(3), knobs: { n: 120, period: 12 } })
    const parts = ['trend', 'seasonal', 'remainder'].map((p) => toFlat(truth.latents?.[p] as Tensor))
    toFlat(data.values).forEach((v, i) => {
      expect(v).toBeCloseTo(
        parts.reduce((s, p) => s + (p[i] ?? 0), 0),
        10,
      )
    })
    const s = parts[1] ?? []
    expect(s[0]).toBeCloseTo(s[12] ?? 0, 10)
    const thin = subsampleSeries(data, key(4), 0.25)
    expect(thin.values.shape).toEqual([30, 1])
  })
  it('real series', () => {
    expect(airPassengers().values.data[0]).toBe(112)
    expect(airPassengers().period).toBe(12)
    expect(sunspots().values.shape).toEqual([309, 1])
    const coal = coalMiningDisasters()
    expect(coal.window).toEqual([1851, 1963])
    expect(toFlat(coalMiningCounts().values).reduce((a, b) => a + b, 0)).toBe(191)
  })
  it('the ECG parser reads an npz of uint16 samples', async () => {
    // A stored (uncompressed) zip holding ecg.npy with three samples.
    const header = "{'descr': '<u2', 'fortran_order': False, 'shape': (3,), }"
    const pad = `${' '.repeat(64 - ((10 + header.length + 1) % 64))}\n`
    const text = new TextEncoder().encode(header + pad)
    const npy = new Uint8Array(10 + text.length + 6)
    npy.set([0x93, 78, 85, 77, 80, 89, 1, 0, text.length & 255, text.length >> 8])
    npy.set(text, 10)
    new DataView(npy.buffer).setUint16(10 + text.length, 1024, true)
    new DataView(npy.buffer).setUint16(12 + text.length, 1224, true)
    new DataView(npy.buffer).setUint16(14 + text.length, 824, true)
    const name = new TextEncoder().encode('ecg.npy')
    const local = new Uint8Array(30 + name.length + npy.length)
    const lv = new DataView(local.buffer)
    lv.setUint32(0, 0x04034b50, true)
    lv.setUint32(18, npy.length, true)
    lv.setUint32(22, npy.length, true)
    lv.setUint16(26, name.length, true)
    local.set(name, 30)
    local.set(npy, 30 + name.length)
    const central = new Uint8Array(46 + name.length)
    const cv = new DataView(central.buffer)
    cv.setUint32(0, 0x02014b50, true)
    cv.setUint32(20, npy.length, true)
    cv.setUint16(28, name.length, true)
    central.set(name, 46)
    const end = new Uint8Array(22)
    const ev = new DataView(end.buffer)
    ev.setUint32(0, 0x06054b50, true)
    ev.setUint16(10, 1, true)
    ev.setUint32(16, local.length, true)
    const zip = new Uint8Array(local.length + central.length + 22)
    zip.set(local, 0)
    zip.set(central, local.length)
    zip.set(end, local.length + central.length)
    const ref = { ...electrocardiogram, sha256: await sha256Hex(zip) }
    const series = await load(ref, {
      cache: memoryCache(),
      fetch: async () => ({ ok: true, status: 200, arrayBuffer: async () => zip.slice().buffer }),
    })
    expect(toFlat(series.values)).toEqual([0, 1, -1])
    expect(series.samplingRate).toBe(360)
  })
})

describe('changepoints', () => {
  const models: Record<string, SegmentModel> = {
    mean: { kind: 'mean', sigma: 0.7, priorMean: 0, priorStd: 2 },
    trend: { kind: 'trend', sigma: 0.5, priorMean: 0, interceptStd: 2, slopeStd: 0.3 },
    variance: { kind: 'variance', mean: 0, alpha: 3, beta: 2 },
    rate: { kind: 'rate', shape: 2, rate: 0.5 },
    ar: { kind: 'ar', sigma: 0.6, priorStd: 0.5 },
  }
  for (const c of loadFixture('methods/datasets/changepoints').cases) {
    it(`BOCPD ${c.name} equals the brute-force sum over segmentations`, () => {
      const y = toFlat(decode(c.inputs[0]) as Tensor)
      const min = (c.params['minSegment'] as number | undefined) ?? 1
      const r = exactBocpd(
        y,
        models[c.params['kind'] as string] as SegmentModel,
        min > 1 ? (run) => (run >= min ? 0.2 : 0) : 0.2,
      )
      expectMatches(tensor(r.logEvidence), c.output, c.rtol, 1e-10, c.name)
    })
  }

  for (const kind of ['mean', 'variance', 'rate', 'ar', 'trend']) {
    it(`${kind} changes: truth, run lengths and calibrated predictives`, () => {
      const { data, truth } = generate(changepointSeries, { key: key(5), knobs: { n: 400, kind, hazard: 0.02 } })
      const cps = truth.changepoints ?? []
      const seg = toFlat(truth.latents?.['segments'] as Tensor)
      expect(seg.filter((v, i) => i > 0 && v !== seg[i - 1]).length).toBe(cps.length)
      const run = truth.extra?.['bocpd'] as () => BocpdResult
      const bocpd = run()
      for (const p of bocpd.runLength) expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
      // The exact model detects most changes: the MAP run length restarts near each true changepoint.
      const restarts = bocpd.mapRunLength.flatMap((r, t) => (r === 1 && t > 0 ? [t] : []))
      const found = cps.filter((c) => restarts.some((r) => Math.abs(r - c) <= 10)).length
      if (kind === 'mean' || kind === 'trend') expect(found / Math.max(1, cps.length)).toBeGreaterThanOrEqual(0.5)
      // The changepoint model explains the data better than a single segment.
      const model = truth.extra?.['model'] as SegmentModel
      const single = exactBocpd(toFlat(data.values), model, 1e-9)
      if (cps.length > 0) expect(bocpd.logEvidence).toBeGreaterThan(single.logEvidence)
      expect(Number.isFinite(bocpd.logEvidence)).toBe(true)
      expect(data.events.length).toBe(cps.length)
    })
  }
})

describe('minimum segment lengths', () => {
  it('no segment is shorter than minSegment', () => {
    const { truth } = generate(changepointSeries, { key: key(3), knobs: { n: 300, hazard: 0.2, minSegment: 6 } })
    const cps = [0, ...(truth.changepoints ?? []), 300]
    for (let i = 1; i < cps.length - 1; i++)
      for (let i = 1; i < cps.length - 1; i++) expect((cps[i] ?? 0) - (cps[i - 1] ?? 0)).toBeGreaterThanOrEqual(6)
    const run = truth.extra?.['bocpd'] as () => BocpdResult
    expect(Number.isFinite(run().logEvidence)).toBe(true)
  })
})

describe('point processes', () => {
  it('Poisson counts match the rate', () => {
    const { data, truth } = generate(poissonProcess, { key: key(6), knobs: { rate: 3, horizon: 1000 } })
    const n = data.times?.shape[0] ?? 0
    expect(Math.abs(n - 3000)).toBeLessThan(4 * Math.sqrt(3000))
    expect(truth.extra?.['logLikelihood']).toBeCloseTo(n * Math.log(3) - 3000, 8)
  })
  it('inhomogeneous counts follow the compensator', () => {
    for (const shape of ['sinusoidal', 'piecewise', 'ramp']) {
      const { data, truth } = generate(inhomogeneousPoisson, { key: key(7), knobs: { shape, base: 5, horizon: 300 } })
      const comp = truth.extra?.['compensator'] as (t: number) => number
      const times = toFlat(data.times as Tensor)
      const half = times.filter((t) => t < 150).length
      expect(Math.abs(half - comp(150)), shape).toBeLessThan(4 * Math.sqrt(comp(150)))
      expect(Math.abs(times.length - comp(300)), shape).toBeLessThan(4 * Math.sqrt(comp(300)))
    }
  })
  it('Hawkes processes: stationary rate, parents and an exact log-likelihood', () => {
    const { data, truth } = generate(hawkesProcess, {
      key: key(8),
      knobs: { mu: 1, alpha: 0.5, beta: 2, horizon: 2000 },
    })
    const times = toFlat(data.times as Tensor)
    expect(Math.abs(times.length / 2000 - 2)).toBeLessThan(0.25)
    const parents = toFlat(truth.latents?.['parents'] as Tensor)
    expect(parents.every((p, i) => p < i)).toBe(true)
    const immigrants = parents.filter((p) => p === -1).length
    expect(Math.abs(immigrants / 2000 - 1)).toBeLessThan(0.15)
    // The recursive log-likelihood equals the direct formula with a numerically integrated compensator.
    const short = generate(hawkesProcess, { key: key(9), knobs: { horizon: 30 } })
    const ts = toFlat(short.data.times as Tensor)
    const intensity = short.truth.extra?.['intensity'] as (t: number) => number
    let integral = 0
    const steps = 60000
    for (let i = 0; i < steps; i++) integral += intensity(((i + 0.5) * 30) / steps) * (30 / steps)
    const direct = ts.reduce((s, t) => s + Math.log(intensity(t)), 0) - integral
    expect(short.truth.extra?.['logLikelihood']).toBeCloseTo(direct, 3)
    const multi = multivariateHawkes(key(10), {
      mu: [0.5, 0.2],
      alpha: [
        [0.2, 0.5],
        [0, 0.3],
      ],
      horizon: 500,
    })
    const marks = toFlat(multi.data.marks as Tensor)
    expect(new Set(marks).size).toBe(2)
  })
  it('spatial Poisson counts match the integrated intensity', () => {
    const { data, truth } = generate(spatialPoisson, { key: key(11), knobs: { rate: 500, bump: 4 } })
    const n = data.locations?.shape[0] ?? 0
    const expected = truth.extra?.['expectedCount'] as number
    expect(Math.abs(n - expected)).toBeLessThan(4 * Math.sqrt(expected) + 2)
  })
})
