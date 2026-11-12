// The partial pooling page's compute: the data, the two classical estimates and a short NUTS run on each form.
import { describe, expect, it } from 'vitest'
import {
  classicalEstimates,
  countsAt,
  FORMS,
  frameAt,
  makeData,
  type PoolingRequest,
  runPooling,
  SIZE_RATIO,
} from './compute.ts'

const host = {
  now: () => performance.now(),
  yieldToPage: () => new Promise<void>((resolve) => setTimeout(resolve, 0)),
  stopped: () => false,
  status: () => undefined,
}
const data = makeData({ groups: 8, smallest: 3, tau: 0.3, seed: 1 })
const request = (method: PoolingRequest['method']): PoolingRequest => ({
  method,
  data,
  seed: 1,
  chains: 2,
  warmup: 60,
  draws: 50,
})

describe('the data', () => {
  it('has sizes from the smallest to SIZE_RATIO times it and standard errors that follow them', () => {
    expect(data.size[0]).toBe(3)
    expect(data.size.at(-1)).toBe(3 * SIZE_RATIO)
    expect(data.se.every((s, j) => Math.abs(s * Math.sqrt(data.size[j] ?? 1) - 1) < 1e-12)).toBe(true)
    expect(makeData(data.settings)).toEqual(data)
  })

  it('has a common estimate between the extremes of the group means, tighter than any group alone', () => {
    const { alone, common } = classicalEstimates(data)
    expect(common.mean).toBeGreaterThan(Math.min(...data.mean))
    expect(common.mean).toBeLessThan(Math.max(...data.mean))
    expect(common.upper - common.lower).toBeLessThan(Math.min(...alone.map((a) => a.upper - a.lower)))
  })
})

describe('the runs', () => {
  for (const form of FORMS)
    it(`records the ${form} run: draws with their iterations, divergences and the group intervals along it`, async () => {
      const r = await runPooling(request(form), host)
      expect(r.steps).toBe(111)
      expect(r.drawsPerChain).toBe(50)
      expect(r.mu).toHaveLength(100)
      expect(r.theta).toHaveLength(8)
      expect(r.theta[0]).toHaveLength(100)
      expect(r.divergent).toHaveLength(100)
      expect(Math.min(...r.at)).toBe(62)
      expect(Math.max(...r.at)).toBe(111)
      expect(frameAt(r, 61)).toBeUndefined()
      const end = frameAt(r, 111)
      expect(end?.lower.every((l, j) => l < (end.mean[j] ?? 0) && (end.mean[j] ?? 0) < (end.upper[j] ?? 0))).toBe(true)
      expect(countsAt(r, 111).draws).toBe(100)
      expect(countsAt(r, 86).draws).toBe(50)
    })

  it('returns what it has when stopped', async () => {
    let n = 0
    const r = await runPooling(request('non-centred'), { ...host, stopped: () => ++n > 3 })
    expect(r.steps).toBeLessThan(111)
  })
})
