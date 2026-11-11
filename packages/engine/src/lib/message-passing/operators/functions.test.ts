import { describe, expect, it } from 'vitest'
import { gaussian, gaussianEta } from '../families.ts'
import { messageArg } from './args.ts'
import { apply, unaryFunctions } from './functions.ts'

/** The moments of x and f(x) under N(m, v) exp(h f(x) - J f(x)^2 / 2) on a fine grid: the reference. */
function brute(m: number, v: number, h: number, J: number, f: (x: number) => number) {
  const sd = Math.sqrt(v)
  const n = 400000
  const lo = m - 12 * sd
  const dx = (24 * sd) / n
  const lw: number[] = []
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i <= n; i++) {
    const x = lo + i * dx
    const fx = f(x)
    const l = -0.5 * ((x - m) / sd) ** 2 + h * fx - 0.5 * J * fx * fx
    lw.push(l)
    if (l > top) top = l
  }
  let z = 0
  let x1 = 0
  let x2 = 0
  let y1 = 0
  let y2 = 0
  for (let i = 0; i <= n; i++) {
    const x = lo + i * dx
    const w = Math.exp((lw[i] as number) - top)
    const fx = f(x)
    z += w
    x1 += w * x
    x2 += w * x * x
    y1 += w * fx
    y2 += w * fx * fx
  }
  return {
    x: { mean: x1 / z, variance: x2 / z - (x1 / z) ** 2 },
    y: { mean: y1 / z, variance: y2 / z - (y1 / z) ** 2 },
    logMass: top + Math.log((z * dx) / (sd * Math.sqrt(2 * Math.PI))),
  }
}

/** The moments a site message implies against a cavity. */
function implied(site: readonly number[], cavity: readonly number[]): { mean: number; variance: number } {
  const j = (cavity[1] ?? 0) + (site[1] ?? 0)
  return { mean: ((cavity[0] ?? 0) + (site[0] ?? 0)) / j, variance: 1 / j }
}

const table = unaryFunctions()
const names = Object.keys(table)
// Input cavities (mean, variance) and output messages (h, J): flat, proper, and improper with a field only.
const inputs: readonly (readonly [number, number])[] = [
  [0, 1],
  [1.5, 0.3],
  [-0.7, 2],
  [0, 25],
]
const outputs: readonly (readonly [number, number])[] = [
  [0, 0],
  [0.6, 2],
  [-1.5, 3],
  [0.8, 0],
]

describe('the unary functions of a Gaussian variable', () => {
  for (const name of names)
    it(`${name}: the messages to the output and to the input match a fine grid`, () => {
      const fn = table[name] as NonNullable<(typeof table)[string]>
      for (const [m, v] of inputs)
        for (const [h, J] of outputs) {
          // The exponential of a wide input has no proper tilt against a field-only output message.
          if (name === 'exp' && ((J === 0 && h !== 0) || v >= 25)) continue
          if (name === 'square' && J === 0 && h !== 0) continue
          const x = gaussianEta(m, v)
          const y = [h, J]
          const args = [messageArg(gaussian, y), messageArg(gaussian, x)]
          const ref = brute(m, v, h, J, fn.f)
          const toY = apply.ep?.({ mode: 'ep', to: 0, args, params: { fn: name }, power: 1 })
          const ty = implied((toY as { message: number[] }).message, y)
          const label = `${name} at x ~ N(${m}, ${v}), y message (${h}, ${J})`
          // The message to y moves the output to the tilted output: its moments are those of the reference (when y has a precision).
          const yMoments = (toY as { message: number[] }).message
          expect((yMoments[1] ?? 0) + J, label).toBeGreaterThan(-1e9)
          if ((yMoments[1] ?? 0) + J > 0) {
            expect(Math.abs(ty.mean - ref.y.mean) / Math.sqrt(ref.y.variance), `${label}: output mean`).toBeLessThan(
              2e-3,
            )
            expect(ty.variance / ref.y.variance, `${label}: output variance`).toBeCloseTo(1, 2)
          }
          if (J !== 0 || h !== 0) {
            const toX = apply.ep?.({ mode: 'ep', to: 1, args, params: { fn: name }, power: 1 })
            const tx = implied((toX as { message: number[] }).message, x)
            expect(Math.abs(tx.mean - ref.x.mean) / Math.sqrt(ref.x.variance), `${label}: input mean`).toBeLessThan(
              2e-3,
            )
            expect(tx.variance / ref.x.variance, `${label}: input variance`).toBeCloseTo(1, 2)
          }
        }
    })

  it('a flat output tells the input nothing, and gets the exact pushforward moments', () => {
    const args = [messageArg(gaussian, [0, 0]), messageArg(gaussian, gaussianEta(0.4, 1.7))]
    for (const name of names) {
      const toX = apply.ep?.({ mode: 'ep', to: 1, args, params: { fn: name }, power: 1 })
      expect(toX?.message).toEqual([0, 0])
      const fn = table[name] as NonNullable<(typeof table)[string]>
      const toY = apply.ep?.({ mode: 'ep', to: 0, args, params: { fn: name }, power: 1 })
      const t = implied((toY as { message: number[] }).message, [0, 0])
      const ref = brute(0.4, 1.7, 0, 0, fn.f)
      expect(Math.abs(t.mean - ref.y.mean), name).toBeLessThan(2e-3 * Math.sqrt(ref.y.variance))
      expect(t.variance / ref.y.variance, name).toBeCloseTo(1, 2)
    }
  })

  it('the evidence is the log mass of the input cavity times the output message pushed through f', () => {
    for (const name of ['tanh', 'relu', 'softplus', 'exp'] as const) {
      const fn = table[name] as NonNullable<(typeof table)[string]>
      const y = gaussianEta(0.3, 0.8)
      const args = [messageArg(gaussian, y), messageArg(gaussian, gaussianEta(0.2, 1.3))]
      const ev = apply.logEvidence?.({ mode: 'ep', to: 0, args, params: { fn: name }, power: 1 }) as number
      const ref = brute(0.2, 1.3, y[0] as number, y[1] as number, fn.f)
      expect(ev, name).toBeCloseTo(gaussian.logPartition(gaussianEta(0.2, 1.3)) + ref.logMass, 3)
    }
  })

  it('an unknown function is reported by check', () => {
    expect(apply.check?.({ fn: 'cube' }, [])).toMatch(/unknown function/)
    expect(apply.check?.({ fn: 'tanh' }, [])).toBeUndefined()
  })
})
