import { describe, expect, it } from 'vitest'
import { add, arange, div, mul, toFlat } from '../../../tensor/index.ts'
import { messageArg } from '../../core/args.ts'
import { gamma, gammaEta, gaussian, gaussianEta } from '../../core/families.ts'
import { apply, applyProductOperator, unaryFunctions } from './functions.ts'

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
// `sqrt` and `recip` are not defined on the whole line: they are checked on positive inputs below.
const names = Object.keys(table).filter((n) => n !== 'sqrt' && n !== 'recip')
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

/** The moments of `x ~ Gamma(shape, rate)` tilted by `exp(h f(x) - J f(x)^2 / 2)`, on a fine grid in `log x`: the reference. */
function bruteGamma(shape: number, rate: number, h: number, J: number, f: (x: number) => number) {
  const n = 200000
  const lo = Math.log(shape / rate) - 14
  const hi = Math.log(shape / rate) + 10
  const dt = (hi - lo) / n
  let top = Number.NEGATIVE_INFINITY
  const lw: number[] = []
  for (let i = 0; i <= n; i++) {
    const t = lo + i * dt
    const fx = f(Math.exp(t))
    const l = shape * t - rate * Math.exp(t) + h * fx - 0.5 * J * fx * fx
    lw.push(l)
    if (l > top) top = l
  }
  let z = 0
  let logX = 0
  let xMean = 0
  let yMean = 0
  let ySecond = 0
  for (let i = 0; i <= n; i++) {
    const t = lo + i * dt
    const x = Math.exp(t)
    const fx = f(x)
    const w = Math.exp((lw[i] as number) - top)
    z += w
    logX += w * t
    xMean += w * x
    yMean += w * fx
    ySecond += w * fx * fx
  }
  return { logX: logX / z, x: xMean / z, y: { mean: yMean / z, variance: ySecond / z - (yMean / z) ** 2 } }
}

describe('a chain of steps and a gamma input', () => {
  const chain = [['mul', 3], ['add', 0.5], ['sqrt'], ['recip']] as const
  const f = (x: number): number => 1 / Math.sqrt(3 * x + 0.5)

  it('a chain is the composition of its steps, and an unknown step is reported by check', () => {
    expect(apply.check?.({ steps: chain }, [])).toBeUndefined()
    expect(apply.check?.({ steps: [['cube']] }, [])).toMatch(/unknown step/)
    // x ~ N(1, 0.01) through 1 / sqrt(3 x + 0.5): the pushforward of a flat output is the delta-method moment.
    const args = [messageArg(gaussian, [0, 0]), messageArg(gaussian, gaussianEta(1, 0.01))]
    const toY = apply.ep?.({ mode: 'ep', to: 0, args, params: { steps: chain }, power: 1 })
    const t = implied((toY as { message: number[] }).message, [0, 0])
    expect(t.mean).toBeCloseTo(f(1), 2)
  })

  it('min and max steps clip the chain, and the message matches a fine grid with the clip binding on one side', () => {
    // x ~ N(0.9, 0.09) through max(min(1 - x^2, 1), 0.05) with a reader N(h, J): the lower bound binds for |x| near 1 and above.
    const steps = [['square'], ['mul', -1], ['add', 1], ['min', 1], ['max', 0.05]] as const
    const clipped = (x: number): number => Math.max(Math.min(1 - x * x, 1), 0.05)
    expect(apply.check?.({ steps }, [])).toBeUndefined()
    const [m, v, h, J] = [0.9, 0.09, 1.1, 4]
    const ref = brute(m, v, h, J, clipped)
    const args = [messageArg(gaussian, [h, J]), messageArg(gaussian, gaussianEta(m, v))]
    const toY = apply.ep?.({ mode: 'ep', to: 0, args, params: { steps }, power: 1 })
    const ty = implied((toY as { message: number[] }).message, [h, J])
    expect(Math.abs(ty.mean - ref.y.mean) / Math.sqrt(ref.y.variance)).toBeLessThan(0.02)
    const toX = apply.ep?.({ mode: 'ep', to: 1, args, params: { steps }, power: 1 })
    const tx = implied((toX as { message: number[] }).message, gaussianEta(m, v))
    expect(Math.abs(tx.mean - ref.x.mean) / Math.sqrt(ref.x.variance)).toBeLessThan(0.02)
    expect(tx.variance / ref.x.variance).toBeGreaterThan(0.9)
    expect(tx.variance / ref.x.variance).toBeLessThan(1.1)
  })

  it('a chain on a Gaussian input matches a fine grid, to the output and to the input', () => {
    // The chain is defined for 3 x + 0.5 > 0: the cavity stays well inside it.
    const [m, v, h, J] = [1.2, 0.0025, 0.9, 3]
    const ref = brute(m, v, h, J, f)
    const args = [messageArg(gaussian, [h, J]), messageArg(gaussian, gaussianEta(m, v))]
    const toY = apply.ep?.({ mode: 'ep', to: 0, args, params: { steps: chain }, power: 1 })
    const ty = implied((toY as { message: number[] }).message, [h, J])
    expect(Math.abs(ty.mean - ref.y.mean) / Math.sqrt(ref.y.variance)).toBeLessThan(2e-3)
    expect(ty.variance / ref.y.variance).toBeCloseTo(1, 2)
    const toX = apply.ep?.({ mode: 'ep', to: 1, args, params: { steps: chain }, power: 1 })
    const tx = implied((toX as { message: number[] }).message, gaussianEta(m, v))
    expect(Math.abs(tx.mean - ref.x.mean) / Math.sqrt(ref.x.variance)).toBeLessThan(2e-3)
    expect(tx.variance / ref.x.variance).toBeCloseTo(1, 2)
  })

  it('a gamma input: the messages match a fine grid in log x (the message to the input is the gamma of the tilted moments)', () => {
    for (const [shape, rate, h, J] of [
      [3, 2, 0.9, 3],
      [1, 0.05, 0.5, 8],
      [4, 1, 0, 0],
    ] as const) {
      const ref = bruteGamma(shape, rate, h, J, f)
      const cavity = gammaEta(shape, rate)
      const args = [messageArg(gaussian, [h, J]), messageArg(gamma, cavity)]
      const toY = apply.ep?.({ mode: 'ep', to: 0, args, params: { steps: chain }, power: 1 })
      const ty = implied((toY as { message: number[] }).message, [h, J])
      expect(Math.abs(ty.mean - ref.y.mean) / Math.sqrt(ref.y.variance), `y mean ${shape} ${rate}`).toBeLessThan(5e-3)
      expect(ty.variance / ref.y.variance, `y variance ${shape} ${rate}`).toBeCloseTo(1, 1)
      if (h === 0 && J === 0) continue
      const toX = apply.ep?.({ mode: 'ep', to: 1, args, params: { steps: chain }, power: 1 })
      const tilted = (toX as { message: number[] }).message.map((q, i) => q + (cavity[i] ?? 0))
      const moments = gamma.moments(tilted)
      expect(moments[0], `E log x ${shape} ${rate}`).toBeCloseTo(ref.logX, 1)
      expect((moments[1] ?? 0) / ref.x, `E x ${shape} ${rate}`).toBeCloseTo(1, 1)
    }
  })
})

/** The moments of `z`, `x` and `y = z f(x)` under `N(z) N(x) exp(h y - J y^2 / 2)` on a fine two-dimensional grid. */
function bruteScaled(
  z: readonly [number, number],
  x: readonly [number, number],
  h: number,
  J: number,
  f: (x: number) => number,
) {
  const n = 1200
  const grid = (m: number, v: number): number[] =>
    toFlat(add(m, mul(add(-9, div(mul(arange(n + 1), 18), n)), Math.sqrt(v))))
  const zs = grid(z[0], z[1])
  const xs = grid(x[0], x[1])
  const dz = (18 * Math.sqrt(z[1])) / n
  const dx = (18 * Math.sqrt(x[1])) / n
  let top = Number.NEGATIVE_INFINITY
  const logw = (a: number, b: number): number => {
    const y = a * f(b)
    return -0.5 * ((a - z[0]) ** 2 / z[1] + (b - x[0]) ** 2 / x[1]) + h * y - 0.5 * J * y * y
  }
  for (const a of zs) for (const b of xs) top = Math.max(top, logw(a, b))
  const sums = { m0: 0, z1: 0, z2: 0, x1: 0, x2: 0, y1: 0, y2: 0 }
  for (const a of zs)
    for (const b of xs) {
      const w = Math.exp(logw(a, b) - top)
      const y = a * f(b)
      sums.m0 += w
      sums.z1 += w * a
      sums.z2 += w * a * a
      sums.x1 += w * b
      sums.x2 += w * b * b
      sums.y1 += w * y
      sums.y2 += w * y * y
    }
  const stat = (s1: number, s2: number): { mean: number; variance: number } => ({
    mean: s1 / sums.m0,
    variance: s2 / sums.m0 - (s1 / sums.m0) ** 2,
  })
  return {
    z: stat(sums.z1, sums.z2),
    x: stat(sums.x1, sums.x2),
    y: stat(sums.y1, sums.y2),
    logMass: top + Math.log(sums.m0 * dz * dx) - Math.log(2 * Math.PI * Math.sqrt(z[1] * x[1])),
  }
}

describe('apply-product: y = z f(x)', () => {
  // The message of one port, which the operator always has.
  const send = (to: number, args: ReturnType<typeof messageArg>[], steps: unknown): number[] =>
    (applyProductOperator.ep as NonNullable<typeof applyProductOperator.ep>)({
      mode: 'ep',
      to,
      args,
      params: { steps: steps as never },
      power: 1,
    }).message as number[]
  const chain = [['mul', 3], ['add', 0.5], ['sqrt'], ['recip']] as const
  const f = (x: number): number => 1 / Math.sqrt(3 * x + 0.5)
  const op = applyProductOperator

  it('the messages to y, z and x match a two-dimensional grid', () => {
    for (const [h, J] of [
      [0, 0],
      [1.1, 4],
      [-0.6, 0.5],
    ] as const) {
      const z = [0.3, 1] as const
      const x = [1.2, 0.01] as const
      const ref = bruteScaled(z, x, h, J, f)
      const args = [
        messageArg(gaussian, [h, J]),
        messageArg(gaussian, gaussianEta(z[0], z[1])),
        messageArg(gaussian, gaussianEta(x[0], x[1])),
      ]
      const cavities = [[h, J], gaussianEta(z[0], z[1]), gaussianEta(x[0], x[1])]
      const names = ['y', 'z', 'x'] as const
      for (const to of [0, 1, 2]) {
        if (h === 0 && J === 0 && to !== 0) {
          expect(send(to, args, chain)).toEqual([0, 0])
          continue
        }
        const m = send(to, args, chain)
        const got = implied(m, cavities[to] as number[])
        const want = ref[names[to] as 'y' | 'z' | 'x']
        const label = `${names[to]} at message (${h}, ${J})`
        expect(Math.abs(got.mean - want.mean) / Math.sqrt(want.variance), label).toBeLessThan(5e-3)
        expect(got.variance / want.variance, label).toBeCloseTo(1, 2)
      }
    }
  })

  it('a gamma input: the product with a gamma-scaled function matches a grid in log x', () => {
    const [h, J] = [0.8, 2]
    const [shape, rate] = [3, 2]
    const z = [0, 1] as const
    // E over x of the Gaussian-in-z integral, by a fine grid: the moments of y = z f(x).
    const n = 40000
    const lo = Math.log(shape / rate) - 12
    const hi = Math.log(shape / rate) + 8
    const rows: { w: number; t: number; x: number; my: number; sy: number }[] = []
    for (let i = 0; i <= n; i++) {
      const t = lo + ((hi - lo) * i) / n
      const x = Math.exp(t)
      const u = f(x)
      const precision = 1 + J * u * u
      const linear = h * u
      const logZ = -0.5 * Math.log(precision) + (0.5 * (linear * linear)) / precision
      rows.push({ w: shape * t - rate * x + logZ, t, x, my: (u * linear) / precision, sy: (u * u) / precision })
    }
    const top = Math.max(...rows.map((r) => r.w))
    let z0 = 0
    let y1 = 0
    let y2 = 0
    let logX = 0
    for (const r of rows) {
      const w = Math.exp(r.w - top)
      z0 += w
      y1 += w * r.my
      y2 += w * (r.sy + r.my * r.my)
      logX += w * r.t
    }
    const want = { mean: y1 / z0, variance: y2 / z0 - (y1 / z0) ** 2 }
    const args = [
      messageArg(gaussian, [h, J]),
      messageArg(gaussian, gaussianEta(z[0], z[1])),
      messageArg(gamma, gammaEta(shape, rate)),
    ]
    const toY = send(0, args, chain)
    const got = implied(toY, [h, J])
    expect(Math.abs(got.mean - want.mean) / Math.sqrt(want.variance)).toBeLessThan(5e-3)
    expect(got.variance / want.variance).toBeCloseTo(1, 2)
    const toX = send(2, args, chain)
    const moments = gamma.moments(toX.map((q, i) => q + (gammaEta(shape, rate)[i] ?? 0)))
    expect(moments[0]).toBeCloseTo(logX / z0, 2)
  })

  it('has the evidence of the grid and reports an unknown step', () => {
    const [h, J] = [0.5, 1.5]
    const z = [0.2, 1] as const
    const x = [1.2, 0.01] as const
    const ref = bruteScaled(z, x, h, J, f)
    const args = [
      messageArg(gaussian, [h, J]),
      messageArg(gaussian, gaussianEta(z[0], z[1])),
      messageArg(gaussian, gaussianEta(x[0], x[1])),
    ]
    const ev = op.logEvidence?.({ mode: 'ep', to: 0, args, params: { steps: chain }, power: 1 }) as number
    expect(ev).toBeCloseTo(
      gaussian.logPartition(gaussianEta(z[0], z[1])) + gaussian.logPartition(gaussianEta(x[0], x[1])) + ref.logMass,
      2,
    )
    expect(op.check?.({ steps: [['cube']] }, [])).toMatch(/unknown step/)
  })
})
