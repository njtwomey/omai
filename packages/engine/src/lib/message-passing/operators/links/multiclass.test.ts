import { describe, expect, it } from 'vitest'
import { type BruteBelief, type BrutePort, bruteExpectation, bruteTilted } from '../../../../testing/message-passing.ts'
import { ndtr } from '../../../numerics/special/_kernels.ts'
import { add, arange, div, mul, toFlat } from '../../../tensor/index.ts'
import { massOf, messageArg, valueArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import type { Arg, Params } from '../../core/types.ts'
import { zeroArray } from '../../numerics/_arrays.ts'
import { defaultOperators } from '../registry.ts'
import { multiclassSite } from './multiclass.ts'

/*
 * The tilted means and variances of the class scores and the log evidence against a brute-force integral over a fine grid of the
 * scores, with no code of the operator in it.
 */

const logSoftmax = (z: readonly number[], y: number): number => {
  const top = Math.max(...z)
  return (z[y] as number) - top - Math.log(z.reduce((a, v) => a + Math.exp(v - top), 0))
}

/** The multinomial probit probability by a fine one-dimensional integral over the noise of the chosen class. */
const logProbit = (z: readonly number[], y: number, n = 160): number => {
  let total = 0
  for (let i = 0; i < n; i++) {
    const e = -8 + ((i + 0.5) * 16) / n
    let p = Math.exp(-0.5 * e * e - 0.5 * Math.log(2 * Math.PI))
    for (let j = 0; j < z.length; j++) if (j !== y) p *= ndtr((z[y] as number) - (z[j] as number) + e)
    total += (p * 16) / n
  }
  return Math.log(total)
}

interface Cavity {
  readonly m: number
  readonly v: number
}

/** The brute force over a grid of K = 3 scores (or fewer free ones: a null cavity is an observed value). */
function brute(
  cavities: readonly Cavity[],
  logf: (z: readonly number[]) => number,
  n = 70,
  span = 6,
): { logMass: number; mean: number[]; variance: number[] } {
  const axes = cavities.map((c) => {
    const sd = Math.sqrt(c.v)
    return toFlat(add(c.m, mul(add(-span, div(mul(mul(add(arange(n), 0.5), 2), span), n)), sd)))
  })
  const steps = cavities.map((c) => (Math.sqrt(c.v) * 2 * span) / n)
  const k = cavities.length
  let mass = 0
  const m1 = zeroArray(k)
  const m2 = zeroArray(k)
  const z = zeroArray(k)
  const walk = (d: number, logw: number): void => {
    if (d === k) {
      const w = Math.exp(logw + logf(z))
      mass += w
      for (let i = 0; i < k; i++) {
        m1[i] = (m1[i] as number) + w * (z[i] as number)
        m2[i] = (m2[i] as number) + w * (z[i] as number) ** 2
      }
      return
    }
    const c = cavities[d] as Cavity
    for (let i = 0; i < n; i++) {
      const x = (axes[d] as number[])[i] as number
      z[d] = x
      walk(
        d + 1,
        logw - (0.5 * (x - c.m) ** 2) / c.v - 0.5 * Math.log(2 * Math.PI * c.v) + Math.log(steps[d] as number),
      )
    }
  }
  walk(0, 0)
  const mean = m1.map((v) => v / mass)
  return { logMass: Math.log(mass), mean, variance: m2.map((v, i) => v / mass - (mean[i] as number) ** 2) }
}

function check(
  link: 'logit' | 'probit',
  y: number,
  cavities: readonly Cavity[],
  tolerance = 2e-3,
  extra: Params = {},
  power = 1,
): void {
  const args: Arg[] = cavities.map((c) => messageArg(gaussian, gaussianEta(c.m, c.v)))
  const params = { y, classes: cavities.length, link, ...extra }
  const log = link === 'logit' ? (z: readonly number[]) => logSoftmax(z, y) : (z: readonly number[]) => logProbit(z, y)
  const logf = (z: readonly number[]): number => power * log(z)
  const b = brute(cavities, logf, cavities.length === 3 ? (link === 'probit' ? 26 : 60) : 200)
  for (let k = 0; k < cavities.length; k++) {
    const got = multiclassSite.ep?.({ mode: 'ep', to: k, args, params, power }).message as readonly number[]
    const c = cavities[k] as Cavity
    // The site is the tilted Gaussian over the cavity in natural parameters.
    const wantPrecision = 1 / (b.variance[k] as number) - 1 / c.v
    const wantH = (b.mean[k] as number) / (b.variance[k] as number) - c.m / c.v
    expect(Math.abs((got[1] as number) - wantPrecision), `${link} class ${k} precision`).toBeLessThan(
      tolerance * (1 + Math.abs(wantPrecision)),
    )
    expect(Math.abs((got[0] as number) - wantH), `${link} class ${k} mean term`).toBeLessThan(
      tolerance * (1 + Math.abs(wantH)),
    )
  }
  const want = b.logMass + args.reduce((s, a) => s + massOf(a), 0)
  const got = multiclassSite.logEvidence?.({ mode: 'ep', to: 0, args, params, power }) as number
  expect(Math.abs(got - want), `${link} evidence ${got} against ${want}`).toBeLessThan(tolerance)
}

describe('multiclass-site: tilted moments and log evidence against a brute-force integral', () => {
  it('softmax, three classes, moderate cavities', () => {
    check('logit', 1, [
      { m: 0.3, v: 1 },
      { m: -0.2, v: 0.8 },
      { m: 0.1, v: 1.5 },
    ])
  })
  it('softmax, three classes, wide cavities (the first sweeps of a run)', () => {
    check(
      'logit',
      2,
      [
        { m: 0, v: 9 },
        { m: 0, v: 9 },
        { m: 0, v: 9 },
      ],
      8e-3,
    )
  })
  it('softmax, two classes is the logistic link', () => {
    check('logit', 0, [
      { m: 0.4, v: 1.2 },
      { m: -0.3, v: 0.6 },
    ])
  })
  it('multinomial probit, three classes', () => {
    check(
      'probit',
      0,
      [
        { m: 0.2, v: 1 },
        { m: -0.1, v: 0.7 },
        { m: 0.4, v: 1.1 },
      ],
      1.5e-2,
    )
  }, 30_000)
  it('multinomial probit with two classes is the binary probit of the difference over sqrt 2', () => {
    // P(y = 0 | z) = Phi((z0 - z1) / sqrt 2): the tilted mean of z0 - z1 has the closed form of the probit site on d with sd sqrt 2.
    check(
      'probit',
      0,
      [
        { m: 0.5, v: 1 },
        { m: -0.2, v: 0.8 },
      ],
      4e-3,
    )
  }, 30_000)
  it('an observed score is a constant coordinate', () => {
    const args = [
      messageArg(gaussian, gaussianEta(0.2, 1)),
      valueArg(gaussian, 0.5),
      messageArg(gaussian, gaussianEta(-0.3, 0.9)),
    ]
    const params = { y: 1, classes: 3 }
    const b = brute(
      [
        { m: 0.2, v: 1 },
        { m: -0.3, v: 0.9 },
      ],
      ([a, c]) => logSoftmax([a as number, 0.5, c as number], 1),
      200,
    )
    const want = b.logMass + massOf(args[0] as Arg) + massOf(args[2] as Arg)
    expect(multiclassSite.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 })).toBeCloseTo(want, 3)
  })
  it('is in the registry with its evidence, and refuses a bad class', () => {
    expect(defaultOperators()['multiclass-site']?.logEvidence).toBeDefined()
    expect(multiclassSite.check?.({ y: 3, classes: 3 }, [])).toMatch(/class must be/)
    expect(multiclassSite.check?.({ y: 1, classes: 3, link: 'cloglog' }, [])).toMatch(/link/)
  })
})

describe('multiclass-site: the auxiliary-variable rule (any number of classes)', () => {
  const aux = { integration: 'auxiliary' }
  const three: Cavity[] = [
    { m: 0.3, v: 1 },
    { m: -0.2, v: 0.8 },
    { m: 0.1, v: 1.5 },
  ]
  const wide: Cavity[] = [
    { m: 0, v: 9 },
    { m: 0, v: 9 },
    { m: 0, v: 9 },
  ]

  it('softmax, two and three classes: the same moments and evidence as the brute-force integral', () => {
    check('logit', 1, three, 2e-3, aux)
    check('logit', 2, wide, 8e-3, aux)
    check(
      'logit',
      0,
      [
        { m: 0.4, v: 1.2 },
        { m: -0.3, v: 0.6 },
      ],
      2e-3,
      aux,
    )
  })
  it('softmax with a power (fractional EP): the factor to the power is integrated', () => {
    check('logit', 0, three, 2e-3, aux, 0.5)
    check('logit', 2, three, 2e-3, aux, 0.25)
    check('logit', 1, three, 2e-3, aux, 1.7)
  })
  it('multinomial probit, two and three classes', () => {
    const two = [
      { m: 0.5, v: 1 },
      { m: -0.2, v: 0.8 },
    ]
    const tri = [
      { m: 0.2, v: 1 },
      { m: -0.1, v: 0.7 },
      { m: 0.4, v: 1.1 },
    ]
    check('probit', 0, two, 4e-3, aux)
    check('probit', 0, tri, 1.5e-2, aux)
    check('probit', 2, tri, 1.5e-2, aux)
  }, 30_000)

  /** The precisions, mean terms and evidence of the softmax rule against the shared brute force (a product Gauss-Legendre rule zoomed onto the mass). */
  function againstBrute(
    y: number,
    cavities: readonly Cavity[],
    panels: number,
    order: number,
    tolerance: number,
  ): void {
    const args: Arg[] = cavities.map((c) => messageArg(gaussian, gaussianEta(c.m, c.v)))
    const ports: BrutePort[] = args.map((a) => ({ kind: 'gaussian', eta: a.eta as readonly number[] }))
    const params = { y, classes: cavities.length, link: 'logit', integration: 'auxiliary' }
    for (let k = 0; k < cavities.length; k++) {
      const ref = bruteTilted(ports, (z) => logSoftmax(z, y), k, { panels, order })
      const mean = ref.moments[0] as number
      const variance = (ref.moments[1] as number) - mean * mean
      const c = cavities[k] as Cavity
      const got = multiclassSite.ep?.({ mode: 'ep', to: k, args, params, power: 1 }).message as readonly number[]
      expect(Math.abs((got[1] as number) - (1 / variance - 1 / c.v)), `class ${k} precision`).toBeLessThan(tolerance)
      expect(Math.abs((got[0] as number) - (mean / variance - c.m / c.v)), `class ${k} mean term`).toBeLessThan(
        tolerance,
      )
      if (k === 0) {
        const evidence = multiclassSite.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number
        expect(Math.abs(evidence - ref.logZ), 'evidence').toBeLessThan(tolerance)
      }
    }
  }
  it('softmax, four classes: against the brute force', () => {
    againstBrute(
      2,
      [
        { m: 0.3, v: 1 },
        { m: -0.5, v: 0.6 },
        { m: 0.8, v: 1.2 },
        { m: 0, v: 2 },
      ],
      6,
      8,
      1e-3,
    )
  }, 120_000)

  /**
   * Two exact identities of any categorical likelihood against any cavities, whatever the number of classes: the masses of the
   * classes add up to one, and the tilted means (second moments) weighted by them add up to the cavity's, as `sum_y P(y | z) = 1`.
   */
  function identities(
    link: 'logit' | 'probit',
    cavities: readonly Cavity[],
  ): { readonly mass: number; readonly mean: number; readonly second: number } {
    const args: Arg[] = cavities.map((c) => messageArg(gaussian, gaussianEta(c.m, c.v)))
    const k = cavities.length
    const base = args.reduce((s, a) => s + massOf(a), 0)
    let mass = 0
    const first = zeroArray(k)
    const second = zeroArray(k)
    for (let y = 0; y < k; y++) {
      const params = { y, classes: k, link }
      const w = Math.exp((multiclassSite.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number) - base)
      mass += w
      for (let j = 0; j < k; j++) {
        const c = cavities[j] as Cavity
        const message = multiclassSite.ep?.({ mode: 'ep', to: j, args, params, power: 1 }).message as readonly number[]
        const precision = 1 / c.v + (message[1] as number)
        const mean = (c.m / c.v + (message[0] as number)) / precision
        first[j] = (first[j] as number) + w * mean
        second[j] = (second[j] as number) + w * (1 / precision + mean * mean)
      }
    }
    let dm = 0
    let ds = 0
    for (let j = 0; j < k; j++) {
      const c = cavities[j] as Cavity
      dm = Math.max(dm, Math.abs((first[j] as number) - c.m))
      ds = Math.max(ds, Math.abs((second[j] as number) - (c.m * c.m + c.v)))
    }
    return { mass: Math.abs(mass - 1), mean: dm, second: ds }
  }
  const spread = (k: number, scale: number, variance: number): Cavity[] =>
    Array.from({ length: k }, (_, j) => ({ m: scale * Math.sin(1.7 * j), v: variance * (1 + 0.2 * (j % 4)) }))

  it('softmax of 7, 10 and 16 classes (the default rule): the classes add up to one and the moments to the cavity', () => {
    for (const [k, scale, variance] of [
      [7, 1, 0.5],
      [10, 3, 9],
      [16, 1.5, 2],
    ] as const) {
      const r = identities('logit', spread(k, scale, variance))
      expect(r.mass, `${k} classes: the masses`).toBeLessThan(1e-6)
      expect(r.mean, `${k} classes: the means`).toBeLessThan(1e-6)
      expect(r.second, `${k} classes: the second moments`).toBeLessThan(1e-5)
    }
  }, 60_000)
  it('multinomial probit of 7 and 12 classes (the default rule): the same identities', () => {
    for (const [k, scale, variance] of [
      [7, 1, 0.5],
      [12, 3, 9],
    ] as const) {
      const r = identities('probit', spread(k, scale, variance))
      expect(r.mass, `${k} classes: the masses`).toBeLessThan(1e-8)
      expect(r.mean, `${k} classes: the means`).toBeLessThan(1e-8)
      expect(r.second, `${k} classes: the second moments`).toBeLessThan(1e-7)
    }
  }, 60_000)
  it('identical cavities: every class has the mass 1 / K', () => {
    for (const k of [8, 20]) {
      const args: Arg[] = Array.from({ length: k }, () => messageArg(gaussian, gaussianEta(0.7, 3)))
      const base = args.reduce((s, a) => s + massOf(a), 0)
      for (const link of ['logit', 'probit']) {
        const params = { y: 3, classes: k, link }
        const ev = multiclassSite.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number
        expect(Math.abs(ev - base + Math.log(k)), `${link} ${k}`).toBeLessThan(1e-8)
      }
    }
  })
  it('agrees with the product grid where that reaches (six classes)', () => {
    const cavities = spread(6, 1, 0.6)
    const args: Arg[] = cavities.map((c) => messageArg(gaussian, gaussianEta(c.m, c.v)))
    const run = (integration: string): number[][] =>
      cavities.map(
        (_, k) =>
          multiclassSite.ep?.({ mode: 'ep', to: k, args, params: { y: 2, classes: 6, integration }, power: 1 })
            .message as number[],
      )
    const a = run('product')
    const b = run('auxiliary')
    for (let k = 0; k < 6; k++) {
      expect(Math.abs((a[k]?.[0] as number) - (b[k]?.[0] as number)), `mean term ${k}`).toBeLessThan(3e-3)
      expect(Math.abs((a[k]?.[1] as number) - (b[k]?.[1] as number)), `precision ${k}`).toBeLessThan(3e-3)
    }
  }, 30_000)
  it('known scores are constants: seven classes, three of them known, against the brute force over the other four', () => {
    const cavities: Cavity[] = [
      { m: 0.2, v: 0.5 },
      { m: -0.4, v: 0.7 },
      { m: 0.9, v: 0.4 },
      { m: 0, v: 1 },
    ]
    const known = [0.5, -1, 0.3]
    const free = [0, 1, 3, 5]
    const place = (z: readonly number[]): number[] => {
      const out = [0, 0, known[0] as number, 0, known[1] as number, 0, known[2] as number]
      free.forEach((i, j) => {
        out[i] = z[j] as number
      })
      return out
    }
    const args: Arg[] = place([]).map((v, i) => {
      const j = free.indexOf(i)
      return j < 0
        ? valueArg(gaussian, v)
        : messageArg(gaussian, gaussianEta((cavities[j] as Cavity).m, (cavities[j] as Cavity).v))
    })
    const ports: BrutePort[] = cavities.map((_, j) => ({
      kind: 'gaussian',
      eta: (args[free[j] as number] as Arg).eta as readonly number[],
    }))
    for (const y of [1, 4]) {
      const params = { y, classes: 7, link: 'logit' }
      const evidence = multiclassSite.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number
      const ref = bruteTilted(ports, (z) => logSoftmax(place(z), y), 0, { panels: 3, order: 8 })
      expect(Math.abs(evidence - ref.logZ), `evidence with y = ${y}`).toBeLessThan(2e-3)
    }
  }, 60_000)
  it('says what it does not do: the product grid stops at six, the probit takes no power, variational passing has no refusal', () => {
    const args: Arg[] = spread(8, 1, 1).map((c) => messageArg(gaussian, gaussianEta(c.m, c.v)))
    expect(() =>
      multiclassSite.ep?.({ mode: 'ep', to: 0, args, params: { y: 0, classes: 8, integration: 'product' }, power: 1 }),
    ).toThrow(/more than the six/)
    expect(() =>
      multiclassSite.ep?.({ mode: 'ep', to: 0, args, params: { y: 0, classes: 8, link: 'probit' }, power: 0.5 }),
    ).toThrow(/power of one/)
    expect(multiclassSite.unsupported?.('vmp', { y: 0, classes: 8 })).toBeUndefined()
    expect(multiclassSite.unsupported?.('ep', { y: 0, classes: 8 })).toBeUndefined()
  })
})

describe('variational message passing beyond six classes', () => {
  /** The beliefs of K scores: a null variance is a known score. */
  const beliefs = (list: readonly (readonly [number, number])[]): Arg[] =>
    list.map(([m, v]) => (v > 0 ? messageArg(gaussian, gaussianEta(m, v)) : valueArg(gaussian, m)))
  const vmp = (args: readonly Arg[], to: number, params: Params): readonly number[] =>
    multiclassSite.vmp?.({ mode: 'vmp', to, args, params, power: 1 }).message as readonly number[]
  const expectedLog = (args: readonly Arg[], params: Params): number =>
    multiclassSite.expectedLog?.({ mode: 'vmp', to: 0, args, params, power: 1 }) as number
  /** The reference of the expected log: a composite Gauss-Legendre rule over the free scores, the known ones fixed. */
  const reference = (args: readonly Arg[], log: (z: readonly number[]) => number, panels?: number): number => {
    const free = args.flatMap((a, i) => (a.value === undefined ? [i] : []))
    const brutes: BruteBelief[] = free.map((i) => ({
      kind: 'gaussian',
      eta: (args[i] as Arg).eta as readonly number[],
      panels: panels ?? (free.length > 2 ? 8 : 24),
    }))
    const z = args.map((a) => (a.value === undefined ? 0 : (a.value as number)))
    return bruteExpectation(brutes, (x) => {
      for (const [k, i] of free.entries()) z[i] = x[k] as number
      return log(z)
    })
  }
  /** The weights `h` and `J` as the natural gradient of the expected log: central differences in the mean and the variance of one belief. */
  const gradient = (list: readonly (readonly [number, number])[], to: number, params: Params): [number, number] => {
    const at = (m: number, v: number): number =>
      expectedLog(beliefs(list.map((b, i) => (i === to ? [m, v] : b))), params)
    const [m, v] = list[to] as readonly [number, number]
    const e = 1e-4
    const dm = (at(m + e, v) - at(m - e, v)) / (2 * e)
    const dv = (at(m, v + e) - at(m, v - e)) / (2 * e)
    return [dm - 2 * m * dv, -2 * dv]
  }
  const small: readonly (readonly [number, number])[] = [
    [0.3, 1.2],
    [-0.2, 0.8],
    [0.1, 1.5],
  ]
  // Eight classes, two of them random and six known: the product rule reaches this one, so the references are exact.
  const eight: readonly (readonly [number, number])[] = [
    [0.4, 1.1],
    [-0.3, 0.7],
    [2, 0],
    [-1, 0],
    [0.5, 0],
    [0, 0],
    [1, 0],
    [-2, 0],
  ]

  it('the softmax expected log is the exact expectation: three free scores and eight with six known, 1e-8', () => {
    for (const [list, y] of [
      [small, 1],
      [eight, 2],
      [eight, 0],
    ] as const) {
      const args = beliefs(list)
      const params = { y, classes: list.length, link: 'logit', integration: 'auxiliary' }
      const ref = reference(args, (z) => logSoftmax(z, y))
      expect(Math.abs(expectedLog(args, params) - ref), `y = ${y}`).toBeLessThan(1e-8)
    }
  })

  it('the softmax messages are those of the product grid (three and eight classes), whichever class is observed', () => {
    for (const [list, y] of [
      [small, 0],
      [small, 2],
      [eight, 1],
      [eight, 4],
    ] as const) {
      const args = beliefs(list)
      const base = { y, classes: list.length, link: 'logit' }
      for (const [to, [, v]] of list.entries()) {
        if (!(v > 0)) continue
        const grid = vmp(args, to, { ...base, integration: 'product' })
        const aux = vmp(args, to, { ...base, integration: 'auxiliary' })
        expect(Math.abs((aux[0] as number) - (grid[0] as number)), `h of ${to}, y = ${y}`).toBeLessThan(2e-5)
        expect(Math.abs((aux[1] as number) - (grid[1] as number)), `J of ${to}, y = ${y}`).toBeLessThan(2e-5)
      }
    }
  })

  it('the messages are the natural gradient of the reported expected log (softmax and probit, eight classes)', () => {
    const wide: readonly (readonly [number, number])[] = [
      [0.4, 1.1],
      [-0.3, 0.7],
      [0.2, 0.9],
      [-1, 0],
      [0.5, 0],
      [0, 0],
      [1, 0],
      [-0.5, 0],
    ]
    for (const link of ['logit', 'probit'] as const)
      for (const y of [0, 5]) {
        const params = { y, classes: 8, link }
        const args = beliefs(wide)
        for (const to of [0, 1, 2]) {
          const [h, j] = gradient(wide, to, params) as [number, number]
          const got = vmp(args, to, params)
          expect(Math.abs((got[0] as number) - h), `${link} y = ${y} h of ${to}`).toBeLessThan(2e-4)
          expect(Math.abs((got[1] as number) - j), `${link} y = ${y} J of ${to}`).toBeLessThan(2e-4)
        }
      }
  })

  it('the probit bound is below the expected log (0.004 to 0.02 nats with one other score random) and exact when the others are known', () => {
    for (const [list, y] of [
      [eight, 2],
      [eight, 0],
      [eight, 1],
    ] as const) {
      const args = beliefs(list)
      const bound = expectedLog(args, { y, classes: 8, link: 'probit' })
      // Two random scores: the default 24 panels of the product rule and 160 nodes of the noise integral take 25 s a reference; 8 panels and
      // 64 nodes take 1 s and agree to 6e-12 (the bound is compared to it to 1e-9 above and to 0.03 nats below).
      const ref = reference(args, (z) => logProbit(z, y, 64), 8)
      expect(bound).toBeLessThanOrEqual(ref + 1e-9)
      expect(ref - bound, `y = ${y}`).toBeLessThan(0.03)
    }
    // The other scores known, the observed class's score random: Jensen's inequality has nothing to bound, so the bound is the expected log.
    const one = beliefs([[0.4, 1.1], ...eight.slice(2)])
    const bound = expectedLog(one, { y: 0, classes: 7, link: 'probit' })
    expect(Math.abs(bound - reference(one, (z) => logProbit(z, 0)))).toBeLessThan(1e-6)
  })

  it('a belief with no density is refused, not sent a message', () => {
    expect(() => vmp([messageArg(gaussian, [0, -1]), ...beliefs(eight.slice(1))], 1, { y: 0, classes: 8 })).toThrow()
  })
})
