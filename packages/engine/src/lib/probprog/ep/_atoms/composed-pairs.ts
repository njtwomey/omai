/**
 * A mixture whose groups have a mean and a precision of their own, composed in one small setup: a label per row, a gamma precision
 * and a centre as wide as its precision allows (the conjugate normal-gamma pair) per group and direction, and each direction of a
 * row a gate over the pairs of the groups. The compiler fuses each pair into one normal-gamma variable, so the messages are closed
 * form. The reference enumerates every assignment of the rows to the groups, each a conjugate update.
 */

import { lgammaOf } from '../../../message-passing/numerics/_scalar.ts'
import { foldIn, key, normal, uniform } from '../../../random/index.ts'
import { dict, int, real } from '../../../space/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import { type ExactMarginal, probabilityMarginal } from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { tolerance } from '../_tolerance.ts'
import type { AtomParts, EpAtom } from '../atom-types.ts'

/** The prior of every pair: the centre of group k starts at `STARTS[k]` and is as wide as `KAPPA` times the precision, the precision is Gamma(SHAPE, RATE). */
const KAPPA = 0.2
const SHAPE = 2
const RATE = 0.5
const STARTS = [-1, 1] as const

/** Two clusters in two directions: the group of each row, and the two directions with a spread of their own per group. */
function simulate(seed: number, n: number, gap: number) {
  const k = key(seed)
  const u = toFlat(uniform(foldIn(k, 1), [n]))
  const z = Array.from(u, (v) => (v < 0.5 ? 0 : 1))
  const noise = [toFlat(normal(foldIn(k, 2), [n])), toFlat(normal(foldIn(k, 3), [n]))]
  const sd = [
    [0.25, 0.4],
    [0.4, 0.3],
  ] as const
  const centre = (c: number, j: number): number => (c === 0 ? -gap / 2 : gap / 2) * (j === 0 ? 1 : 0.9)
  const length = z.map((c, i) => centre(c, 0) + (sd[c]?.[0] as number) * (noise[0]?.[i] as number))
  const wait = z.map((c, i) => centre(c, 1) + (sd[c]?.[1] as number) * (noise[1]?.[i] as number))
  return { z, length, wait }
}

/** A normal-gamma pair `(m, kappa, a, b)` after some points, and the log of the marginal likelihood of those points. */
interface Pair {
  readonly m: number
  readonly kappa: number
  readonly a: number
  readonly b: number
  readonly logMarginal: number
}

function update(x: readonly number[], m: number): Pair {
  const n = x.length
  const bar = n === 0 ? 0 : x.reduce((s, v) => s + v, 0) / n
  const scatter = x.reduce((s, v) => s + (v - bar) ** 2, 0)
  const kappa = KAPPA + n
  const a = SHAPE + n / 2
  const b = n === 0 ? RATE : RATE + scatter / 2 + (KAPPA * n * (bar - m) ** 2) / (2 * kappa)
  const logMarginal =
    n === 0
      ? 0
      : lgammaOf(a) -
        lgammaOf(SHAPE) +
        SHAPE * Math.log(RATE) -
        a * Math.log(b) +
        0.5 * (Math.log(KAPPA) - Math.log(kappa)) -
        (n / 2) * Math.log(2 * Math.PI)
  return { m: (KAPPA * m + n * bar) / kappa, kappa, a, b, logMarginal }
}

/** The marginal of a centre (a mixture of Student t) or of a precision (a mixture of gamma) over the weighted pairs: its moments and its density. */
function marginalOf(parts: readonly (Pair & { readonly w: number })[], of: 'centre' | 'precision'): ExactMarginal {
  const total = parts.reduce((s, p) => s + p.w, 0)
  const moments = parts.map((p) => {
    if (of === 'precision') return { w: p.w / total, mean: p.a / p.b, variance: p.a / p.b ** 2 }
    const df = 2 * p.a
    return { w: p.w / total, mean: p.m, variance: (p.b / (p.a * p.kappa)) * (df / (df - 2)) }
  })
  const mean = moments.reduce((s, q) => s + q.w * q.mean, 0)
  const sd = Math.sqrt(moments.reduce((s, q) => s + q.w * (q.variance + q.mean ** 2), 0) - mean * mean)
  const pdf = (x: number): number =>
    parts.reduce((s, p) => {
      const weight = p.w / total
      if (of === 'precision')
        return x <= 0
          ? s
          : s + weight * Math.exp(p.a * Math.log(p.b) - lgammaOf(p.a) + (p.a - 1) * Math.log(x) - p.b * x)
      const df = 2 * p.a
      const scale2 = p.b / (p.a * p.kappa)
      const q = (x - p.m) ** 2 / scale2
      return (
        s +
        weight *
          Math.exp(
            lgammaOf((df + 1) / 2) -
              lgammaOf(df / 2) -
              0.5 * Math.log(df * Math.PI * scale2) -
              ((df + 1) / 2) * Math.log(1 + q / df),
          )
      )
    }, 0)
  return { mean, sd, pdf, ...(of === 'precision' ? { range: [0, mean + 8 * sd] as const } : {}) }
}

interface Reference {
  readonly exact: Record<string, ExactMarginal>
  readonly logEvidence: number
}

/** The exact posterior by enumerating the 2^n assignments of the rows to the two groups, each a conjugate update of both directions. Computed when first read. */
function referenceOf(data: { readonly length: readonly number[]; readonly wait: readonly number[] }): () => Reference {
  let memo: Reference | undefined
  return () => {
    if (memo !== undefined) return memo
    const n = data.length.length
    const logw: number[] = []
    const fits: Record<'length' | 'wait', Pair[]>[] = []
    for (let code = 0; code < 2 ** n; code++) {
      const inside = (c: number): number[] => data.length.flatMap((_, i) => (((code >> i) & 1) === c ? [i] : []))
      const fit = (xs: readonly number[]): Pair[] =>
        [0, 1].map((c) =>
          update(
            inside(c).map((i) => xs[i] as number),
            STARTS[c] as number,
          ),
        )
      const f = { length: fit(data.length), wait: fit(data.wait) }
      fits.push(f)
      logw.push(n * Math.log(0.5) + [...f.length, ...f.wait].reduce((s, p) => s + p.logMarginal, 0))
    }
    const top = Math.max(...logw)
    const weights = logw.map((v) => Math.exp(v - top))
    const total = weights.reduce((s, v) => s + v, 0)
    const exact: Record<string, ExactMarginal> = {}
    for (const c of [0, 1])
      for (const [name, direction] of [
        ['Length', 'length'],
        ['Wait', 'wait'],
      ] as const) {
        const parts = fits.map((f, i) => ({ w: weights[i] as number, ...(f[direction][c] as Pair) }))
        exact[`centre${name}[${c}]`] = marginalOf(parts, 'centre')
        exact[`precision${name}[${c}]`] = marginalOf(parts, 'precision')
      }
    for (let i = 0; i < n; i++)
      exact[`which[${i}]`] = probabilityMarginal(
        weights.reduce((s, w, code) => s + (((code >> i) & 1) === 1 ? w : 0), 0) / total,
      )
    memo = { exact, logEvidence: top + Math.log(total) }
    return memo
  }
}

function problem(k: Config): AtomParts {
  const [seed, n, gap] = [configNumber(k, 'seed'), configNumber(k, 'n'), configNumber(k, 'gap')]
  const data = simulate(seed, n, gap)
  // region gate-normal-gamma-pairs
  const definition = defineModel('gate-normal-gamma-pairs', (m) => {
    const row = m.range('row')
    const group = m.range('group', { size: 2 })
    const length = m.data('length', { dims: [row] })
    const wait = m.data('wait', { dims: [row] })
    // where the centre of each group is expected to start
    const start = m.data('start', { dims: [group] })
    // the group of each row
    const which = m.latent('which', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
    // a precision of its own in each direction, for each group
    const precisionLength = m.latent('precisionLength', d.gamma(2, 0.5), { dims: [group] })
    const precisionWait = m.latent('precisionWait', d.gamma(2, 0.5), { dims: [group] })
    // a centre is as uncertain as its group is wide: its precision is a fifth of the group's
    const spreadLength = ops.div(1, ops.sqrt(ops.mul(precisionLength, 0.2)))
    const spreadWait = ops.div(1, ops.sqrt(ops.mul(precisionWait, 0.2)))
    const centreLength = m.latent('centreLength', d.normal(start, spreadLength), { dims: [group] })
    const centreWait = m.latent('centreWait', d.normal(start, spreadWait), { dims: [group] })
    // each row is in one group, whose centre and precision it follows
    const sdLength = ops.div(1, ops.sqrt(precisionLength.at(which)))
    const sdWait = ops.div(1, ops.sqrt(precisionWait.at(which)))
    m.observe('length', d.normal(centreLength.at(which), sdLength), length)
    m.observe('wait', d.normal(centreWait.at(which), sdWait), wait)
  })
  // endregion
  const ref = referenceOf(data)
  const quantity = (
    name: string,
  ): { readonly name: string; readonly label: string; readonly support: 'real' | 'positive' } => {
    const parts = /^(centre|precision)(Length|Wait)\[(\d)\]$/.exec(name)
    const [what, direction, group] = [parts?.[1] ?? '', parts?.[2] ?? '', Number(parts?.[3] ?? 0)]
    return {
      name,
      label: `${what} of the ${direction.toLowerCase()} of group ${group + 1}`,
      support: what === 'centre' ? 'real' : 'positive',
    }
  }
  const names = [0, 1].flatMap((c) => [
    `centreLength[${c}]`,
    `centreWait[${c}]`,
    `precisionLength[${c}]`,
    `precisionWait[${c}]`,
  ])
  return {
    build: {
      definition,
      data: { length: data.length, wait: data.wait, start: [...STARTS] },
      quantities: [
        ...names.map(quantity),
        ...data.z.map((_, i) => ({
          name: `which[${i}]`,
          label: `row ${i + 1} is in group 2`,
          support: 'probability' as const,
          hidden: true,
        })),
      ],
    },
    reference: {
      // lazy: the enumeration is summed when a reference is first asked for
      get exact() {
        return ref().exact
      },
      get exactLogEvidence() {
        return ref().logEvidence
      },
      referenceSource: `every one of the ${2 ** n} assignments of the rows to the two groups enumerated, each a conjugate update of the four pairs`,
    },
    present: {
      equations: [
        `${n} rows, seed ${seed}, the groups ${gap} apart`,
        'which_i ~ Categorical(1/2, 1/2)',
        'precision_jk ~ Gamma(2, 0.5), for each direction j and group k',
        `centre_jk ~ N(start_k, 1 / (${KAPPA} precision_jk)),  start = (${STARTS.join(', ')})`,
        'length_i ~ N(centre_{length, which_i}, 1 / precision_{length, which_i}), and wait_i likewise',
      ],
      figure: {
        caption: 'The rows, marked by the group each was drawn from.',
        xLabel: 'length',
        yLabel: 'wait',
        points: [
          { name: 'rows', x: data.length, y: data.wait, group: data.z, groupNames: { 0: 'group 1', 1: 'group 2' } },
        ],
      },
    },
  }
}

/**
 * A mixture of two groups in two directions whose centres and precisions are conjugate pairs: the label of each row is a
 * categorical, each group has a precision of its own in each direction and a centre as wide as that precision allows, and each
 * direction of a row is a gate over the pairs of the two groups. The compiler makes each pair one normal-gamma variable, so the
 * messages are closed form. The data are two simulated clusters; the reference enumerates every assignment of the rows.
 */
const gateNormalGammaPairsAtom: EpAtom = defineAtom({
  name: 'gate-normal-gamma-pairs',
  title: 'A mixture whose groups have a mean and a precision of their own',
  what: 'Two groups in two directions, each direction of each group a mean and a precision with the conjugate normal-gamma prior, and a gate on the label of each row; the compiler makes each pair one normal-gamma variable and EP reports the centres, the precisions and the probability that each row is in the second group.',
  factors: [
    'prior (normal-gamma x4)',
    'discrete (a categorical label per row)',
    'gate per row and direction (gaussian-normal-gamma branches)',
  ],
  family: 'composed',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    'measured on eight seeds of ten rows 1.2 apart: the centres and the precisions are within 0.2 reference sds of the exact ones with spreads 0.7 to 1.0 of them; the probability that a row is in the second group is within 0.2 sds of the exact one but too sure (its spread 0.3 to 1.0 of the exact), as EP is on the label of an ambiguous row',
    { meanZ: 0.3, sdRatio: [0.2, 1.6], ks: 0.2 },
  ),
  knobs: dict({
    seed: int(1, 9999, { default: 1, label: 'seed (the simulated rows)' }),
    n: int(6, 12, { default: 10, label: 'rows' }),
    gap: real(0.6, 4, { default: 1.2, step: 0.1, label: 'distance between the groups' }),
  }),
  ep: { tol: 1e-6 },
  parts: problem,
})

/** The atoms of the `composed-pairs` file. */
export const composedPairsAtoms: readonly EpAtom[] = [gateNormalGammaPairsAtom]
