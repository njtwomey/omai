/**
 * The conjugate pairs inside gates. A component of a mixture that has a mean `M` and a precision `T` of its own, with the mean's
 * prior scaled by the precision (`M ~ N(m, 1 / (kappa T))`, `T ~ Gamma(a, b)`: the classic normal-gamma prior) and the component's
 * observations `N(y; M, 1 / (c T))` read as the branch of a `gate` on the label, compiles to a `gaussian-gamma-precision` factor
 * for the prior of the mean and one such branch per observation, whose messages are quadratures over the precision. The same
 * model is one `normal-gamma` variable per component, a `prior` on it and a `gaussian-normal-gamma` branch per observation, whose
 * messages are closed form and exact: this pass rewrites the first graph into the second where the pairs satisfy the conjugate
 * form (the mean and the precision have no other factor, the offset of every precision is zero), and leaves everything else as it
 * was. It reads the finished graph, so it needs no idiom of the model language: any way of writing the pair that compiles to
 * these factors is fused. It is for expectation propagation (a variational rule has no expected log for the gate over the new
 * branches); `toFactorGraph` runs it only when asked (`conjugatePairs`).
 */

import type { FactorSpec, GateBranch, VariableSpec } from '../../../message-passing/index.ts'
import { normalGammaEta } from '../../../message-passing/index.ts'
import type { Context } from './_context.ts'
import type { SiteLayout } from './types.ts'

/** A pair of variables that is a normal-gamma pair: the mean with its conditional prior, the precision with its gamma prior. */
interface Pair {
  readonly mean: string
  readonly precision: string
  /** The factors that define the pair (the conditional prior of the mean, the gamma prior of the precision), by index. */
  readonly meanPrior: number
  readonly precisionPrior: number
  /** The prior mean `m`, the precision multiple `kappa` of the mean's variance, and the gamma's shape and rate. */
  readonly m: number
  readonly kappa: number
  readonly shape: number
  readonly rate: number
}

type Arg = FactorSpec['args'][number]

const numberOf = (a: Arg | undefined): number | undefined =>
  typeof a === 'object' && a !== null && typeof a.const === 'number' ? a.const : undefined

/** The `scale` and `offset` of a `gaussian-gamma-precision` factor or branch (the precision is `scale tau + offset`). */
function precisionParams(params: FactorSpec['params']): { readonly scale: number; readonly offset: number } {
  const p = params ?? {}
  return {
    scale: typeof p['scale'] === 'number' ? p['scale'] : 1,
    offset: typeof p['offset'] === 'number' ? p['offset'] : 0,
  }
}

/** The candidate pairs of a graph: each Gaussian variable whose one prior is `gaussian-gamma-precision` on a gamma variable that has a gamma prior only. */
function candidatesOf(factors: readonly FactorSpec[], families: ReadonlyMap<string, VariableSpec>): Pair[] {
  const priorOf = new Map<string, number>()
  const gammaPriors = new Map<string, number[]>()
  const uses = new Map<string, number[]>()
  factors.forEach((f, i) => {
    for (const a of f.args) if (typeof a === 'string') uses.set(a, [...(uses.get(a) ?? []), i])
    if (f.operator === 'prior' && typeof f.args[0] === 'string') {
      const v = families.get(f.args[0])
      if (v?.family.kind === 'gamma') gammaPriors.set(f.args[0], [...(gammaPriors.get(f.args[0]) ?? []), i])
    }
    if (f.operator === 'gaussian-gamma-precision' && typeof f.args[0] === 'string' && typeof f.args[2] === 'string') {
      const v = families.get(f.args[0])
      if (v?.family.kind === 'gaussian' && v.observed === undefined) priorOf.set(f.args[0], i)
    }
  })
  const pairs: Pair[] = []
  const claimed = new Map<string, number>()
  for (const [mean, at] of priorOf) {
    const f = factors[at] as FactorSpec
    const m = numberOf(f.args[1])
    const precision = f.args[2] as string
    const { scale, offset } = precisionParams(f.params)
    const gammaAt = gammaPriors.get(precision)
    if (m === undefined || offset !== 0 || !(scale > 0) || gammaAt?.length !== 1) continue
    // The mean's prior is its only factor outside the gates (the gate's branches are checked where they are rewritten), and the
    // precision has its gamma prior and nothing else outside the gates.
    const loose = (name: string, own: readonly number[]): boolean =>
      (uses.get(name) ?? []).some((i) => !own.includes(i) && factors[i]?.operator !== 'gate')
    if (loose(mean, [at]) || loose(precision, [...gammaAt, at])) continue
    const eta = (factors[gammaAt[0] as number] as FactorSpec).params?.['eta']
    if (!Array.isArray(eta) || typeof eta[0] !== 'number' || typeof eta[1] !== 'number') continue
    claimed.set(precision, (claimed.get(precision) ?? 0) + 1)
    pairs.push({
      mean,
      precision,
      meanPrior: at,
      precisionPrior: gammaAt[0] as number,
      m,
      kappa: scale,
      shape: eta[0] + 1,
      rate: -eta[1],
    })
  }
  // A precision shared by several means is not a pair of any of them.
  return pairs.filter((p) => claimed.get(p.precision) === 1)
}

/** The branches of a gate factor. */
const branchesOf = (f: FactorSpec): GateBranch[] => (f.params?.['branches'] ?? []) as unknown as GateBranch[]

/**
 * Whether every use of the pair inside the gates is a `gaussian-gamma-precision` branch of one observation on the mean and the
 * precision together, with a precision that is a multiple of the variable (no offset). A gate that reads only one of the two, or
 * through another operator, rules the pair out.
 */
function gatesAgree(pair: Pair, factors: readonly FactorSpec[]): { readonly ok: boolean; readonly gates: number } {
  let gates = 0
  for (const f of factors) {
    if (f.operator !== 'gate') continue
    const vars = f.args.slice(1)
    const slotOfMean = vars.indexOf(pair.mean)
    const slotOfPrecision = vars.indexOf(pair.precision)
    if (slotOfMean < 0 && slotOfPrecision < 0) continue
    if (slotOfMean < 0 || slotOfPrecision < 0) return { ok: false, gates }
    gates++
    for (const b of branchesOf(f)) {
      const mentions = b.args.filter((a): a is number => typeof a === 'number')
      const reads = mentions.some((a) => a === slotOfMean || a === slotOfPrecision)
      if (!reads) continue
      const [y, mean, precision] = b.args
      const { offset } = precisionParams(b.params)
      if (
        b.operator !== 'gaussian-gamma-precision' ||
        typeof y !== 'object' ||
        typeof y.const !== 'number' ||
        mean !== slotOfMean ||
        precision !== slotOfPrecision ||
        offset !== 0
      )
        return { ok: false, gates }
    }
  }
  return { ok: true, gates }
}

/**
 * Rewrite the conjugate pairs inside gates as normal-gamma variables, in place: each pair becomes one variable of the family
 * `normal-gamma` (under the mean's name) with a `prior` factor, each gate branch on the pair a `gaussian-normal-gamma` branch, and the
 * sites of the mean and of the precision read the one variable (`part` `mean` and `precision`).
 *
 * @param cx - The compilation, whose graph and sites are rewritten.
 * @returns The number of pairs fused.
 */
export function fuseConjugatePairs(cx: Context): number {
  const variables = cx.g.variables
  const factors = cx.g.factors
  const byName = new Map(variables.map((v) => [v.name, v] as const))
  const pairs = candidatesOf(factors, byName).filter((p) => {
    const agree = gatesAgree(p, factors)
    return agree.ok && agree.gates > 0
  })
  if (pairs.length === 0) return 0
  const fused = new Map(pairs.map((p) => [p.mean, p] as const))
  const meanOfPrecision = new Map(pairs.map((p) => [p.precision, p.mean] as const))
  const dropped = new Set(pairs.flatMap((p) => [p.meanPrior, p.precisionPrior]))
  const out: FactorSpec[] = []
  for (const [i, f] of factors.entries()) {
    const pair = pairs.find((p) => p.meanPrior === i)
    if (pair !== undefined) {
      out.push({
        operator: 'prior',
        args: [pair.mean],
        params: { eta: normalGammaEta(pair.m, pair.kappa, pair.shape, pair.rate) as unknown as number[] },
        ...(f.id === undefined ? {} : { id: f.id }),
      })
      continue
    }
    if (dropped.has(i)) continue
    out.push(f.operator === 'gate' ? rewriteGate(f, fused, meanOfPrecision) : f)
  }
  factors.splice(0, factors.length, ...out)
  const kept = variables.filter((v) => !meanOfPrecision.has(v.name))
  variables.splice(
    0,
    variables.length,
    ...kept.map((v): VariableSpec => (fused.has(v.name) ? { name: v.name, family: { kind: 'normal-gamma' } } : v)),
  )
  cx.sites.splice(0, cx.sites.length, ...cx.sites.map((s) => siteOf(s, fused, meanOfPrecision)))
  return pairs.length
}

/** A gate with the branches on fused pairs rewritten and the precision variables of the pairs dropped from its variables. */
function rewriteGate(
  f: FactorSpec,
  fused: ReadonlyMap<string, Pair>,
  meanOfPrecision: ReadonlyMap<string, string>,
): FactorSpec {
  const vars = f.args.slice(1)
  if (!vars.some((v) => typeof v === 'string' && meanOfPrecision.has(v))) return f
  // The new position of each old variable: the precisions of fused pairs go, the rest keep their order.
  const position = new Map<number, number>()
  const kept: Arg[] = []
  vars.forEach((v, i) => {
    if (typeof v === 'string' && meanOfPrecision.has(v)) return
    position.set(i, kept.length)
    kept.push(v as Arg)
  })
  const branches = branchesOf(f).map((b): GateBranch => {
    const [y, mean, precision] = b.args
    const precisionName = typeof precision === 'number' ? vars[precision] : undefined
    const meanName = typeof mean === 'number' ? vars[mean] : undefined
    if (
      b.operator === 'gaussian-gamma-precision' &&
      typeof meanName === 'string' &&
      typeof precisionName === 'string' &&
      fused.has(meanName) &&
      meanOfPrecision.get(precisionName) === meanName
    ) {
      const { scale } = precisionParams(b.params)
      return {
        operator: 'gaussian-normal-gamma',
        args: [y as { readonly const: number }, position.get(mean as number) as number],
        params: scale === 1 ? {} : { precision: scale },
      }
    }
    return {
      ...b,
      args: b.args.map((a) => (typeof a === 'number' ? (position.get(a) as number) : a)),
    }
  })
  return {
    ...f,
    args: [f.args[0] as Arg, ...kept],
    params: { ...f.params, arity: kept.length, branches: branches as unknown as never },
  }
}

/** A site read from the variables of the fused pairs: the mean's site and the precision's now share the variable of the pair. */
function siteOf(
  s: SiteLayout,
  fused: ReadonlyMap<string, Pair>,
  meanOfPrecision: ReadonlyMap<string, string>,
): SiteLayout {
  const names = s.variables.filter((v): v is string => v !== null)
  if (names.length > 0 && names.every((v) => fused.has(v))) return { ...s, family: 'normal-gamma', part: 'mean' }
  if (names.length > 0 && names.every((v) => meanOfPrecision.has(v)))
    return {
      ...s,
      family: 'normal-gamma',
      part: 'precision',
      variables: s.variables.map((v) => (v === null ? null : (meanOfPrecision.get(v) as string))),
    }
  return s
}
