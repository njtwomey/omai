/**
 * The graph structure that arithmetic on random Gaussian values makes: the `product` factor of two random values, a unary function
 * as an `apply` factor (with the chain of steps behind each output, so a known affine map of an output extends the chain instead of
 * approximating twice), the quotient by a chain, a unary function of a gamma precision, and the larger or smaller of two values.
 * Each makes a variable named after the site being compiled and a factor, and returns the affine form of that variable.
 */

import type { Json } from '../../../../base/index.ts'
import { unaryFunctions } from '../../../../message-passing/index.ts'
import { type Aff, type Atom, combine, constAff, type Precision, scaleAff } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { extremumOfItself, randomTimesItself } from '../_reasons.ts'

/** One step of an `apply` chain: a function of the registry by name, or `mul`, `add`, `pow` with its constant. */
export type ApplyStep = readonly [string, number?]

/**
 * The product of two affine forms: a scaling when one is known, else the `product` factor of the engine on a variable for
 * each (made by `materialise`), whose output is a new Gaussian variable (the moment-matched product of two Gaussians:
 * expectation propagation only).
 */
export function mulAff(cx: Context, a: Aff, b: Aff, index: number): Aff {
  if (a.terms.size === 0) return scaleAff(b, a.offset)
  if (b.terms.size === 0) return scaleAff(a, b.offset)
  const site = cx.current
  const left = cx.g.materialise(a, site, undefined) as string
  const right = cx.g.materialise(b, site, undefined) as string
  if (left === right) throw randomTimesItself(left)
  const out = cx.g.auxName(site, index)
  cx.g.need('product')
  cx.g.addVariable({ name: out, family: { kind: 'gaussian' } })
  cx.g.addFactor({ operator: 'product', args: [out, left, right] }, out)
  const key = cx.g.atomKey({ kind: 'var', name: out })
  cx.g.atoms.set(key, { kind: 'var', name: out })
  return { terms: new Map([[key, 1]]), offset: 0 }
}

/**
 * A unary function of an affine value: a known value is evaluated; a random one becomes a new Gaussian variable defined
 * by the `apply` factor of the engine (`exp`, `square`, `tanh`, `softplus`, `abs`, `sigmoid`, `relu`, `sqrt`, the reciprocal:
 * the output matched by moments under the input's cavity, expectation propagation only). When the value is a known affine
 * map of the output of an earlier `apply` (`sqrt(d exp(x) + n)`), the new factor is the whole chain from the earlier one's
 * input, not a second Gaussian approximation on top of the first.
 */
export function applyAff(cx: Context, fn: string, a: Aff, index: number): Aff {
  if (a.terms.size === 0) return constAff((unaryFunctions()[fn] as { f: (x: number) => number }).f(a.offset))
  return applyChain(cx, chainOf(cx, a), [fn], index)
}

/**
 * `clip(a, lo, hi)` of a random value for known bounds: a function of the one variable behind `a`, so the chain of its `apply` factor
 * gets a `min` and a `max` step (a known value is clipped on the spot). No second Gaussian approximation: the clip is inside the chain.
 */
export function clipAff(cx: Context, a: Aff, lo: number, hi: number, index: number): Aff {
  if (a.terms.size === 0) return constAff(Math.min(Math.max(a.offset, lo), hi))
  return applyChain(
    cx,
    chainOf(cx, a),
    [
      ['min', hi],
      ['max', lo],
    ],
    index,
  )
}

/**
 * `numerator / divisor` for a divisor that is a known affine map of the output of an `apply` (a function of a latent): the
 * `apply-product` factor of the numerator and the divisor's chain with the reciprocal appended. Undefined for any other divisor.
 */
export function quotientByChain(cx: Context, numerator: Aff, divisor: Aff, index: number): Aff | undefined {
  const terms = [...divisor.terms].filter(([, c]) => c !== 0)
  const atom = terms.length === 1 ? (cx.g.atoms.get(terms[0]?.[0] as string) as Atom) : undefined
  if (atom?.kind !== 'var' || !cx.applied.has(atom.name)) return undefined
  const base = chainOf(cx, divisor)
  const steps = [...base.steps, ['recip'] as ApplyStep]
  if (numerator.terms.size === 0) return scaleAff(applyChain(cx, base, ['recip'], index), numerator.offset)
  const z = cx.g.materialise(numerator, cx.current, undefined) as string
  const out = cx.g.auxName(cx.current, index)
  cx.g.need('apply-product')
  cx.g.addVariable({ name: out, family: { kind: 'gaussian' } })
  cx.g.addFactor(
    { operator: 'apply-product', args: [out, z, base.input], params: { steps: steps as unknown as Json } },
    out,
  )
  const key = cx.g.atomKey({ kind: 'var', name: out })
  cx.g.atoms.set(key, { kind: 'var', name: out })
  return { terms: new Map([[key, 1]]), offset: 0 }
}

/** A known multiple of a gamma variable plus an offset (a precision) through a unary function: an `apply` factor on the gamma variable. */
export function applyGamma(cx: Context, fn: string, g: Precision, index: number): Aff {
  const steps: ApplyStep[] = []
  if (g.scale !== 1) steps.push(['mul', g.scale])
  if (g.offset !== 0) steps.push(['add', g.offset])
  return applyChain(cx, { input: g.variable, steps }, [fn], index)
}

/**
 * The variable an affine value is read from and the steps from it: the output of an earlier `apply` under a known affine map is
 * followed back to that factor's input (its steps, then the map), any other value is a variable of its own (no steps).
 */
export function chainOf(cx: Context, a: Aff): { readonly input: string; readonly steps: readonly ApplyStep[] } {
  const terms = [...a.terms].filter(([, c]) => c !== 0)
  const [key, coefficient] = terms[0] as [string, number]
  const atom = terms.length === 1 ? (cx.g.atoms.get(key) as Atom) : undefined
  const earlier = atom?.kind === 'var' ? cx.applied.get(atom.name) : undefined
  if (earlier === undefined) return { input: cx.g.materialise(a, cx.current, undefined) as string, steps: [] }
  const steps: ApplyStep[] = [...earlier.steps]
  if (coefficient !== 1) steps.push(['mul', coefficient])
  if (a.offset !== 0) steps.push(['add', a.offset])
  return { input: earlier.input, steps }
}

export function applyChain(
  cx: Context,
  base: { readonly input: string; readonly steps: readonly ApplyStep[] },
  more: readonly (string | ApplyStep)[],
  index: number,
): Aff {
  const steps = [...base.steps, ...more.map((m): ApplyStep => (typeof m === 'string' ? [m] : m))]
  const site = cx.current
  const out = cx.g.auxName(site, index)
  cx.g.need('apply')
  cx.g.addVariable({ name: out, family: { kind: 'gaussian' } })
  const single = steps.length === 1 && steps[0]?.[1] === undefined ? steps[0]?.[0] : undefined
  cx.g.addFactor(
    {
      operator: 'apply',
      args: [out, base.input],
      params: single === undefined ? { steps: steps as unknown as Json } : { fn: single },
    },
    out,
  )
  cx.applied.set(out, { input: base.input, steps })
  const key = cx.g.atomKey({ kind: 'var', name: out })
  cx.g.atoms.set(key, { kind: 'var', name: out })
  return { terms: new Map([[key, 1]]), offset: 0 }
}

/** The larger or smaller of two random values: the `max` or `min` factor of the engine on a new variable (expectation propagation, Clark's moments). */
export function extremum(cx: Context, operator: 'max' | 'min', a: Aff, b: Aff, index: number): Aff {
  if (a.terms.size === 0 || b.terms.size === 0) {
    // max(x, k) = k + relu(x - k) and min(x, k) = x - relu(x - k): a rectifier of an affine value (the `apply` factor).
    const [k, other] = a.terms.size === 0 ? [a.offset, b] : [b.offset, a]
    const r = applyAff(cx, 'relu', combine(other, constAff(k), -1), index)
    return operator === 'max' ? combine(r, constAff(k), 1) : combine(other, r, -1)
  }
  const site = cx.current
  const left = cx.g.materialise(a, site, undefined) as string
  const right = cx.g.materialise(b, site, undefined) as string
  if (left === right) throw extremumOfItself(operator === 'max' ? 'maximum' : 'minimum')
  const out = cx.g.auxName(site, index)
  cx.g.need(operator)
  cx.g.addVariable({ name: out, family: { kind: 'gaussian' } })
  cx.g.addFactor({ operator, args: [out, left, right] }, out)
  const key = cx.g.atomKey({ kind: 'var', name: out })
  cx.g.atoms.set(key, { kind: 'var', name: out })
  return { terms: new Map([[key, 1]]), offset: 0 }
}
