/**
 * The parameters of a node as the mappings read them: a distribution parameter as a symbolic value (with the family's default
 * where the model gives none), as known numbers over the node's dimensions, the value an observed node observes, the cells of an
 * observed node's parameter over its dimensions (made once and kept, so every route that reads the parameter sees the same cells),
 * the discrete variables a node depends on, the variable of a discrete latent, and the gamma precisions behind a random scale.
 * Shared by the mappings; none of them makes a mapping decision.
 */

import { asTensor, toFlat } from '../../../tensor/index.ts'
import type { NamedValue } from '../operators.ts'
import { type DistSpec, type ExprSpec, type ModelNode, nodeRefs } from '../spec.ts'
import { type Cell, isConstCell, type Sym, scaledOf } from './_cells.ts'
import type { Context } from './_context.ts'
import { precisionExpr } from './_idioms.ts'
import {
  observedValueRandom,
  parameterMissingIn,
  precisionNotGammaLatent,
  precisionNotKnownMultiple,
  precisionNotNonNegative,
  priorParameterRandom,
  scaleNotPrecisionIdiomWithIdiom,
  siteHasNoDiscrete,
  valueOverDimDoesNotFit,
} from './_reasons.ts'
import { Refusal } from './_refusal.ts'
import type { EvalMode } from './_symbolic/_affine.ts'
import { evalSym } from './_symbolic/_affine.ts'

export function paramSym(cx: Context, dist: DistSpec, name: string, mode: EvalMode = 'plain'): Sym {
  const e = dist.params[name]
  if (e !== undefined) return evalSym(cx, e, mode)
  const d = cx.families[dist.family]?.defaults[name]
  if (d === undefined) throw parameterMissingIn(dist.family, name)
  return cx.fromNamed({ value: asTensor(d, 'float64'), dims: [] })
}

export function constParam(
  cx: Context,
  node: ModelNode & { readonly dist: DistSpec },
  name: string,
  what: string,
): number[] {
  const sym = paramSym(cx, node.dist, name)
  return cx.over(sym, node.dims).map((c) => {
    if (!isConstCell(c)) throw priorParameterRandom(name, node.name, what)
    return c.aff.offset
  })
}

/**
 * The gamma variables that are the precisions of a Gaussian node whose scale is random, element by element: the scale
 * must be the idiom `1 / sqrt(tau)` of a gamma (or exponential) latent. Anything else is refused with the idiom named.
 */
export function randomPrecisions(
  cx: Context,
  node: ModelNode & { readonly dist: DistSpec },
  param = 'scale',
): { readonly variable: string; readonly scale: number; readonly offset: number }[] {
  const scale = node.dist.params[param]
  const idiom =
    'write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only)'
  const tau = scale === undefined ? undefined : precisionExpr(cx.idiom, scale)
  if (tau === undefined) throw scaleNotPrecisionIdiomWithIdiom(param, node.name, idiom)
  let sym: Sym
  try {
    sym = evalSym(cx, tau)
  } catch (e) {
    if (e instanceof Refusal && e.kind === 'operator') throw precisionNotKnownMultiple(node.name, e.message, idiom)
    throw e
  }
  return cx.over(sym, node.dims).map((c) => {
    const scaled = scaledOf(c)
    if (scaled === undefined)
      throw precisionNotGammaLatent(
        node.name,
        c.kind === 'rv' ? `${c.family} variable` : 'value computed from other latents',
        idiom,
      )
    if (!(scaled.scale >= 0 && scaled.offset >= 0 && scaled.scale + scaled.offset > 0))
      throw precisionNotNonNegative(node.name, scaled.scale, scaled.offset)
    return scaled
  })
}

/** The vector at one element of a const parameter that carries an event axis (a Dirichlet's concentrations). */
export function eventVector(_cx: Context, nv: NamedValue, dims: readonly string[], idx: readonly number[]): number[] {
  const k = nv.value.shape[nv.value.shape.length - 1] as number
  let flat = 0
  nv.dims.forEach((d, axis) => {
    const at = dims.indexOf(d)
    if (at < 0) throw valueOverDimDoesNotFit(d, dims.join(', '))
    flat = flat * (nv.value.shape[axis] as number) + (idx[at] as number)
  })
  return toFlat(nv.value).slice(flat * k, (flat + 1) * k)
}

/** The variable of a scalar discrete latent. */
export function rvName(cx: Context, site: string): string {
  const cell = cx.syms.get(site)?.cells[0]
  if (cell?.kind !== 'rv') throw siteHasNoDiscrete(site)
  return cell.variable
}

/** The discrete latents (Bernoulli) a node's parameters depend on, through derived nodes. */
export function discreteParents(cx: Context, node: ModelNode): string[] {
  const found: string[] = []
  const seen = new Set<string>()
  const visit = (name: string): void => {
    if (seen.has(name)) return
    seen.add(name)
    const n = cx.nodes.get(name)
    if (n === undefined) return
    if (n.kind === 'latent' && n.dist.family === 'bernoulli' && cx.syms.has(name)) found.push(name)
    else if (n.kind === 'derived') for (const r of nodeRefs(n)) visit(r)
  }
  for (const r of nodeRefs(node)) visit(r)
  return found
}

export function evalNamedValue(cx: Context, node: ModelNode & { readonly kind: 'observed' }): NamedValue {
  try {
    return cx.evalNamed(node.value)
  } catch (e) {
    if (e instanceof Refusal && e.kind === 'node') throw observedValueRandom(node.name, e.message)
    throw e
  }
}

/** The cells of a distribution parameter or a link's argument of an observed node over its dimensions, made once. */
export function cellsOf(cx: Context, node: ModelNode, key: string, make: () => Sym): Cell[] {
  const id = `${node.name}|${key}`
  let cells = cx.cells.get(id)
  if (cells === undefined) {
    cells = cx.over(make(), node.dims)
    cx.cells.set(id, cells)
  }
  return cells
}

/** The cell of a parameter of an observed node at element `i`. */
export function paramCell(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  name: string,
  i: number,
  mode: EvalMode = 'plain',
): Cell {
  return cellsOf(cx, node, name, () => paramSym(cx, node.dist, name, mode))[i] as Cell
}

/** The value of a known expression at element `i` of an observed node. */
export function constAt(cx: Context, node: ModelNode, e: ExprSpec, i: number): number {
  return (cx.over(cx.fromNamed(cx.evalNamed(e)), node.dims)[i] as Cell & { kind: 'aff' }).aff.offset
}

/** Whether a parameter of a node's distribution is known (a default counts). */
export function paramIsConst(cx: Context, node: ModelNode & { readonly dist: DistSpec }, name: string): boolean {
  const e = node.dist.params[name]
  return e === undefined || cx.isConst(e)
}
