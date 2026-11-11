/**
 * Derived nodes: the arithmetic of the value algebra evaluated for a node, kept as a symbolic value, a known constant, a lazy
 * expression over discrete variables or a recorded variable; the non-centred vector `z / S(x)` as one Gaussian vector with a
 * `gaussian-mv-function-scale` factor (`denseNonCentred`), and a vector of scalar products tied to one Gaussian vector variable.
 */

import type { Json } from '../../../../base/index.ts'
import { elementId } from '../../ids.ts'
import type { ModelNode } from '../../spec.ts'
import { type Cell, isConstCell, prod } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { discreteParents } from '../_params.ts'
import { Refusal } from '../_refusal.ts'
import { evalSym } from '../_symbolic/_affine.ts'
import type { ApplyStep } from '../_symbolic/_functions.ts'
import { chainOf } from '../_symbolic/_functions.ts'

export function derived(
  cx: Context,
  node: ModelNode & { readonly kind: 'derived' },
  compileLatent: (node: ModelNode & { readonly kind: 'latent' }) => void,
): void {
  const absorbed = cx.nonCentred.get(node.name)
  let scalars = false
  if (absorbed !== undefined) {
    if (denseNonCentred(cx, node)) return
    scalars = true
    // The dense factor does not apply to this divisor: the standard normals are compiled as the scalar variables they would have been.
    const z = cx.nodes.get(absorbed) as ModelNode & { readonly kind: 'latent' }
    cx.nonCentred.delete(node.name)
    cx.withSite(z.name, () => compileLatent(z))
  }
  if (cx.known.has(node.name)) {
    // A constant found before the walk: its value is known, and the custom functions its evaluation called count as evaluated.
    for (const name of cx.customsCalled.get(node.name) ?? []) cx.handledCustoms.add(name)
    cx.layout(node, 'derived', undefined, 'constant', new Array<null>(prod(cx.shapeOf(node.dims))).fill(null), false)
    return
  }
  if (discreteParents(cx, node).length > 0) {
    cx.lazy.set(node.name, node.expr)
    return
  }
  const sym = evalSym(cx, node.expr)
  const shape = cx.shapeOf(node.dims)
  const cells = cx.over(sym, node.dims)
  if (cells.every(isConstCell)) {
    // Constant only after the symbolic evaluation (the expression mentions a random value, the cells do not): order-dependent.
    cx.known.declare(node.name, cx.toNamed({ dims: node.dims, cells }), cx.position)
    cx.known.semantic.push(node.name)
    cx.layout(node, 'derived', undefined, 'constant', new Array<null>(prod(shape)).fill(null), false)
    return
  }
  if (scalars && tiedToVector(cx, node, cells)) return
  // Stick-breaking weights are not a random variable of the graph (the assignments read the breaks through one factor), so there is
  // nothing to record: the weights are read back from the breaks.
  if (!node.record || cells.some((c) => c.kind === 'sticks')) {
    cx.syms.set(node.name, { dims: node.dims, cells })
    return
  }
  const vars: (string | null)[] = []
  const out: Cell[] = cells.map((c, i) => {
    if (c.kind === 'rv') {
      vars.push(c.variable)
      return c
    }
    if (c.kind === 'sticks') throw new Refusal('structure', `'${node.name}' is a stick-breaking weight vector`)
    if (c.kind === 'scaled' || c.kind === 'lookup' || c.kind === 'recip')
      throw new Refusal(
        'structure',
        `'${node.name}' is a ${c.kind === 'scaled' ? 'scaled precision' : c.kind === 'recip' ? 'multiple of an inverse-gamma variance' : 'lookup by a discrete variable'}: it cannot be recorded as a value`,
      )
    const name = cx.elementName(node.name, node.dims, i)
    const v = cx.g.materialise(c.aff, node.name, node.dims.length === 0 ? undefined : i, name)
    if (typeof v === 'number') {
      vars.push(null)
      return c
    }
    vars.push(v)
    return cx.g.varAtom(v)
  })
  cx.finishLatent(node, 'gaussian', out, vars, {}, 'derived')
}

/**
 * `D = z / S(x)` as one Gaussian vector variable `D` and a `gaussian-mv-function-scale` factor with the variable `x` the divisor
 * is a chain of unary steps of (`1 / sqrt(exp(x) d_j + n_j)`: each coefficient's chain, a known divisor a constant). The standard
 * normals `z` never become variables: given `x` the vector is Gaussian with the diagonal covariance `1 / S_j(x)^2`, which the
 * factor integrates against the cavity of the whole vector. False (and nothing made) when the divisors are not chains of one variable.
 */
export function denseNonCentred(cx: Context, node: ModelNode & { readonly kind: 'derived' }): boolean {
  const divisor = node.expr.kind === 'op' ? node.expr.args[1] : undefined
  if (divisor === undefined) return false
  let cells: Cell[]
  try {
    cells = cx.over(evalSym(cx, divisor), node.dims)
  } catch (e) {
    if (e instanceof Refusal) return false
    throw e
  }
  let input: string | undefined
  const chains: ApplyStep[][] = []
  for (const cell of cells) {
    if (isConstCell(cell)) {
      const s = cell.aff.offset
      if (!(s > 0 && Number.isFinite(s))) return false
      chains.push([
        ['mul', 0],
        ['add', 1 / s],
      ])
      continue
    }
    if (cell.kind !== 'aff') return false
    const terms = [...cell.aff.terms].filter(([, c]) => c !== 0)
    const atom = terms.length === 1 ? cx.g.atoms.get(terms[0]?.[0] as string) : undefined
    if (atom?.kind !== 'var' || !cx.applied.has(atom.name)) return false
    const base = chainOf(cx, cell.aff)
    if (input !== undefined && input !== base.input) return false
    input = base.input
    chains.push([...base.steps, ['recip']])
  }
  if (input === undefined) return false
  const n = cells.length
  cx.g.need('gaussian-mv-function-scale')
  cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
  cx.g.addFactor(
    {
      operator: 'gaussian-mv-function-scale',
      args: [node.name, input],
      params: { steps: chains as unknown as Json },
    },
    node.name,
  )
  cx.syms.set(node.name, {
    dims: node.dims,
    cells: Array.from({ length: n }, (_, j) => cx.g.compAtom(node.name, j, n)),
  })
  if (node.record) cx.layout(node, 'derived', 'gaussian-mv', 'packed', [node.name], true)
  return true
}

/**
 * A non-centred vector whose coefficients each have a scale of their own (`z_j / sqrt(alpha_j)`, ARD): the products are scalar
 * variables (`apply-product`), each a coordinate of one Gaussian vector variable `D` (a `dot-constant` factor), and `D` is what the
 * rest of the model reads. The scalar factors are as before; the likelihood sees one vector, so its message to the coefficients is
 * dense and the posterior keeps their correlations. False (and nothing made) when a cell is not one variable.
 */
export function tiedToVector(
  cx: Context,
  node: ModelNode & { readonly kind: 'derived' },
  cells: readonly Cell[],
): boolean {
  const outs: string[] = []
  for (const cell of cells) {
    if (cell.kind !== 'aff' || cell.aff.offset !== 0 || cell.aff.terms.size !== 1) return false
    const [key, coefficient] = [...cell.aff.terms][0] as [string, number]
    const atom = cx.g.atoms.get(key)
    if (atom?.kind !== 'var' || coefficient !== 1 || outs.includes(atom.name)) return false
    outs.push(atom.name)
  }
  const n = outs.length
  cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
  for (const [j, out] of outs.entries()) {
    const a = new Array<number>(n).fill(0)
    a[j] = 1
    cx.g.addFactor({ operator: 'dot-constant', args: [out, node.name], params: { a } }, elementId(node.name, j))
  }
  cx.syms.set(node.name, {
    dims: node.dims,
    cells: Array.from({ length: n }, (_, j) => cx.g.compAtom(node.name, j, n)),
  })
  if (node.record) cx.layout(node, 'derived', 'gaussian-mv', 'packed', [node.name], true)
  return true
}
