/**
 * What the static analyses of a spec read from its expressions: the operators an expression applies, whether it holds a
 * `custom` call, the distribution and expressions of a node, the latents an expression depends on (through derived
 * nodes), whether it is affine in a chosen set of latents, and the forms in which a latent is used (itself, an `at`
 * lookup of it). The conjugacy, linear-Gaussian block and graph analyses and `analyseSpec` itself share these.
 */

import type { ExprSpec, ModelNode, ModelSpec } from './spec.ts'

/** The nodes an expression can refer to by name (value nodes: observed nodes and factors cannot be referred to). */
export type NodesByName = ReadonlyMap<string, ModelNode>

/** The memo of {@link latentsIn} for derived nodes, shared by the calls of one analysis. */
export type LatentMemo = Map<string, Set<string>>

/**
 * The nodes of a spec an expression can refer to, by name.
 *
 * @param spec - The spec.
 * @returns Data, latent, derived, generated and param nodes by name.
 */
export function nodesByName(spec: ModelSpec): Map<string, ModelNode> {
  const byName = new Map<string, ModelNode>()
  for (const n of spec.nodes) if (n.kind !== 'observed' && n.kind !== 'factor') byName.set(n.name, n)
  return byName
}

/** Collect the operators an expression applies into `out`. */
export function exprOps(e: ExprSpec, out: Set<string>): void {
  if (e.kind === 'op') {
    out.add(e.op)
    for (const a of e.args) exprOps(a, out)
  } else if (e.kind === 'custom') for (const a of e.args) exprOps(a, out)
}

/** Whether an expression holds a `custom` call. */
export function exprHasCustom(e: ExprSpec): boolean {
  if (e.kind === 'custom') return true
  return e.kind === 'op' && e.args.some(exprHasCustom)
}

/** The expressions a node evaluates: distribution parameters, an observed value, a derived or factor expression. */
export function nodeExprs(n: ModelNode): ExprSpec[] {
  switch (n.kind) {
    case 'latent':
    case 'generated':
      return Object.values(n.dist.params)
    case 'observed':
      return [...Object.values(n.dist.params), n.value]
    case 'derived':
    case 'factor':
      return [n.expr]
    default:
      return []
  }
}

/** The distribution of a latent, generated or observed node. */
export function distOf(
  n: ModelNode,
): { readonly family: string; readonly params: { readonly [k: string]: ExprSpec } } | undefined {
  return n.kind === 'latent' || n.kind === 'generated' || n.kind === 'observed' ? n.dist : undefined
}

/** The latents an expression depends on, through derived nodes too. */
export function latentsIn(e: ExprSpec, byName: NodesByName, memo: LatentMemo): Set<string> {
  const out = new Set<string>()
  const walk = (x: ExprSpec): void => {
    if (x.kind === 'ref') {
      const node = byName.get(x.name)
      if (node?.kind === 'latent') out.add(x.name)
      else if (node?.kind === 'derived') {
        let m = memo.get(x.name)
        if (m === undefined) {
          m = latentsIn(node.expr, byName, memo)
          memo.set(x.name, m)
        }
        for (const l of m) out.add(l)
      }
    } else if (x.kind === 'op' || x.kind === 'custom') for (const a of x.args) walk(a)
  }
  walk(e)
  return out
}

/** Whether an expression depends on any latent of `vars` (every latent when `vars` is left out). */
export function usesLatent(e: ExprSpec, byName: NodesByName, memo: LatentMemo, vars?: ReadonlySet<string>): boolean {
  for (const l of latentsIn(e, byName, memo)) if (vars === undefined || vars.has(l)) return true
  return false
}

/**
 * Whether an expression is affine in the latents of `vars` (every latent when left out), constants, data and the other
 * latents aside: a sum of terms each linear in at most one of them, with coefficients that do not depend on them.
 *
 * @param e - The expression.
 * @param byName - The nodes it can refer to.
 * @param memo - The memo of derived nodes' latents.
 * @param vars - The latents treated as variables; the rest are constants.
 * @returns Whether the expression is affine in them.
 */
export function isAffineIn(e: ExprSpec, byName: NodesByName, memo: LatentMemo, vars?: ReadonlySet<string>): boolean {
  const uses = (x: ExprSpec): boolean => usesLatent(x, byName, memo, vars)
  const aff = (x: ExprSpec): boolean => isAffineIn(x, byName, memo, vars)
  switch (e.kind) {
    case 'const':
      return true
    case 'ref': {
      const node = byName.get(e.name)
      return node?.kind === 'derived' ? aff(node.expr) : true
    }
    case 'custom':
      return !e.args.some(uses)
    case 'op': {
      const [a, b] = e.args
      switch (e.op) {
        case 'add':
        case 'sub':
        case 'neg':
        case 'sum':
        case 'mean':
          return e.args.every(aff)
        case 'mul':
        case 'matmul':
          return a !== undefined && b !== undefined && aff(a) && aff(b) && !(uses(a) && uses(b))
        case 'div':
        case 'at':
          return a !== undefined && b !== undefined && aff(a) && !uses(b)
        default:
          return !e.args.some(uses)
      }
    }
  }
}

/** Whether `e` is the reference `name`, or a lookup `at(name, index)`; the form of use that keeps a pair conjugate. */
export function useOf(e: ExprSpec, name: string): 'direct' | 'index' | undefined {
  if (e.kind === 'ref' && e.name === name) return 'direct'
  if (e.kind === 'op' && e.op === 'at' && e.args[0]?.kind === 'ref' && e.args[0].name === name) return 'index'
  return undefined
}
