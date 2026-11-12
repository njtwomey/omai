/**
 * An observation whose likelihood is chosen by an index among branches of different families (`gate(poisson,negativeBinomial)`,
 * model operator `d.gate`): one engine `gate` per element, whose branch `k` is the factor that branch's family maps to when it is
 * observed on its own. The branches are not mapped a second time here: each is run through the table of element mappings as if it
 * were an observed node of its family (a synthetic node of the branch's parameters), and the one likelihood factor it made is taken
 * out of the graph and becomes the gate's branch over the variables it reads (the value algebra it needed, an auxiliary predictor, a
 * packed vector, stays where the mapping put it). A branch with no random parameter has no factor: its likelihood is a known weight of
 * its state, a `table` over the index beside the gate, from the library's own density of the family.
 */

import type { Json } from '../../../../base/index.ts'
import type { Observed, OperatorName, Params } from '../../../../message-passing/index.ts'
import { asTensor, toFlat } from '../../../../tensor/index.ts'
import { gateBranches } from '../../dists/families.ts'
import type { DistSpec, ExprSpec, ModelNode } from '../../spec.ts'
import { GateBuilder, isOperatorName, VALUE_ALGEBRA_OPERATORS } from '../_builder.ts'
import type { Cell } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { cellsOf, constAt } from '../_params.ts'
import {
  gateBranchNotOneLikelihood,
  gateBranchRefused,
  gateIndexNotDiscrete,
  gateIndexOutsideBranches,
  gateStatesVersusBranches,
} from '../_reasons.ts'
import { Refusal } from '../_refusal.ts'
import { evalSym } from '../_symbolic/_affine.ts'
import type { Element } from '../_table.ts'

type Observation = ModelNode & { readonly kind: 'observed' }

/** What maps the observation of one branch as if it stood alone: the table of element mappings (passed in, the table sits above this file). */
export type BranchMapper = (cx: Context, node: Observation, el: Element) => string | null

/**
 * The branch `k` of a gate as an observed node of its own family: its parameters are the gate's `k.<name>` ones, named after the gate's
 * node (`y~k`) so that what the mapping keeps for a node (the cells of its parameters) is its own.
 */
function branchNode(node: Observation, family: string, k: number): Observation {
  const params: Record<string, ExprSpec> = {}
  for (const [name, e] of Object.entries(node.dist.params))
    if (name.startsWith(`${k}.`)) params[name.slice(`${k}.`.length)] = e
  const dist: DistSpec = { family, params }
  return { ...node, name: `${node.name}~${k}`, dist }
}

/** The weight of a branch whose parameters are all known: the library's density of the family at the value (the likelihood of that state). */
function knownWeight(cx: Context, node: Observation, branch: Observation, y: number, i: number): number {
  const def = cx.families[branch.dist.family]
  const args: Record<string, ReturnType<typeof asTensor>> = {}
  for (const name of def?.params ?? []) {
    const e = branch.dist.params[name]
    const v = e === undefined ? def?.defaults[name] : constAt(cx, node, e, i)
    if (v !== undefined) args[name] = asTensor(v, 'float64')
  }
  return Math.exp(toFlat(def?.make(args).logProb(asTensor(y, 'float64')) ?? asTensor(Number.NaN))[0] as number)
}

/** A constant argument as a branch takes it: a number. */
function numberArg(node: string, k: number, family: string, c: Observed): number {
  if (typeof c !== 'number')
    throw gateBranchNotOneLikelihood(node, k, family, 'it reads a constant that is not a number')
  return c
}

/**
 * Map one element of an observation of a gate family. The index is a known value (the branch it names is mapped alone, as an ordinary
 * observation) or a discrete or Bernoulli latent, whose states are the branches. Each branch with a random parameter is mapped by
 * `mapBranch` and its one likelihood factor becomes branch `k` of the gate; each branch whose parameters are all known is a weight of
 * its state, a `table` over the index (omitted when every weight is one). Refused, naming the node, the branch and the reason: an index
 * that is neither, a number of states that is not the number of branches, a branch the mapping refuses, a branch that maps to no factor
 * or to several (a hurdle, a nested gate or a table over a discrete parent is not one likelihood factor).
 */
export function observedGateFamilies(
  cx: Context,
  node: Observation,
  el: Element,
  mapBranch: BranchMapper,
): string | null {
  const families = gateBranches(el.fam) as readonly string[]
  const indexExpr = node.dist.params['index'] as ExprSpec
  const index = cellsOf(cx, node, 'gateIndex', () => evalSym(cx, indexExpr))[el.i] as Cell
  const branchOf = (k: number): Observation => branchNode(node, families[k] as string, k)
  const asBranch = (k: number): Element => ({ ...el, fam: families[k] as string, name: `${el.name}~${k}` })

  if (index.kind === 'aff' && index.aff.terms.size === 0) {
    const k = index.aff.offset
    if (!(Number.isInteger(k) && k >= 0 && k < families.length))
      throw gateIndexOutsideBranches(node.name, k, families.length)
    return mapBranch(cx, branchOf(k), { ...el, fam: families[k] as string })
  }
  if (index.kind !== 'rv' || (index.family !== 'discrete' && index.family !== 'bernoulli'))
    throw gateIndexNotDiscrete(node.name)
  const z = index.variable
  const states = index.family === 'bernoulli' ? 2 : cx.g.sizeOf(z)
  if (states !== families.length) throw gateStatesVersusBranches(node.name, states, families.length)

  const gate = new GateBuilder(z)
  const weights: number[] = []
  let random = false
  for (const [k, family] of families.entries()) {
    const branch = branchOf(k)
    if (Object.values(branch.dist.params).every((e) => cx.isConst(e))) {
      weights.push(knownWeight(cx, node, branch, el.y, el.i))
      gate.add({})
      continue
    }
    if (family === 'hurdlePoisson')
      throw gateBranchNotOneLikelihood(
        node.name,
        k,
        family,
        'a hurdle is a Bernoulli on its zero part and a truncated Poisson on the rest, two factors',
      )
    weights.push(1)
    random = true
    const before = cx.g.factors.length
    try {
      mapBranch(cx, branch, asBranch(k))
    } catch (e) {
      if (e instanceof Refusal) throw gateBranchRefused(node.name, k, family, e.message)
      throw e
    }
    const added = cx.g.factors.splice(before)
    const own = added.filter((f) => !VALUE_ALGEBRA_OPERATORS.some((a) => a === f.operator))
    cx.g.factors.push(...added.filter((f) => !own.includes(f)))
    const [factor] = own
    if (factor === undefined || own.length > 1)
      throw gateBranchNotOneLikelihood(node.name, k, family, `it maps to ${own.length} factors`)
    if (!isOperatorName(factor.operator) || factor.operator === 'gate' || factor.operator === 'table')
      throw gateBranchNotOneLikelihood(node.name, k, family, `it maps to a ${factor.operator} factor`)
    gate.add({
      operator: factor.operator as OperatorName,
      args: factor.args.map((a) =>
        typeof a === 'string' ? gate.slot(a) : { const: numberArg(node.name, k, family, a.const) },
      ),
      ...(factor.params === undefined ? {} : { params: factor.params as Params }),
    })
  }
  const id = cx.elementName(node.name, node.dims, el.i)
  if (weights.some((w) => w !== 1))
    cx.g.addFactor(
      { operator: 'table', args: [z], params: { states: [states], values: weights as unknown as Json } },
      random ? `${id}|weights` : id,
    )
  if (random) cx.g.addFactor(gate.factor(), id)
  return null
}
