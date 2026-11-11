/**
 * Values chosen by a discrete variable: a known index picks a component, an index that is a known function of a discrete variable
 * is a lookup of the components, a categorical or Bernoulli latent is the `select` gate that chooses among them (`gate`), and
 * `gather` (`at`), `pick`, `where` and the comparison of a lookup with a known value are written in these terms. Arithmetic on a
 * lookup, a value shifted by a state and the product of two gamma variables are `special`. Evaluation emits the `select` and
 * `gamma-product` factors; what a gate keeps as a lookup instead depends on the `EvalMode` it is evaluated in.
 */

import type { FamilyKind } from '../../../../message-passing/index.ts'
import {
  affOf,
  type Cell,
  combine,
  constAff,
  constCell,
  isConstCell,
  lookup,
  lookupArithmetic,
  lookupOfForms,
  prod,
  recipArithmetic,
  type Sym,
  scaledArithmetic,
  scaledOf,
  unflatten,
} from '../_cells.ts'
import type { Context } from '../_context.ts'
import { Refusal } from '../_refusal.ts'
import type { EvalMode } from './_affine.ts'

/** Arithmetic on a discrete variable's lookup or on a gamma variable's precision: undefined when neither operand is one. */
export function special(cx: Context, mode: EvalMode, op: string, x: Cell, y: Cell, index?: number): Cell | undefined {
  const shifted = shiftedByState(cx, mode, op, x, y)
  if (shifted !== undefined) return shifted
  const look = lookupArithmetic(op, x, y, (v) => cx.g.sizeOf(v))
  if (look !== undefined) return look
  const inverse = recipArithmetic(op, x, y)
  if (inverse !== undefined) return inverse
  const left = scaledOf(x)
  const right = scaledOf(y)
  if (op === 'mul' && left !== undefined && right !== undefined && left.offset === 0 && right.offset === 0) {
    if (left.variable === right.variable)
      throw new Refusal(
        'operator',
        `the square of the gamma variable '${left.variable}' as a precision has no message rule`,
      )
    // The product of two gamma variables (a local times a global precision): a gamma variable defined by `gamma-product`.
    const site = cx.current
    const out = cx.g.auxName(site, index)
    cx.g.need('gamma-product')
    cx.g.addVariable({ name: out, family: { kind: 'gamma' } })
    cx.g.addFactor({ operator: 'gamma-product', args: [out, left.variable, right.variable] }, out)
    return { kind: 'scaled', variable: out, scale: left.scale * right.scale, offset: 0 }
  }
  return scaledArithmetic(op, x, y)
}

/**
 * A Gaussian value shifted by a known function of a discrete variable (`g + offset[c]`, `offset[c] - g`, a wrapped normal's
 * copies a period apart): a gate that selects among the shifted forms `g + offset[k]` by the variable, the `select` factor on a
 * new variable. Undefined when the operation is not an addition or subtraction of a lookup and a random Gaussian form.
 */
export function shiftedByState(cx: Context, mode: EvalMode, op: string, x: Cell, y: Cell): Cell | undefined {
  if (op !== 'add' && op !== 'sub') return undefined
  const states = (v: string): number => cx.g.sizeOf(v)
  const asLookup = (c: Cell): (Cell & { kind: 'lookup' }) | undefined => {
    if (c.kind === 'lookup') return c.items === undefined ? c : undefined
    if (c.kind === 'rv' && (c.family === 'discrete' || c.family === 'bernoulli'))
      return lookup(
        c.variable,
        Array.from({ length: c.family === 'bernoulli' ? 2 : states(c.variable) }, (_, i) => i),
      )
    return undefined
  }
  const random = (c: Cell): boolean => c.kind === 'aff' && !isConstCell(c)
  const [look, other, lookFirst] =
    asLookup(x) !== undefined && random(y)
      ? [asLookup(x) as Cell & { kind: 'lookup' }, y, true]
      : asLookup(y) !== undefined && random(x)
        ? [asLookup(y) as Cell & { kind: 'lookup' }, x, false]
        : [undefined, x, false]
  if (look === undefined) return undefined
  const sign = op === 'add' ? 1 : -1
  const components = look.values.map((v): Cell => {
    const shifted = lookFirst ? combine(constAff(v), affOf(other), sign) : combine(affOf(other), constAff(v), sign)
    return { kind: 'aff', aff: shifted }
  })
  return gate(cx, mode, components, look.variable)
}

/** A comparison of a known function of a discrete variable with a known value: a lookup of 0 and 1 over its states. */
export function compare(cx: Context, op: string, x: Cell, y: Cell): Cell {
  const asLookup = (c: Cell): (Cell & { kind: 'lookup' }) | undefined =>
    lookupArithmetic('add', c, constCell(0), (v) => cx.g.sizeOf(v)) as (Cell & { kind: 'lookup' }) | undefined
  const a = asLookup(x)
  const b = asLookup(y)
  const f = (u: number, v: number): number => Number(op === 'gt' ? u > v : op === 'lt' ? u < v : u === v)
  if (a !== undefined && isConstCell(y))
    return lookup(
      a.variable,
      a.values.map((u) => f(u, y.aff.offset)),
    )
  if (b !== undefined && isConstCell(x))
    return lookup(
      b.variable,
      b.values.map((u) => f(x.aff.offset, u)),
    )
  if (isConstCell(x) && isConstCell(y)) return constCell(f(x.aff.offset, y.aff.offset))
  throw new Refusal(
    'operator',
    `'${op}' of random values has no message rule (only a discrete variable against a known value is mapped, as a gate)`,
  )
}

export function gather(cx: Context, mode: EvalMode, a: Sym, idx: Sym): Sym {
  const first = a.dims[0]
  if (first === undefined) throw new Refusal('structure', 'at: the first operand needs a dimension to index')
  const rest = a.dims.slice(1)
  const dims = [...idx.dims, ...rest]
  const shape = cx.shapeOf(dims)
  const restShape = cx.shapeOf(rest)
  const cells: Cell[] = []
  for (let i = 0; i < prod(shape); i++) {
    const at = unflatten(i, shape)
    const code = cx.cellAt(idx, dims, at)
    let restFlat = 0
    for (let k = 0; k < rest.length; k++)
      restFlat = restFlat * (restShape[k] as number) + (at[idx.dims.length + k] as number)
    cells.push(choose(cx, mode, code, cx.size(first), (j) => a.cells[j * prod(restShape) + restFlat] as Cell, first))
  }
  return { dims, cells }
}

/**
 * The cell chosen by an index cell among `states` components: a known index picks one, an index that is a known function of a
 * discrete variable (a change point's `t > tau`) is a lookup of the components, and a categorical or Bernoulli latent is the
 * gate that selects among them.
 */
export function choose(
  cx: Context,
  mode: EvalMode,
  code: Cell,
  states: number,
  component: (state: number) => Cell,
  dim: string,
  asLookup = false,
): Cell {
  if (isConstCell(code)) {
    const c = code.aff.offset
    if (!Number.isInteger(c) || c < 0 || c >= states)
      throw new Refusal('data', `at: index ${c} is out of range for '${dim}'`)
    return component(c)
  }
  if (code.kind === 'lookup' && code.items === undefined) {
    const comps = code.values.map((v) => {
      if (!Number.isInteger(v) || v < 0 || v >= states)
        throw new Refusal('data', `at: index ${v} is out of range for '${dim}'`)
      return component(v)
    })
    return lookupOf(cx, comps, code.variable)
  }
  if (code.kind !== 'rv' || code.family !== 'discrete')
    throw new Refusal(
      'operator',
      'at: the index must be data or a categorical latent (a gate); a computed random index has no message rule',
    )
  const comps = Array.from({ length: states }, (_, k) => component(k))
  // The branches of a `pick` stay the states of one lookup (their random Gaussian forms are the branches of the gate the observation
  // is), instead of one `select` variable: the observation is a gate of likelihoods, exact in the discrete variable.
  if (asLookup && comps.every((c) => c.kind === 'aff')) return lookupOf(cx, comps, code.variable)
  return gate(cx, mode, comps, code.variable)
}

/**
 * `pick(values, index)`: along the last dimension of `values`, the entry that `index` names, for every element of the other
 * dimensions (a different regression per class of a mixture of regressions: `values` the predictions of each class, `index` the
 * class of each row). A known index selects, a categorical latent is the gate over the branches.
 */
export function pick(cx: Context, mode: EvalMode, values: Sym, idx: Sym): Sym {
  const component = values.dims[values.dims.length - 1]
  if (component === undefined) throw new Refusal('structure', 'pick: the values need a dimension to pick along')
  const states = cx.size(component)
  const kept = values.dims.slice(0, -1)
  const dims = cx.joinDims([{ dims: kept, cells: [] }, idx])
  const shape = cx.shapeOf(dims)
  const cells: Cell[] = []
  for (let i = 0; i < prod(shape); i++) {
    const at = unflatten(i, shape)
    const code = cx.cellAt(idx, dims, at)
    const compAt = (k: number): Cell => {
      let flat = 0
      for (const d of kept) flat = flat * cx.size(d) + (at[dims.indexOf(d)] as number)
      return values.cells[flat * states + k] as Cell
    }
    cells.push(choose(cx, mode, code, states, compAt, component, true))
  }
  return { dims, cells }
}

/**
 * `y = c_z`: a gate that picks one of several random values by a discrete (or Bernoulli) variable, the `select` factor of
 * the engine on a new variable of the components' family. The components are Gaussian (an affine value becomes a variable)
 * or variables of one other family; a known component has no variable to select and is refused.
 */
export function gate(cx: Context, mode: EvalMode, components: readonly Cell[], z: string): Cell {
  const site = cx.current
  // Known components: the value is a known function of the discrete variable (a lookup), not a random value to select.
  if (components.every(isConstCell)) return lookupOfForms(z, components)
  // A mix of known and random Gaussian forms: a lookup whose random states carry their forms (a gate of Gaussian branches).
  if (components.every((c) => c.kind === 'aff') && components.some(isConstCell)) return lookupOfForms(z, components)
  // Read by an observation whose precision is chosen by the same discrete variable: the components stay states of one lookup, so
  // the observation is one gate over its branches (random means and gamma precisions of each state), not a select per value.
  if (mode === 'lookup') {
    if (components.every((c) => c.kind === 'aff')) return lookupOfForms(z, components)
    const gammas = components.map((c) => (isConstCell(c) ? undefined : scaledOf(c)))
    if (components.every((c, k) => isConstCell(c) || gammas[k] !== undefined))
      return lookup(
        z,
        components.map((c) => (isConstCell(c) ? c.aff.offset : 0)),
        { precisions: gammas },
      )
  }
  const first = components[0] as Cell
  const rv = first.kind === 'rv'
  const kind: FamilyKind = rv ? first.family : 'gaussian'
  const names = components.map((c) => {
    if (rv) {
      if (c.kind !== 'rv' || c.family !== kind)
        throw new Refusal('operator', `a gate selects among values of one family: they are not all ${kind}`)
      return c.variable
    }
    const v = cx.g.materialise(affOf(c), site, undefined)
    if (typeof v === 'number')
      throw new Refusal(
        'operator',
        'a gate selects among random values: a known component has no variable (give it a prior)',
      )
    return v
  })
  const out = cx.g.auxName(site, undefined)
  cx.g.need('select')
  cx.g.addVariable({ name: out, family: { kind } })
  cx.g.addFactor({ operator: 'select', args: [out, z, ...names], params: { components: names.length } }, out)
  if (rv) return { kind: 'rv', variable: out, family: kind }
  const key = cx.g.atomKey({ kind: 'var', name: out })
  cx.g.atoms.set(key, { kind: 'var', name: out })
  return { kind: 'aff', aff: { terms: new Map([[key, 1]]), offset: 0 } }
}

/**
 * The value that is `components[k]` in state `k` of the discrete variable `z`, as a lookup: known numbers, or Gaussian forms
 * that are random in some states (read by a gate of Gaussian branches, over the variables the branches share).
 */
export function lookupOf(_cx: Context, components: readonly Cell[], z: string): Cell {
  if (!components.every((c) => c.kind === 'aff'))
    throw new Refusal(
      'operator',
      'a value chosen by a discrete variable must be known or a Gaussian form in every state',
    )
  return lookupOfForms(z, components)
}

/** `where(c, a, b)`: a known condition picks a branch; a Bernoulli latent condition is the gate `c ? a : b`. */
export function whereGate(cx: Context, mode: EvalMode, [cond, a, b]: [Sym, Sym, Sym]): Sym {
  return cx.zip([cond, a, b], ([c, x, y]) => {
    if (isConstCell(c as Cell)) return (c as Cell & { kind: 'aff' }).aff.offset !== 0 ? (x as Cell) : (y as Cell)
    const chosen = c as Cell
    if (chosen.kind === 'lookup' && chosen.items === undefined)
      return lookupOf(
        cx,
        chosen.values.map((v) => (v !== 0 ? (x as Cell) : (y as Cell))),
        chosen.variable,
      )
    if (chosen.kind !== 'rv' || chosen.family !== 'bernoulli')
      throw new Refusal(
        'operator',
        'where: the condition must be known or a Bernoulli latent; a comparison of random values has no message rule',
      )
    return gate(cx, mode, [y as Cell, x as Cell], chosen.variable)
  })
}
