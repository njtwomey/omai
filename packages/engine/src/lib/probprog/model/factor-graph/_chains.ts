/**
 * The chains of an unrolled scan: the steps `h[t] ~ N(a h[t-1] + c . u + d, 1 / (scale tau + offset))` of one series, with known
 * coefficients, Gaussian drivers `u` that every step reads (the level of an autoregression) and one gamma precision, which the
 * walk collects as it maps the nodes and writes, once it has mapped them all, as one `gaussian-chain` factor on the states, the
 * drivers and the precision (placed where the first step would have been). A step with its own mean variable and `gaussian-gamma-precision` factor sends each variable a message
 * from two neighbour cavities taken as independent, which they are not once a driver or the precision is shared (the drivers came
 * out twice as wide as the exact ones, the precision's sixty messages did not settle); the one factor integrates the chain
 * against all the cavities at once. The states keep their own variables and layouts, so nothing that reads the series changes;
 * only the factors that tie neighbours do. Only with `packScalars` (the engine's joint-structure layouts; the variational rule
 * has no chain factor), and only when the registry has the operator.
 */

import type { Json } from '../../../base/index.ts'
import type { Aff, Cell, Precision } from './_cells.ts'
import type { Context } from './_context.ts'

/** One series of steps being collected: its states in order, and what every step has in common. */
export interface ChainBuild {
  /** The variables `h[t0] .. h[t1]` of the steps' states: the first is only the previous state of the first step. */
  readonly states: string[]
  /** The coefficient of the previous state. */
  readonly a: number
  /** The drivers' variables, in the order of the atoms' keys, and their coefficients. */
  readonly driveVars: readonly string[]
  readonly drive: readonly number[]
  readonly intercept: number
  readonly precision: Precision
  /** Where in the graph's factors the first step would have been added: the chain factor goes there, ahead of the readings of the states it links. */
  readonly at: number
}

/** The name of a state of an unrolled scan, `<scan>[<t>]`: the scan and the step. */
const STEP = /^(.+)\[(\d+)\]$/

const close = (x: number, y: number): boolean => Math.abs(x - y) <= 1e-12 * Math.max(1, Math.abs(x), Math.abs(y))

/** Whether a chain's steps all have what the next one has: the coefficients, the drivers, the intercept and the precision. */
function sameStep(chain: ChainBuild, next: Omit<ChainBuild, 'states' | 'at'>): boolean {
  return (
    close(chain.a, next.a) &&
    close(chain.intercept, next.intercept) &&
    chain.driveVars.length === next.driveVars.length &&
    chain.driveVars.every(
      (v, k) => v === next.driveVars[k] && close(chain.drive[k] as number, next.drive[k] as number),
    ) &&
    chain.precision.variable === next.precision.variable &&
    close(chain.precision.scale, next.precision.scale) &&
    close(chain.precision.offset, next.precision.offset)
  )
}

/**
 * Take a step of an unrolled scan into the chain it continues, if it is one: a scalar normal latent `<scan>[t]` whose mean is an affine
 * form with a known nonzero coefficient on the previous state `<scan>[t-1]` and Gaussian drivers for the rest, and whose precision is
 * the gamma variable of the chain's. The state's variable is already made (the caller made it); what is not made is the mean
 * variable and the precision factor, which the chain factor replaces. A step that does not continue the chain ends it and starts another.
 *
 * @param cx - The compilation.
 * @param name - The state's node name (and variable).
 * @param mean - The cell of the state's mean.
 * @param precision - The precision of the state's step.
 * @returns Whether the step is part of a chain (its mean variable and precision factor are then not to be made).
 */
export function takeChainStep(cx: Context, name: string, mean: Cell, precision: Precision): boolean {
  if (!cx.packScalars || cx.g.ops['gaussian-chain'] === undefined || mean.kind !== 'aff') return false
  const found = STEP.exec(name)
  if (found === null || Number(found[2]) < 1) return false
  const scan = found[1] as string
  const previous = `${scan}[${Number(found[2]) - 1}]`
  const aff: Aff = mean.aff
  const previousKey = cx.g.atomKey({ kind: 'var', name: previous })
  const a = aff.terms.get(previousKey)
  if (a === undefined || a === 0 || !Number.isFinite(a)) return false
  const keys = [...aff.terms.keys()].filter((k) => k !== previousKey).sort()
  const driveVars: string[] = []
  for (const key of keys) {
    const atom = cx.g.atoms.get(key)
    if (atom?.kind !== 'var') return false
    driveVars.push(atom.name)
  }
  const step: Omit<ChainBuild, 'states' | 'at'> = {
    a,
    driveVars,
    drive: keys.map((k) => aff.terms.get(k) as number),
    intercept: aff.offset,
    precision,
  }
  const open = cx.chains.get(scan)
  if (open !== undefined && open.states.at(-1) === previous && sameStep(open, step)) {
    open.states.push(name)
    return true
  }
  if (open !== undefined) cx.chainsDone.push(open)
  cx.chains.set(scan, { states: [previous, name], ...step, at: cx.g.factors.length })
  return true
}

/**
 * The precision of the weak prior a state with an uninformative cavity carries inside the chain factor (`ridge` of `gaussian-chain`),
 * as a fraction of the precision of the first state's prior. A likelihood with an improper tail, such as the log-variance reading of a
 * return near zero (the posterior of the series has a dozen of them in sixty returns), sends its state a message with almost no
 * precision and a mean far away; the factor integrates the precision over its whole range and would let it run to zero, and with
 * 0.01 a few such states still made the messages cycle. 0.05 holds them, and moves no state that has a cavity of its own.
 */
const RIDGE = 0.05

/** The precision of the prior factor on a state (1 when it has none: the states of a chain whose first state is not a prior's). */
function firstPrecision(cx: Context, state: string): number {
  for (const f of cx.g.factors) {
    if (f.operator !== 'prior' || f.args[0] !== state) continue
    const eta = f.params?.['eta']
    if (Array.isArray(eta) && typeof eta[1] === 'number' && eta[1] > 0) return eta[1]
  }
  return 1
}

/**
 * Write every chain the walk collected as its `gaussian-chain` factor, in the order they were started, after the nodes: the
 * arguments are the precision, the states and the drivers. The factor is named after the last state.
 *
 * @param cx - The compilation.
 */
export function finishChains(cx: Context): void {
  const chains = [...cx.chainsDone, ...cx.chains.values()].sort((p, q) => p.at - q.at)
  for (const [k, chain] of chains.entries()) {
    const params: Record<string, Json> = { length: chain.states.length, coefficient: chain.a }
    if (chain.drive.length > 0) params['drive'] = [...chain.drive]
    if (chain.intercept !== 0) params['intercept'] = chain.intercept
    if (chain.precision.scale !== 1) params['scale'] = chain.precision.scale
    if (chain.precision.offset !== 0) params['offset'] = chain.precision.offset
    params['ridge'] = RIDGE * firstPrecision(cx, chain.states[0] as string)
    cx.g.addFactor(
      { operator: 'gaussian-chain', args: [chain.precision.variable, ...chain.states, ...chain.driveVars], params },
      chain.states.at(-1) as string,
    )
    // The factor comes ahead of the readings of the states (a state's likelihood then meets a proper cavity), where the steps would have been.
    const made = cx.g.factors.pop()
    if (made !== undefined) cx.g.factors.splice(chain.at + k, 0, made)
  }
  cx.chains.clear()
  cx.chainsDone.length = 0
}
