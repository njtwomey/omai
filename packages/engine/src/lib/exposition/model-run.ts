/**
 * The exposition record of inference on a model graph (`modelRun`): a `Structure` (a plate diagram from
 * `toStructure`, a factor graph from `graphStructure`) with, at each recorded step of a run, what is believed of each
 * node (a posterior summary by node id, the observed value of an observed one), the messages that passed along the
 * edges with a readable badge, and, beside the picture, the convergence series (change, evidence or bound, improper
 * messages). The ids are the library's id convention, a site's name for its node, so the record is built from a tape
 * or an anytime snapshot without the model (`modelRunFromTape` in `model-run-tape.ts`) and a view paints it on the
 * picture while the run goes on.
 */

import type { Json } from '../base/index.ts'
import { OmaiError } from '../base/index.ts'
import type { Structure } from '../scene/index.ts'
import { seal } from './_check.ts'

/** What is believed of one node at a step. Every field but `text` is optional: a family gives what it has. */
export interface NodeState {
  /** The family of the belief (`gaussian`, `beta`, `discrete`, `normal` for samples...), when known. */
  readonly family?: string
  /** The posterior mean (for a discrete belief, its probability of the most likely state is `p`). */
  readonly mean?: number
  readonly sd?: number
  /** The ends of an interval, when the method gives one (draws). */
  readonly lower?: number
  readonly upper?: number
  /** The probabilities of a discrete belief, or the probability of true of a Bernoulli one. */
  readonly probs?: readonly number[]
  /** The value of an observed node. */
  readonly observed?: Json
  /** The belief has no density yet (an improper message product): drawn as such, never as a number. */
  readonly improper?: boolean
  /** Split R-hat and effective sample size of a sampled site. */
  readonly rhat?: number
  readonly ess?: number
  /** The elements of a vector site, `w[0]`, `w[1]`: mean and sd of each. */
  readonly elements?: readonly { readonly name: string; readonly mean: number; readonly sd: number }[]
  /** The belief in a line: `0.56 ± 0.83`, `p = 0.7`, `observed 2`, `improper`. */
  readonly text: string
}

/** A message that passed along an edge at a step, with its natural parameters read as a distribution. */
export interface MessageBadge {
  /** The ends, as node ids of the structure (a factor of a plate diagram is its site). Equal ends: a note on the node. */
  readonly from: string
  readonly to: string
  /** The message in words: `N(μ = 0.83, σ² = 2.1)`, `Beta(2, 3)`, `p = 0.7`. */
  readonly text: string
  /** The operator of the factor that sent it, when the tape says. */
  readonly operator?: string
}

/** One recorded state of the run. */
export interface ModelRunStep {
  /** The tape step (or, for an anytime snapshot, the step it was taken at). */
  readonly step: number
  /** What is believed of each node by id; nodes not named keep what they had. */
  readonly nodes: Readonly<Record<string, NodeState>>
  /** The messages sent at this step. */
  readonly messages?: readonly MessageBadge[]
  readonly caption?: string
}

/** A series beside the picture: change over sweeps, the evidence estimate or the bound, a count of improper messages. */
export interface ModelRunSeries {
  /** `change`, `evidence`, `elbo`, `improper`, `skipped`...: the path's last name on the tape. */
  readonly name: string
  readonly label: string
  readonly steps: readonly number[]
  readonly values: readonly number[]
  /** Which way is better, where one is: `higher` for the evidence and the bound, `lower` for the change. */
  readonly direction?: 'higher' | 'lower'
}

/**
 * Inference on a model graph: the structure, the recorded steps, the series and where the run is. `status` is
 * `running` for a run in progress (an anytime snapshot, a tape without its end): a view shows the latest state and
 * more steps arrive as the run goes on.
 */
export interface ModelRun {
  readonly kind: 'modelRun'
  readonly structure: Structure
  readonly steps: readonly ModelRunStep[]
  readonly series: readonly ModelRunSeries[]
  readonly status: 'running' | 'completed' | 'converged' | 'diverged' | 'stopped' | 'failed'
  /** The method that ran (`ep`, `vmp`, `nuts`), when the tape says. */
  readonly method?: string
}

/**
 * A model run, with its ids checked against the structure.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `value` when a step names a node or an edge the structure lacks, the steps are out of order, or a
 *   series has different numbers of steps and values.
 * @example
 * ```ts
 * import { modelRun } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * const s = structure('model-graph', [{ id: 'mu', label: 'mu', role: 'latent' }], [])
 * modelRun({ structure: s, status: 'running', series: [], steps: [{ step: 1, nodes: { mu: { mean: 0.5, sd: 1, text: '0.5 ± 1' } } }] }).steps.length // => 1
 * ```
 */
export function modelRun(record: Omit<ModelRun, 'kind'>): ModelRun {
  const ids = new Set(record.structure.nodes.map((n) => n.id))
  let previous = Number.NEGATIVE_INFINITY
  for (const [t, step] of record.steps.entries()) {
    if (step.step < previous) throw new OmaiError('value', `modelRun: steps[${t}] is out of order`)
    previous = step.step
    const unknown = Object.keys(step.nodes).find((id) => !ids.has(id))
    if (unknown !== undefined)
      throw new OmaiError('value', `modelRun: steps[${t}] names the node ${unknown}, which the structure lacks`)
    const lost = (step.messages ?? []).find((m) => !(ids.has(m.from) && ids.has(m.to)))
    if (lost !== undefined)
      throw new OmaiError(
        'value',
        `modelRun: steps[${t}] has a message ${lost.from} to ${lost.to} between unknown nodes`,
      )
  }
  for (const s of record.series)
    if (s.steps.length !== s.values.length)
      throw new OmaiError(
        'shape',
        `modelRun: series ${s.name} has ${s.steps.length} steps and ${s.values.length} values`,
      )
  return seal({ kind: 'modelRun', ...record })
}

/**
 * The state of a run at a step: the latest recorded one at or before it, with what every node was last said to be (a
 * step names only the nodes that changed), and the messages of that step.
 *
 * @param record - The record.
 * @param step - The tape step (default the last recorded).
 * @returns The beliefs by node id, the messages and the step's caption; `undefined` for a record with no steps.
 * @example
 * ```ts
 * import { modelRun, modelRunAt } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * const s = structure('model-graph', [{ id: 'a', label: 'a', role: 'latent' }, { id: 'b', label: 'b', role: 'latent' }], [])
 * const r = modelRun({ structure: s, status: 'running', series: [], steps: [
 *   { step: 1, nodes: { a: { mean: 1, text: '1' } } },
 *   { step: 2, nodes: { b: { mean: 2, text: '2' } } },
 * ] })
 * Object.keys(modelRunAt(r, 2)?.nodes ?? {}) // => ['a', 'b']
 * ```
 */
export function modelRunAt(
  record: ModelRun,
  step?: number,
):
  | {
      readonly step: number
      readonly nodes: Readonly<Record<string, NodeState>>
      readonly messages: readonly MessageBadge[]
      readonly caption?: string
    }
  | undefined {
  const upto = step === undefined ? record.steps : record.steps.filter((s) => s.step <= step)
  const last = upto.at(-1)
  if (last === undefined) return undefined
  const nodes: Record<string, NodeState> = {}
  for (const s of upto) Object.assign(nodes, s.nodes)
  return {
    step: last.step,
    nodes,
    messages: last.messages ?? [],
    ...(last.caption === undefined ? {} : { caption: last.caption }),
  }
}

/** How much a step moved the run: the largest change of a belief it reports, else how many beliefs it names. */
function movement(record: ModelRun, step: ModelRunStep): number {
  const change = record.series.find((s) => s.name === 'change')
  const at = change?.steps.indexOf(step.step) ?? -1
  const value = at < 0 ? undefined : change?.values[at]
  return value !== undefined && Number.isFinite(value) ? value : Object.keys(step.nodes).length
}

/**
 * A model run cut down to at most `maxSteps` steps, for a record that has to cross a worker boundary or sit in a page:
 * the first and the last step are always kept, and of the steps between them the ones where the beliefs moved most (the
 * largest change the run reported at the step), in their original order. What a dropped step said about a node is carried
 * into the next kept step, so the state at any kept step is the state it had in the full record; the series are cut to
 * the kept steps. A record with no more steps than the cap is returned as it is.
 *
 * @param record - The record.
 * @param maxSteps - The most steps to keep (at least 2).
 * @returns The thinned record.
 * @throws OmaiError `value` when `maxSteps` is below 2.
 * @example
 * ```ts
 * import { modelRun, thinModelRun } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * const s = structure('model-graph', [{ id: 'a', label: 'a', role: 'latent' }], [])
 * const steps = [0, 1, 2, 3, 4, 5].map((t) => ({ step: t, nodes: { a: { mean: t, text: String(t) } } }))
 * const change = { name: 'change', label: 'change', steps: [1, 2, 3, 4, 5], values: [9, 1, 5, 2, 0] }
 * const r = modelRun({ structure: s, status: 'completed', series: [change], steps })
 * thinModelRun(r, 4).steps.map((x) => x.step) // => [0, 1, 3, 5]
 * ```
 */
export function thinModelRun(record: ModelRun, maxSteps: number): ModelRun {
  if (maxSteps < 2) throw new OmaiError('value', `thinModelRun: maxSteps must be at least 2, got ${maxSteps}`)
  const n = record.steps.length
  if (n <= maxSteps) return record
  const ranked = record.steps
    .slice(1, n - 1)
    .map((step, i) => ({ i: i + 1, move: movement(record, step) }))
    .sort((a, b) => b.move - a.move || a.i - b.i)
    .slice(0, maxSteps - 2)
  const keep = new Set([0, n - 1, ...ranked.map((r) => r.i)])
  const steps: ModelRunStep[] = []
  let carried: Record<string, NodeState> = {}
  for (const [i, step] of record.steps.entries()) {
    carried = { ...carried, ...step.nodes }
    if (!keep.has(i)) continue
    steps.push({ ...step, nodes: carried })
    carried = {}
  }
  const kept = new Set(steps.map((s) => s.step))
  const series = record.series.map((s) => {
    const idx = s.steps.flatMap((t, i) => (kept.has(t) ? [i] : []))
    return { ...s, steps: idx.map((i) => s.steps[i] as number), values: idx.map((i) => s.values[i] as number) }
  })
  return modelRun({
    structure: record.structure,
    steps,
    series,
    status: record.status,
    ...(record.method === undefined ? {} : { method: record.method }),
  })
}
