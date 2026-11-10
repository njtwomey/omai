/**
 * Readers of inference tapes for the `modelRun` record: `modelRunFromTape` turns a tape (finished, or still running:
 * the events so far) and any anytime snapshots into the beliefs of the nodes of a model graph at each step, the messages
 * with a readable badge, and the convergence series; `describeMessage` reads natural parameters as a distribution;
 * `nodeStateOf` and `nodeStatesOfRows` read one marginal or the rows of a posterior summary. It reads only the
 * documented paths: the message passing paths `mp/marginals`, `mp/message`, `mp/snapshot`, `mp/change`, `mp/evidence`,
 * `mp/elbo`, `mp/improper`, `mp/skipped` and the `<method>/snapshot` posterior JSON of the sampling and variational
 * methods (per-site rows with mean, sd, interval, R-hat and ESS), under any prefix or none.
 */

import type { Json } from '../base/index.ts'
import type { Structure } from '../scene/index.ts'
import {
  type MessageBadge,
  type ModelRun,
  type ModelRunSeries,
  type ModelRunStep,
  modelRun,
  type NodeState,
} from './model-run.ts'

/** The part of a tape event this reads: a `Run`'s events have these fields, so a tape can be handed over as it is. */
export interface TapeEventLike {
  readonly type: string
  readonly step?: number
  readonly name?: string
  readonly value?: number
  readonly kind?: string
  readonly from?: string
  readonly to?: string
  readonly payload?: unknown
  readonly frame?: unknown
  readonly status?: string
}

/** A tape: finished, or the events so far of a run in progress. */
export interface TapeLike {
  readonly events: readonly TapeEventLike[]
}

/** An anytime snapshot (`runAnytime`'s callback) with the step it was taken at. */
export interface SnapshotAt {
  readonly step: number
  readonly snapshot: unknown
}

/** Options of {@link modelRunFromTape}. */
export interface ModelRunOptions {
  /** The path prefix of the method's tape, `mp/` (default: found from the tape, or none). */
  readonly prefix?: string
  /** The most steps kept, thinned evenly with the last always kept (default 300). */
  readonly maxSteps?: number
  /** The most message badges kept per step (default 24); `0` keeps none. */
  readonly maxMessages?: number
  /** The method's name, for the record (default the tape's `run` event's). */
  readonly method?: string
}

const num = (x: unknown): number | undefined => (typeof x === 'number' && Number.isFinite(x) ? x : undefined)
const isObject = (x: unknown): x is Readonly<Record<string, unknown>> =>
  typeof x === 'object' && x !== null && !Array.isArray(x)

/** A number for a line of text: three significant digits, no trailing noise. */
const fmt = (x: number): string => String(Number(x.toPrecision(3)))

/**
 * Natural parameters as a distribution in words. The layouts are those of the message passing families: a Gaussian is
 * `[h, J]` (mean `h / J`, variance `1 / J`), a gamma `[shape - 1, -rate]`, a beta `[a - 1, b - 1]`, a Bernoulli its
 * logit, a discrete message log weights, a Dirichlet `alpha - 1`, a multivariate Gaussian `h` then the precision matrix.
 * A message that is not proper (a precision at or below zero) is said to be, with its parameters.
 *
 * @param family - The family of the variable the message is for (`gaussian`, `gamma`, `beta`, `bernoulli`, `discrete`, `dirichlet`, `gaussian-mv`); another is written as its parameters.
 * @param eta - The natural parameters.
 * @returns The message in words.
 * @index function Natural parameters of a message as a readable distribution: `describeMessage('gaussian', [h, J])`.
 * @example
 * ```ts
 * import { describeMessage } from 'omai-engine/exposition'
 * describeMessage('gaussian', [0.5, 2]) // => 'N(μ = 0.25, σ² = 0.5)'
 * describeMessage('bernoulli', [0]) // => 'p = 0.5'
 * describeMessage('gaussian', [1, -1]) // => 'improper (precision -1)'
 * ```
 */
export function describeMessage(family: string | undefined, eta: readonly number[]): string {
  const a = eta[0] ?? Number.NaN
  const b = eta[1] ?? Number.NaN
  switch (family) {
    case 'gaussian':
      return b > 0 ? `N(μ = ${fmt(a / b)}, σ² = ${fmt(1 / b)})` : `improper (precision ${fmt(b)})`
    case 'gamma':
      return a + 1 > 0 && -b > 0
        ? `Gamma(${fmt(a + 1)}, rate ${fmt(-b)})`
        : `improper (shape ${fmt(a + 1)}, rate ${fmt(-b)})`
    case 'beta':
      return a + 1 > 0 && b + 1 > 0 ? `Beta(${fmt(a + 1)}, ${fmt(b + 1)})` : `improper (${fmt(a + 1)}, ${fmt(b + 1)})`
    case 'bernoulli':
      return `p = ${fmt(1 / (1 + Math.exp(-a)))}`
    case 'discrete': {
      const top = Math.max(...eta)
      const w = eta.map((v) => Math.exp(v - top))
      const total = w.reduce((s, v) => s + v, 0)
      return `[${w.map((v) => fmt(v / total)).join(', ')}]`
    }
    case 'dirichlet':
      return `Dir(${eta.map((v) => fmt(v + 1)).join(', ')})`
    case 'gaussian-mv': {
      const d = Math.round((-1 + Math.sqrt(1 + 4 * eta.length)) / 2)
      const prec = eta.slice(d)
      const diag = Array.from({ length: d }, (_, i) => prec[i * d + i] ?? 0)
      return diag.every((v) => v > 0)
        ? `N-${d}(precision diag ${diag.map(fmt).join(', ')})`
        : `improper (precision diag ${diag.map(fmt).join(', ')})`
    }
    default:
      return `η = [${eta.map(fmt).join(', ')}]`
  }
}

/**
 * The belief of one marginal as the message passing run reports it (`mp/marginals`): `{ family, mean, variance }` for
 * a Gaussian, gamma or beta, `{ p }` for a Bernoulli, `{ probs }` for a discrete variable, `{ mean: [..], variance: [..] }`
 * for a vector, `{ observed }` for an observed node, `improper: true` where there is no density yet.
 *
 * @param marginal - The marginal's JSON.
 * @returns The node state, or `undefined` when it is not an object.
 * @example
 * ```ts
 * import { nodeStateOf } from 'omai-engine/exposition'
 * nodeStateOf({ family: 'gaussian', mean: 0.5, variance: 4 })?.text // => '0.5 ± 2'
 * nodeStateOf({ family: 'bernoulli', p: 0.7 })?.text // => 'p = 0.7'
 * nodeStateOf({ family: 'gaussian', observed: 2 })?.text // => 'observed 2'
 * ```
 */
export function nodeStateOf(marginal: unknown): NodeState | undefined {
  if (!isObject(marginal)) return undefined
  const family = typeof marginal['family'] === 'string' ? marginal['family'] : undefined
  const withFamily = family === undefined ? {} : { family }
  if (marginal['observed'] !== undefined)
    return {
      ...withFamily,
      observed: marginal['observed'] as Json,
      text: `observed ${JSON.stringify(marginal['observed'])}`,
    }
  if (marginal['improper'] === true) return { ...withFamily, improper: true, text: 'improper' }
  const probs = Array.isArray(marginal['probs'])
    ? ((marginal['probs'] as unknown[]).filter((v) => num(v) !== undefined) as number[])
    : undefined
  if (probs !== undefined && probs.length > 0) {
    const best = probs.indexOf(Math.max(...probs))
    return { ...withFamily, probs, text: `state ${best} (${fmt(probs[best] as number)})` }
  }
  const p = num(marginal['p'])
  if (p !== undefined)
    return { ...withFamily, mean: p, sd: Math.sqrt(p * (1 - p)), probs: [1 - p, p], text: `p = ${fmt(p)}` }
  const mean = marginal['mean']
  const variance = marginal['variance']
  if (Array.isArray(mean)) {
    const elements = (mean as unknown[]).flatMap((m, i) => {
      const v = Array.isArray(variance) ? num(variance[i]) : undefined
      const mu = num(m)
      return mu === undefined ? [] : [{ name: `[${i}]`, mean: mu, sd: v === undefined ? 0 : Math.sqrt(Math.max(0, v)) }]
    })
    if (elements.length === 0) return undefined
    const avg = elements.reduce((s, e) => s + e.mean, 0) / elements.length
    return { ...withFamily, mean: avg, elements, text: `${elements.length} coords, mean ${fmt(avg)}` }
  }
  const mu = num(mean)
  if (mu === undefined) return undefined
  const v = num(variance)
  return v === undefined
    ? { ...withFamily, mean: mu, text: fmt(mu) }
    : { ...withFamily, mean: mu, sd: Math.sqrt(Math.max(0, v)), text: `${fmt(mu)} ± ${fmt(Math.sqrt(Math.max(0, v)))}` }
}

/** The rows of a posterior summary (`<method>/snapshot`'s `summary`) as node states by site, and by coordinate name. */
interface Row {
  readonly site: string
  readonly name: string
  readonly mean: number
  readonly sd: number
  readonly lower?: number
  readonly upper?: number
  readonly rhat?: number
  readonly ess?: number
}

function rowOf(x: unknown): Row | undefined {
  if (!isObject(x) || typeof x['site'] !== 'string' || typeof x['name'] !== 'string') return undefined
  const mean = num(x['mean'])
  const sd = num(x['sd']) ?? 0
  if (mean === undefined) return undefined
  const lower = num(x['lower'])
  const upper = num(x['upper'])
  const rhat = num(x['rhat'])
  const ess = num(x['ess'])
  return {
    site: x['site'],
    name: x['name'],
    mean,
    sd,
    ...(lower === undefined ? {} : { lower }),
    ...(upper === undefined ? {} : { upper }),
    ...(rhat === undefined ? {} : { rhat }),
    ...(ess === undefined ? {} : { ess }),
  }
}

/**
 * Node states from the rows of a posterior summary: one state per site (a scalar site's own row, or for a vector site
 * the mean of its coordinates' means with the coordinates as `elements`, and the worst R-hat and the least ESS), and one
 * per coordinate under its name (`w[3]`), so a diagram that draws elements finds each.
 *
 * @param rows - The `summary` of a `<method>/snapshot`.
 * @returns The states by node id.
 * @example
 * ```ts
 * import { nodeStatesOfRows } from 'omai-engine/exposition'
 * const s = nodeStatesOfRows([{ site: 'w', name: 'w[0]', mean: 1, sd: 0.5 }, { site: 'w', name: 'w[1]', mean: 3, sd: 0.5 }])
 * [s['w']?.text, s['w[1]']?.text] // => ['2 coords, mean 2', '3 ± 0.5']
 * ```
 */
export function nodeStatesOfRows(rows: readonly unknown[]): Record<string, NodeState> {
  const out: Record<string, NodeState> = {}
  const bySite = new Map<string, Row[]>()
  for (const r of rows.map(rowOf)) {
    if (r === undefined) continue
    bySite.set(r.site, [...(bySite.get(r.site) ?? []), r])
    out[r.name] = stateOfRow(r)
  }
  for (const [site, rs] of bySite) {
    if (rs.length === 1 && rs[0]?.name === site) continue
    const avg = rs.reduce((s, r) => s + r.mean, 0) / rs.length
    const rhats = rs.flatMap((r) => (r.rhat === undefined ? [] : [r.rhat]))
    const esss = rs.flatMap((r) => (r.ess === undefined ? [] : [r.ess]))
    out[site] = {
      mean: avg,
      elements: rs.map((r) => ({ name: r.name, mean: r.mean, sd: r.sd })),
      ...(rhats.length === 0 ? {} : { rhat: Math.max(...rhats) }),
      ...(esss.length === 0 ? {} : { ess: Math.min(...esss) }),
      text: `${rs.length} coords, mean ${fmt(avg)}`,
    }
  }
  return out
}

function stateOfRow(r: Row): NodeState {
  return {
    mean: r.mean,
    sd: r.sd,
    ...(r.lower === undefined ? {} : { lower: r.lower }),
    ...(r.upper === undefined ? {} : { upper: r.upper }),
    ...(r.rhat === undefined ? {} : { rhat: r.rhat }),
    ...(r.ess === undefined ? {} : { ess: r.ess }),
    text: `${fmt(r.mean)} ± ${fmt(r.sd)}`,
  }
}

const SERIES: Readonly<Record<string, { readonly label: string; readonly direction?: 'higher' | 'lower' }>> = {
  change: { label: 'largest change of a belief', direction: 'lower' },
  evidence: { label: 'log evidence estimate', direction: 'higher' },
  elbo: { label: 'variational lower bound', direction: 'higher' },
  logEvidence: { label: 'log evidence', direction: 'higher' },
  objective: { label: 'objective' },
  improper: { label: 'improper updates' },
  skipped: { label: 'updates skipped' },
  damped: { label: 'updates damped' },
  reset: { label: 'messages reset' },
  divergences: { label: 'divergent transitions', direction: 'lower' },
}

/**
 * The record of inference on a model graph, read from a tape. The beliefs come from the marginals and snapshots on the
 * tape (`mp/marginals`, `mp/snapshot`, the frame's marginals, or a method's `<name>/snapshot` summary rows), keyed by
 * node id, and only for ids the structure has; the messages from `mp/message` events (`from` a factor id, `to` a
 * variable: a factor of a plate diagram, which has no factor nodes, stands for its site) with their natural parameters
 * written by {@link describeMessage} for the family of the variable; the series from the metrics `change`, `evidence`,
 * `elbo`, `improper`, `skipped`, `damped`, `reset`. A tape with no `end` event, or only anytime snapshots, makes a
 * record with status `running`, so the same call serves a run in progress: call it again with more events.
 *
 * @param structure - The model graph: `toStructure(model)` or `graphStructure(spec)`.
 * @param source - The tape and/or the anytime snapshots (`{ step, snapshot }`, as `runAnytime` hands them out).
 * @param options - The path prefix, how many steps and messages to keep.
 * @returns The record.
 * @concept structures
 * @index function Model-graph inference record from a tape or anytime snapshots.
 * @example
 * ```ts
 * import { modelRunFromTape } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * const s = structure('model-graph', [{ id: 'x', label: 'x', role: 'latent' }], [])
 * const events = [
 *   { type: 'metric', step: 1, name: 'mp/change', value: 0.5 },
 *   { type: 'message', step: 1, kind: 'mp/marginals', payload: { x: { family: 'gaussian', mean: 0.5, variance: 1 } } },
 * ]
 * const r = modelRunFromTape(s, { run: { events } })
 * [r.status, r.steps[0]?.nodes['x']?.text, r.series[0]?.name] // => ['running', '0.5 ± 1', 'change']
 * ```
 */
export function modelRunFromTape(
  structure: Structure,
  source: { readonly run?: TapeLike; readonly snapshots?: readonly SnapshotAt[] },
  options: ModelRunOptions = {},
): ModelRun {
  const events = source.run?.events ?? []
  const ids = new Set(structure.nodes.map((n) => n.id))
  const prefix =
    options.prefix ??
    (() => {
      for (const e of events) {
        const m = e.type === 'message' ? /^(.*\/)?(marginals|message|snapshot)$/.exec(e.kind ?? '') : null
        if (m !== null) return m[1] ?? ''
      }
      return ''
    })()
  const method = options.method ?? (prefix === '' ? undefined : prefix.replace(/\/$/, ''))
  const at = new Map<number, { nodes: Record<string, NodeState>; raw: TapeEventLike[]; caption?: string }>()
  const cell = (step: number): { nodes: Record<string, NodeState>; raw: TapeEventLike[]; caption?: string } => {
    let c = at.get(step)
    if (c === undefined) {
      c = { nodes: {}, raw: [] }
      at.set(step, c)
    }
    return c
  }
  const series = new Map<string, { steps: number[]; values: number[] }>()
  const push = (name: string, step: number, value: number | undefined): void => {
    if (value === undefined || !Object.hasOwn(SERIES, name)) return
    const s = series.get(name) ?? { steps: [], values: [] }
    if (s.steps.at(-1) === step) s.values[s.values.length - 1] = value
    else {
      s.steps.push(step)
      s.values.push(value)
    }
    series.set(name, s)
  }
  const family = new Map<string, string>()
  const ingestMarginals = (step: number, marginals: unknown): void => {
    if (!isObject(marginals)) return
    for (const [id, m] of Object.entries(marginals)) {
      const state = nodeStateOf(m)
      if (state === undefined) continue
      if (state.family !== undefined) family.set(id, state.family)
      if (ids.has(id)) cell(step).nodes[id] = state
    }
  }
  const ingestSnapshot = (step: number, snap: unknown): void => {
    if (!isObject(snap)) return
    if (snap['marginals'] !== undefined) ingestMarginals(step, snap['marginals'])
    if (Array.isArray(snap['summary'])) {
      for (const [id, state] of Object.entries(nodeStatesOfRows(snap['summary'])))
        if (ids.has(id)) cell(step).nodes[id] = state
    }
    // Snapshots carry the run's progress too: a run known only by its snapshots still has convergence series.
    const sweeps = num(snap['sweeps'])
    if (sweeps !== undefined) cell(step).caption = `sweep ${sweeps}`
    push('change', step, num(snap['change']))
    if (isObject(snap['improper'])) {
      push('improper', step, num(snap['improper']['events']))
      push('skipped', step, num(snap['improper']['skipped']))
    }
    push('logEvidence', step, num(snap['logEvidence']))
  }
  let status: ModelRun['status'] = 'running'
  const name = (kind: string | undefined): string | undefined =>
    kind === undefined
      ? undefined
      : prefix !== '' && kind.startsWith(prefix)
        ? kind.slice(prefix.length)
        : prefix === ''
          ? kind
          : undefined
  for (const e of events) {
    const step = e.step ?? 0
    if (e.type === 'end' && e.status !== undefined) status = e.status as ModelRun['status']
    else if (e.type === 'metric') push(name(e.name) ?? '', step, num(e.value))
    else if (e.type === 'frame' && isObject(e.frame)) ingestMarginals(step, e.frame['marginals'])
    else if (e.type === 'message') {
      const kind = name(e.kind)
      if (kind === 'marginals') ingestMarginals(step, e.payload)
      else if (kind === 'snapshot') ingestSnapshot(step, e.payload)
      else if (kind === 'message') cell(step).raw.push(e)
    }
  }
  for (const s of source.snapshots ?? []) ingestSnapshot(s.step, s.snapshot)
  const resolve = (id: string | undefined): string | undefined => {
    if (id === undefined) return undefined
    if (ids.has(id)) return id
    const site = id.startsWith('factor:') ? id.slice('factor:'.length) : undefined
    return site !== undefined && ids.has(site) ? site : undefined
  }
  const maxMessages = options.maxMessages ?? 24
  const steps: ModelRunStep[] = [...at.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([step, c]) => {
      const messages: MessageBadge[] = []
      for (const m of c.raw) {
        if (messages.length >= maxMessages) break
        const from = resolve(m.from)
        const to = resolve(m.to)
        const p = isObject(m.payload) ? m.payload : undefined
        const eta = Array.isArray(p?.['eta'])
          ? (p['eta'] as unknown[]).flatMap((v) => (num(v) === undefined ? [] : [v as number]))
          : undefined
        if (from === undefined || to === undefined || eta === undefined) continue
        const fam = typeof p?.['family'] === 'string' ? p['family'] : (family.get(m.to ?? '') ?? family.get(to))
        messages.push({
          from,
          to,
          text: describeMessage(fam, eta),
          ...(typeof p?.['operator'] === 'string' ? { operator: p['operator'] } : {}),
        })
      }
      return {
        step,
        nodes: c.nodes,
        ...(messages.length === 0 ? {} : { messages }),
        caption: c.caption ?? `step ${step}`,
      }
    })
  const cap = options.maxSteps ?? 300
  const kept =
    steps.length <= cap
      ? steps
      : steps.filter(
          (_, i) =>
            i === steps.length - 1 ||
            Math.floor((i * (cap - 1)) / (steps.length - 1)) !== Math.floor(((i - 1) * (cap - 1)) / (steps.length - 1)),
        )
  const outSeries: ModelRunSeries[] = [...series.entries()].map(([n, s]) => ({
    name: n,
    label: SERIES[n]?.label ?? n,
    steps: s.steps,
    values: s.values,
    ...(SERIES[n]?.direction === undefined ? {} : { direction: SERIES[n].direction }),
  }))
  return modelRun({ structure, steps: kept, series: outSeries, status, ...(method === undefined ? {} : { method }) })
}
