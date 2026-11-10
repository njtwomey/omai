/**
 * What the per-engine inference methods share, and nothing that chooses between them: the capability verdict over either
 * analysis, the wrapper that makes an engine's algorithm resumable when it keeps derived structure in a closure, the JSON
 * of a snapshot, and the knob readers. Internal to the folder.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import type { Key } from '../../random/index.ts'
import { type Algorithm, stateFromJson, stateToJson } from '../../trace/index.ts'
import type { InferenceEngine } from '../engines/engine.ts'
import type { SpecAnalysis } from '../model/analysis.ts'
import type { Posterior } from '../posterior.ts'
import type { Verdict } from './method.ts'
import { analysisOf, isProgramAnalysis, type MethodModel, type ModelAnalysis, programOf } from './model-input.ts'

export { analysisOf, isProgramAnalysis, type MethodModel, type ModelAnalysis, programOf }

/** A verdict that the method applies. */
export const yes = (reason: string): Verdict => ({ applies: true, reason, missing: [] })

/**
 * What a verdict's reason adds when the spec has no observed node: the model is a density (the funnel, the banana), the
 * target is the prior, and a sampler or an approximation of it is a sensible thing to run. Empty for a model with data.
 */
export const priorOnly = (a: { readonly hasLikelihood: boolean }): string =>
  a.hasLikelihood ? '' : '; no observed nodes, so the target is the prior (a density)'

/** A verdict that it does not, with what is missing and what would help. */
export const no = (reason: string, missing: readonly string[], hint?: string): Verdict => ({
  applies: false,
  reason,
  missing,
  ...(hint === undefined ? {} : { hint }),
})

/**
 * A capability check over either analysis: the method's own reading of a spec's features, and, for a program's
 * analysis, the verdict of the engine it wraps (the engine's `applies`, so the two never disagree about a program).
 */
export function checkOf(engine: InferenceEngine, ofSpec: (a: SpecAnalysis) => Verdict): (a: ModelAnalysis) => Verdict {
  return (a) => {
    if (!isProgramAnalysis(a)) return ofSpec(a)
    const f = engine.applies(a)
    return { applies: f.applies, reason: f.reason, missing: [] }
  }
}

/** A number knob of a configuration. */
export function num(config: Readonly<Record<string, unknown>>, name: string): number {
  const v = config[name]
  if (typeof v !== 'number') throw new OmaiError('value', `the knob '${name}' is not a number`)
  return v
}

/** A flag knob of a configuration. */
export const flag = (config: Readonly<Record<string, unknown>>, name: string): boolean => config[name] === true

/** A choice knob of a configuration. */
export function pick(config: Readonly<Record<string, unknown>>, name: string): string {
  const v = config[name]
  if (typeof v !== 'string') throw new OmaiError('value', `the knob '${name}' is not a choice`)
  return v
}

/**
 * Make an engine's algorithm resumable. Several engines derive structure in `init` and keep it in a closure (the blocks of
 * a Gibbs sweep, the supports of the parameters of SVI, the summaries' scale of ABC), so a state restored into a fresh
 * algorithm would step without it. The wrapper takes the closure's structure from one `init` at a key of its own (given
 * once, not the session's), in a fresh run, before the first restore and before the first step of a state given from
 * outside (a fork), so the structure is the same in all three and a
 * resumed run equals an uninterrupted one. The state serialises with `stateToJson` unless a codec is given.
 *
 * @param inner - The engine's algorithm.
 * @param primeKey - The key the closure's structure is derived at.
 * @param codec - How a state becomes JSON and back (default `stateToJson` and `stateFromJson`).
 * @returns The algorithm with `serialise` and `deserialise`, and `prime`, which sets the structure up (for a method whose
 *   state wraps the engine's and so decodes it itself).
 */
export function restorable<S, K extends string = string>(
  inner: Algorithm<S, K>,
  primeKey: Key,
  codec: { readonly to: (s: S) => Json; readonly from: (j: Json) => S } = {
    to: (s) => stateToJson(s),
    from: (j) => stateFromJson(j) as S,
  },
): Algorithm<S, K> & { readonly prime: () => void } {
  let primed = false
  const prime = (): void => {
    if (primed) return
    inner.init(primeKey)
    primed = true
  }
  return {
    ...inner,
    init: () => {
      primed = true
      return inner.init(primeKey)
    },
    // A session forked or started from a state never runs `init` or `deserialise` on its own algorithm, so the first step
    // sets the structure up.
    step: (s, ctx) => {
      prime()
      return inner.step(s, ctx)
    },
    serialise: codec.to,
    deserialise: (j) => {
      prime()
      return codec.from(j)
    },
    prime,
  }
}

/** A number as JSON (a non-finite one is `null`). */
export const jsonNumber = (x: number): number | null => (Number.isFinite(x) ? x : null)

/**
 * A posterior as the JSON a page plots while a method runs: the engine, the status, the evidence, the diagnostics and
 * the summary table (mean, sd, median, interval and, for chains, ESS and R-hat) of every coordinate.
 *
 * @param p - The posterior.
 * @returns JSON.
 */
export function posteriorJson(p: Posterior): Json {
  let rows: Json = []
  try {
    rows = p.summary().map((r) => ({
      site: r.site,
      name: r.name,
      mean: jsonNumber(r.mean),
      sd: jsonNumber(r.sd),
      median: jsonNumber(r.median),
      lower: jsonNumber(r.lower),
      upper: jsonNumber(r.upper),
      ...(r.ess === undefined ? {} : { ess: jsonNumber(r.ess) }),
      ...(r.rhat === undefined ? {} : { rhat: jsonNumber(r.rhat) }),
    }))
  } catch {
    rows = []
  }
  return {
    engine: p.engine,
    form: p.form,
    status: p.status,
    sites: [...p.sites],
    ...(p.logEvidence === undefined ? {} : { logEvidence: jsonNumber(p.logEvidence.value) }),
    diagnostics: JSON.parse(
      JSON.stringify(p.diagnostics, (_, v) => (typeof v === 'number' && !Number.isFinite(v) ? null : v)),
    ),
    summary: rows,
  }
}

/**
 * The self-check every method runs before it starts: `check` on the model's analysis (made once, when first needed), and the
 * verdict's reason thrown as the error when the method does not apply. A method does this in its `init`, so a page that
 * greys the method out with the same verdict and a run that starts anyway say the same thing.
 *
 * @param method - The method's name, for the message.
 * @param model - The model.
 * @param check - The method's check (none: it applies to anything).
 * @returns A function to call at the start of a run.
 */
export function guard(
  method: string,
  model: MethodModel,
  check: ((a: ModelAnalysis) => Verdict) | undefined,
): () => void {
  let checked = false
  return () => {
    if (checked || check === undefined) return
    const v = check(analysisOf(model))
    if (!v.applies)
      throw new OmaiError('value', `${method}: ${v.reason}${v.hint === undefined ? '' : ` (${v.hint})`}`, {
        missing: [...v.missing],
      })
    checked = true
  }
}
