/**
 * Importance sampling as an inference method: batches of draws, each step one batch (keyed by the draw's index, so a run
 * taken in batches draws what the one-call `importanceInference` draws at the same key). The draws and their log weights
 * so far are the state, which serialises, so a run resumes and forks; `posterior` reads the weighted posterior of the draws so
 * far with the evidence estimate, the effective sample size and the Pareto k-hat, which improve while the run goes on.
 */

import type { Json } from '../../base/index.ts'
import { foldIn } from '../../random/index.ts'
import { importanceEss, logEvidence } from '../../sampling/index.ts'
import { bool, type DictSpace, dict, int } from '../../space/index.ts'
import { concat, type Tensor } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import { importanceDraw, importanceEngine, importancePosterior, stackSiteValues } from '../engines/importance.ts'
import type { ProgramTrace } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import {
  checkOf,
  flag,
  guard,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  posteriorJson,
  programOf,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod } from './method.ts'

/** The paths the importance method logs. */
export type ImportancePath = 'importance/draws' | 'importance/ess' | 'importance/logEvidence' | 'importance/snapshot'

/** The state of {@link importanceMethod}: the batches of draws taken so far, as sites stacked per batch, and their log weights. */
export interface ImportanceRunState {
  readonly n: number
  readonly batches: readonly Readonly<Record<string, Tensor>>[]
  readonly logWeights: readonly Float64Array[]
}

/** The knobs of {@link importanceMethod} (the budget and snapshot knobs are added to them). */
export const importanceKnobs: DictSpace = dict({
  numSamples: int(2, 100_000_000, { default: 2000, label: 'draws', description: 'the run ends after this many draws' }),
  batchSize: int(1, 1_000_000, { default: 100, label: 'batch size', description: 'draws per step' }),
  smooth: bool({
    default: false,
    label: 'Pareto smoothing',
    description: 'use Pareto-smoothed weights for the posterior',
  }),
})

const merge = (batches: readonly Readonly<Record<string, Tensor>>[]): Record<string, Tensor> => {
  const out: Record<string, Tensor> = {}
  for (const name of Object.keys(batches[0] ?? {})) {
    const parts = batches.map((b) => b[name] as Tensor)
    out[name] = parts.length === 1 ? (parts[0] as Tensor) : (concat(parts, 0) as Tensor)
  }
  return out
}

/**
 * Importance sampling on a model as a method with a fine step: one batch of draws per step, from the prior weighted by
 * the likelihood of the observed sites (or from a proposal program, given as an option: a program is not a knob). The state
 * serialises; `posterior(state)` is the weighted posterior of the draws so far with the evidence, ESS and Pareto k-hat. The
 * run ends `completed` after `numSamples` draws.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The name, and a proposal program over the latent sites (default the prior).
 * @returns The method, with `check` (a likelihood and latent nodes) and `posterior`.
 * @concept importance-sampling
 * @index estimator Importance sampling in batches, with evidence, ESS and Pareto k-hat as it goes.
 * @example
 * ```ts
 * import { importanceMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const method = importanceMethod(model)
 * const s = session(method, { steps: 100, mode: 'eager', config: { numSamples: 4000, batchSize: 500 } })
 * Math.abs(item(method.posterior?.(s.state, s.config).mean('m') ?? 0) - 0.5) < 0.05 // => true
 * ```
 */
export function importanceMethod(
  model: MethodModel,
  options: { readonly name?: string; readonly proposal?: Model } = {},
): InferenceMethod<ImportanceRunState, Config, ImportancePath, ModelAnalysis> {
  const program = programOf(model)
  const posterior = (state: ImportanceRunState, config: Config): Posterior => {
    if (state.n < 2)
      return definePosterior({
        form: 'weighted',
        engine: 'importance',
        sites: [],
        status: 'failed',
        draws: {},
        diagnostics: { reason: 'fewer than two draws so far' },
      })
    const lw = new Float64Array(state.n)
    let at = 0
    for (const b of state.logWeights) {
      lw.set(b, at)
      at += b.length
    }
    const p = importancePosterior(merge(state.batches), lw, {
      ...(options.proposal === undefined ? {} : { proposal: options.proposal }),
      smooth: flag(config, 'smooth'),
    })
    return state.n >= num(config, 'numSamples') ? p : { ...p, status: 'stopped' }
  }
  const check = checkOf(importanceEngine(), (a) => {
    if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an observed node')
    if (a.latent.length === 0) return no('no latent nodes', ['latent nodes'])
    return yes(
      `draws from the prior weighted by ${a.observed.length} observed node(s); accuracy falls as the posterior concentrates`,
    )
  })
  const ensure = guard(options.name ?? 'importance', model, check)
  return defineInferenceMethod<ImportanceRunState, Config, ImportancePath, ModelAnalysis>({
    name: options.name ?? 'importance',
    family: 'smc',
    description: 'importance sampling in batches, with the evidence and the Pareto k-hat',
    config: importanceKnobs,
    check,
    posterior,
    defaults: { 'importance/draws': Log(), 'importance/ess': Log(), 'importance/logEvidence': Log() },
    contract: (config): Contract<ImportanceRunState, ImportancePath> =>
      defineContract({
        'importance/draws': scalar({ description: 'draws taken so far' }),
        'importance/ess': scalar({
          description: "Kish's effective sample size of the weights so far",
          direction: 'higher',
        }),
        'importance/logEvidence': scalar({ description: 'the running estimate of log p(data)' }),
        'importance/snapshot': derived<ImportanceRunState>({
          cost: 'dear',
          description:
            'the weighted posterior of the draws so far as JSON: means, sds, quantiles, evidence, Pareto k-hat',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key): Algorithm<ImportanceRunState, ImportancePath> => {
      const total = num(config, 'numSamples')
      const batch = num(config, 'batchSize')
      return {
        name: 'probprog/importance-method',
        config: { numSamples: total, batchSize: batch },
        init: () => {
          ensure()
          return { n: 0, batches: [], logWeights: [] }
        },
        step: (s, ctx) => {
          const b = Math.min(batch, total - s.n)
          const traces: ProgramTrace[] = []
          const lw = new Float64Array(b)
          for (let i = 0; i < b; i++) {
            const d = importanceDraw(
              program,
              undefined,
              { key: foldIn(key, 0), ...(options.proposal === undefined ? {} : { proposal: options.proposal }) },
              s.n + i,
            )
            traces.push(d.trace)
            lw[i] = d.logWeight
          }
          const first = traces[0]
          const names =
            first === undefined
              ? []
              : [...first.sites.values()]
                  .filter((x) => (x.type === 'sample' && !x.isObserved) || x.type === 'deterministic')
                  .map((x) => x.name)
          const next = {
            n: s.n + b,
            batches: [...s.batches, stackSiteValues(traces, names)],
            logWeights: [...s.logWeights, lw],
          }
          ctx.log('importance/draws', next.n)
          const all = Float64Array.from(next.logWeights.flatMap((x) => [...x]))
          if (all.some((v) => Number.isFinite(v))) {
            ctx.log('importance/logEvidence', logEvidence(all).value)
            ctx.log('importance/ess', importanceEss(all))
          }
          return next
        },
        done: (s) => (s.n >= total ? 'completed' : undefined),
        serialise: (s) => stateToJson(s),
        deserialise: (j: Json) => stateFromJson(j) as ImportanceRunState,
      }
    },
  })
}
