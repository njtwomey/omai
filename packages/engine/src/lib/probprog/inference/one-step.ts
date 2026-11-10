/**
 * The method of an inference that is one computation and not an iteration (conjugate updates, linear-Gaussian
 * conditioning, junction-tree enumeration, forward-backward, rejection ABC): honestly one step. It gets what every method
 * has (knobs, a typed tape, a capability check that says why not, a posterior, anytime knobs that cost nothing here)
 * without pretending to be steppable: the step runs the engine, logs the evidence, and the run is over. Internal to the
 * folder.
 */

import type { Json } from '../../base/index.ts'
import type { DictSpace } from '../../space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  derived,
  type RunStatus,
  scalar,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import type { Posterior } from '../posterior.ts'
import { guard, type MethodModel, type ModelAnalysis, posteriorJson } from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type MethodFamily, type Verdict } from './method.ts'

/** The state of a one-step method: whether it has run, how it ended and the evidence it found. */
export interface OneShotState {
  readonly done: boolean
  readonly status: RunStatus
  readonly logEvidence: number | null
}

/** The paths of a one-step method under a prefix. */
export type OneShotPath<P extends string> = `${P}/logEvidence` | `${P}/snapshot`

/** What defines a one-step method. */
export interface OneShotDefinition<P extends string> {
  /** The method's name, and the prefix of its paths. */
  readonly name: P
  readonly family: MethodFamily
  readonly description: string
  readonly config: DictSpace
  readonly model: MethodModel
  readonly check: (a: ModelAnalysis) => Verdict
  /** Run the engine. */
  readonly run: (config: Config) => Posterior
}

/**
 * A method of one step. Its posterior is the engine's, computed once per configuration and kept (the state is only the
 * outcome, so it serialises; the engines here are deterministic, so a resumed run reads the same posterior).
 *
 * @param def - The definition.
 * @returns The method.
 */
export function oneShotMethod<P extends string>(
  def: OneShotDefinition<P>,
): InferenceMethod<OneShotState, Config, OneShotPath<P>, ModelAnalysis> {
  const cache = new Map<string, Posterior>()
  const posterior = (_state: OneShotState, config: Config): Posterior => {
    const id = JSON.stringify(config)
    let p = cache.get(id)
    if (p === undefined) {
      p = def.run(config)
      cache.set(id, p)
    }
    return p
  }
  const ensure = guard(def.name, def.model, def.check)
  const evidence = `${def.name}/logEvidence` as const
  const snapshot = `${def.name}/snapshot` as const
  return defineInferenceMethod<OneShotState, Config, OneShotPath<P>, ModelAnalysis>({
    name: def.name,
    family: def.family,
    description: def.description,
    config: def.config,
    check: def.check,
    posterior,
    contract: (config) =>
      ({
        [evidence]: scalar({ description: 'the log evidence log p(data), when the method finds it' }),
        [snapshot]: derived<OneShotState>({
          cost: 'dear',
          description: 'the posterior as JSON: means, sds, quantiles',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }) as Contract<OneShotState, OneShotPath<P>>,
    make: (config): Algorithm<OneShotState, OneShotPath<P>> => ({
      name: `probprog/${def.name}`,
      config: {},
      init: () => {
        ensure()
        return { done: false, status: 'completed', logEvidence: null }
      },
      step: (s, ctx) => {
        const p = posterior(s, config)
        const logEvidence = p.logEvidence?.value ?? null
        if (logEvidence !== null && Number.isFinite(logEvidence)) ctx.log(evidence as never, logEvidence)
        return { done: true, status: p.status === 'failed' ? 'failed' : 'converged', logEvidence }
      },
      done: (s) => (s.done ? s.status : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j: Json) => stateFromJson(j) as OneShotState,
    }),
  })
}
