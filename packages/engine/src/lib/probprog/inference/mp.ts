/**
 * Message passing as inference methods on model specs: expectation propagation (`epSpecMethod`), variational message
 * passing (`vmpSpecMethod`) and belief propagation (`bpSpecMethod`), each its own method with the knobs of its engine
 * (schedule, granularity, damping, improper policy, tolerance, sweep cap) and the typed tape of
 * `omai-engine/message-passing` (`epContract` and `vmpContract`: per-sweep change, the evidence or bound, `message`
 * events and the marginals by variable id), plus the budget and snapshot knobs every inference method has. A bound spec
 * is compiled once to a factor graph (`toFactorGraph`: plates expanded, ids as in `ids.ts`); a step is one sweep (or one
 * factor, or one message, by `granularity`); the state is the messages, so a run serialises, resumes, seeks and forks; and
 * `posterior(state)` reads any state as the library's `Posterior` keyed by site name, in the spec's shapes, which is what
 * an anytime snapshot shows. `check` says whether the model has a factor graph at all and why not, naming the node and
 * the reason (an opaque `custom` call, a family with no message family, a random scale, a non-affine operator), and
 * whether the method has a message rule for every operator of the graph (VMP has none for `probit` or `dot-constant`).
 * Nothing chooses among the three: the user does, and a method that does not apply refuses with that reason.
 */

import { OmaiError } from '../../base/index.ts'
import {
  bpMethod,
  type CompiledGraph,
  compileGraph,
  type EpPath,
  epDefaults,
  epMethod,
  defaultOperators as factorOperators,
  type MpState,
  messagePassingKnobs,
  type VmpPath,
  vmpMethod,
} from '../../message-passing/index.ts'
import {
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
} from '../../trace/index.ts'
import { analyseSpec } from '../model/analysis.ts'
import {
  type FactorGraphCompilation,
  factorGraphPosterior,
  factorGraphVerdict,
  type MessageRule,
  toFactorGraph,
} from '../model/factors.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import { guard, isProgramAnalysis, type MethodModel, type ModelAnalysis, no, posteriorJson, yes } from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths an expectation propagation or belief propagation method logs: the engine's, and the posterior of a state as JSON. */
export type EpSpecPath = EpPath | 'mp/posterior'

/** The paths a variational message passing method logs: the engine's, and the posterior of a state as JSON. */
export type VmpSpecPath = VmpPath | 'mp/posterior'

const TITLE: Readonly<Record<MessageRule, string>> = {
  ep: 'expectation propagation',
  vmp: 'variational message passing',
  bp: 'belief propagation',
}

/** What a model compiles to, made once per method: the compilation, its graph, and the reason when it has none. */
interface Prepared {
  readonly compiled: FactorGraphCompilation
  readonly graph: CompiledGraph | undefined
  readonly failure: string | undefined
}

function prepare(model: MethodModel): Prepared | undefined {
  if (typeof model === 'function') return undefined
  const compiled = toFactorGraph(model, { analysis: analyseSpec(model) })
  if (compiled.graph === undefined) return { compiled, graph: undefined, failure: undefined }
  try {
    return { compiled, graph: compileGraph(compiled.graph, factorOperators()), failure: undefined }
  } catch (e) {
    return { compiled, graph: undefined, failure: e instanceof Error ? e.message : String(e) }
  }
}

/** What the analysis alone says, before the graph is built: the features a message-passing method needs. */
function staticVerdict(a: ModelAnalysis, rule: MessageRule): Verdict | undefined {
  if (isProgramAnalysis(a)) return undefined
  const title = TITLE[rule]
  if (a.opaque.length > 0)
    return no(
      `${title} needs an exponential-family factor graph: ${a.opaque.map((n) => `'${n}'`).join(', ')} ${a.opaque.length > 1 ? 'are' : 'is'} opaque to message passing (a custom call, or a family with no exponential-family form)`,
      ['exponential-family factors'],
      'use a sampling or variational method (nuts, svi), or write the node with operators of the registry',
    )
  if (a.factors.length > 0)
    return no(
      `${title} cannot take the log-density term${a.factors.length > 1 ? 's' : ''} ${a.factors.map((n) => `'${n}'`).join(', ')}: a factor node has no message operator`,
      ['factor operators'],
      'express the term as a distribution, or use a sampling method',
    )
  if (a.params.length > 0)
    return no(
      `${title} has no counterpart for the param node${a.params.length > 1 ? 's' : ''} ${a.params.map((n) => `'${n}'`).join(', ')}: a param is a learnable value, not a random variable`,
      ['random variables only'],
      'give it a prior (lift), or use svi',
    )
  if (rule === 'bp' && a.continuous.length > 0)
    return no(
      `belief propagation sums over finite discrete variables, and ${a.continuous.map((n) => `'${n}'`).join(', ')} ${a.continuous.length > 1 ? 'are' : 'is'} continuous`,
      ['finite discrete latents only'],
      'use expectation propagation for Gaussian and conjugate models',
    )
  if (a.unboundedDiscrete.length > 0)
    return no(
      `${title} has no message family for the unbounded discrete latent${a.unboundedDiscrete.length > 1 ? 's' : ''} ${a.unboundedDiscrete.map((n) => `'${n}'`).join(', ')}`,
      ['finite discrete latents'],
      'bound the support, or use a sampling method',
    )
  return undefined
}

function specMethod<P extends string>(
  rule: MessageRule,
  model: MethodModel,
  options: { readonly name?: string },
): InferenceMethod<MpState, Config, P, ModelAnalysis> {
  const name = options.name ?? rule
  const title = TITLE[rule]
  const prepared = prepare(model)
  const knobs = messagePassingKnobs(rule)
  const check = (a: ModelAnalysis): Verdict => {
    if (prepared === undefined)
      return no(
        `${title} runs on a model spec: a closure program has no factor graph (lifting a program to a spec is not provided)`,
        ['a model spec'],
        'write the model with defineModel',
      )
    const s = staticVerdict(a, rule)
    if (s !== undefined) return s
    if (prepared.failure !== undefined)
      return no(`the factor graph does not compile: ${prepared.failure}`, ['a valid factor graph'], 'see toFactorGraph')
    const v = factorGraphVerdict(prepared.compiled, prepared.graph, rule)
    return v.ok
      ? yes(v.reason)
      : no(
          v.reason,
          v.missing,
          rule === 'vmp'
            ? 'expectation propagation has message rules for more operators'
            : 'see toFactorGraph(model).issues',
        )
  }
  const ensure = guard(name, model, check)
  const why = (): string =>
    prepared === undefined
      ? 'a closure program has no factor graph'
      : (prepared.failure ?? prepared.compiled.issues.map((i) => `${i.node}: ${i.reason}`).join('; '))
  const tol = (config: Config): number => (typeof config['tol'] === 'number' ? config['tol'] : epDefaults.tol)
  const posterior = (state: MpState, config: Config): Posterior => {
    if (prepared?.graph === undefined)
      return definePosterior({
        form: 'variational',
        engine: name,
        sites: [],
        status: 'failed',
        diagnostics: { reason: why() },
      })
    const converged = state.sweeps >= 1 && state.cursor === 0 && state.change < tol(config)
    return factorGraphPosterior(prepared.compiled, prepared.graph, state, {
      engine: name,
      rule,
      status: converged ? 'converged' : 'stopped',
    })
  }
  const base = knobs.contract as Contract<MpState, string>
  const contract = (): Contract<MpState, P> =>
    defineContract({
      ...base,
      'mp/posterior': derived<MpState>({
        cost: 'dear',
        description:
          'the posterior of the state as JSON: the marginal of every site, in the spec shapes, with the evidence or bound',
        compute: (s) => posteriorJson(posterior(s, {})),
      }),
    } as never) as Contract<MpState, P>
  return defineInferenceMethod<MpState, Config, P, ModelAnalysis>({
    name,
    family: 'message-passing',
    description: `${title} on the factor graph of a model spec: one sweep per step, marginals by site`,
    config: knobs.config,
    check,
    posterior,
    defaults: (rule === 'vmp'
      ? { 'mp/change': Log(), 'mp/elbo': Log() }
      : { 'mp/change': Log(), 'mp/evidence': Log() }) as never,
    contract,
    make: (config, key) => {
      ensure()
      const g = prepared?.graph
      if (g === undefined) throw new OmaiError('value', `${name}: ${why()}`)
      const m = rule === 'ep' ? epMethod(g) : rule === 'vmp' ? vmpMethod(g) : bpMethod(g)
      return m.make(config, key) as never
    },
  })
}

/**
 * Expectation propagation on a bound spec: the spec's factor graph (plates expanded, Gaussian, gamma, beta, Dirichlet and
 * Bernoulli variables, affine maps, probit and logistic links, conjugate likelihoods) run with moment matching, one
 * sweep per step. The marginals come back as a `Posterior` by site name; the tape has the engine's `mp/*` paths.
 *
 * @param model - A bound spec.
 * @param options - A name for the method (default `ep`).
 * @returns The method, with `check` (why a model has no factor graph, or an operator no rule) and `posterior`.
 * @concept expectation-propagation
 * @index estimator Expectation propagation on a model spec's factor graph, posterior by site.
 * @example
 * ```ts
 * import { epSpecMethod } from 'omai-engine/probprog/inference'
 * import { defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * import { session } from 'omai-engine/trace'
 * const spec = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 2)), 1), b.data('y', { dims: ['row'] })) })
 * const m = epSpecMethod(spec.bind({ y: [1, 2, 3] }))
 * const s = session(m, { steps: 50, mode: 'eager' })
 * [s.status, Number(m.posterior?.(s.state, s.config).mean('mu').data[0]).toFixed(4)] // => ['converged', '1.8462']
 * ```
 */
export function epSpecMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<MpState, Config, EpSpecPath, ModelAnalysis> {
  return specMethod<EpSpecPath>('ep', model, options)
}

/**
 * Variational message passing on a bound spec: the expected-log message of every factor, so each variable's belief is the
 * mean-field distribution that minimises KL(q || p), one sweep per step, with the variational bound on the tape. It applies
 * where every operator of the graph has a variational rule (Gaussian and conjugate models; `check` names the operators
 * that have none, such as `probit` and the affine maps).
 *
 * @param model - A bound spec.
 * @param options - A name for the method (default `vmp`).
 * @returns The method.
 * @concept variational-message-passing
 * @index estimator Variational message passing on a model spec's factor graph, posterior by site.
 * @example
 * ```ts
 * import { vmpSpecMethod } from 'omai-engine/probprog/inference'
 * import { defineModel } from 'omai-engine/probprog/model'
 * import { beta, bernoulli } from 'omai-engine/probprog/model/dists'
 * import { session } from 'omai-engine/trace'
 * const spec = defineModel('coin', (b) => { b.observe('y', bernoulli({ probs: b.latent('p', beta(1, 1)) }), b.data('y', { dims: ['flip'], support: 'binary' })) })
 * const m = vmpSpecMethod(spec.bind({ y: [1, 1, 0, 1] }))
 * const s = session(m, { steps: 50, mode: 'eager' })
 * Number(m.posterior?.(s.state, s.config).mean('p').data[0]).toFixed(4) // => '0.6667'
 * ```
 */
export function vmpSpecMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<MpState, Config, VmpSpecPath, ModelAnalysis> {
  return specMethod<VmpSpecPath>('vmp', model, options)
}

/**
 * Belief propagation on a bound spec of finite discrete variables: Bernoulli nodes with tables of their parents' states
 * as factors, sum-product messages (exact on a tree, `schedule: 'tree'`), the Bethe estimate of the evidence. Continuous
 * latents are refused with the reason.
 *
 * @param model - A bound spec.
 * @param options - A name for the method (default `bp`).
 * @returns The method.
 * @concept belief-propagation
 * @index estimator Belief propagation on a model spec of Bernoulli nodes, posterior by site.
 * @example
 * ```ts
 * import { bpSpecMethod } from 'omai-engine/probprog/inference'
 * import { defineModel, ops } from 'omai-engine/probprog/model'
 * import { bernoulli } from 'omai-engine/probprog/model/dists'
 * import { session } from 'omai-engine/trace'
 * const spec = defineModel('chain', (b) => {
 *   const rain = b.latent('rain', bernoulli({ probs: 0.3 }))
 *   b.observe('wet', bernoulli({ probs: ops.where(rain, 0.9, 0.1) }), b.data('wet', { support: 'binary' }))
 * })
 * const m = bpSpecMethod(spec.bind({ wet: 1 }))
 * const s = session(m, { config: { schedule: 'tree' }, steps: 20, mode: 'eager' })
 * Number(m.posterior?.(s.state, s.config).mean('rain').data[0]).toFixed(4) // => '0.7941'
 * ```
 */
export function bpSpecMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<MpState, Config, EpSpecPath, ModelAnalysis> {
  return specMethod<EpSpecPath>('bp', model, options)
}
