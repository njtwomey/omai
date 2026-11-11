/**
 * Message passing as inference methods on model specs: expectation propagation (`epSpecMethod`), variational message
 * passing (`vmpSpecMethod`) and belief propagation (`bpSpecMethod`), each its own method with the knobs of its engine
 * (schedule, granularity, damping, improper policy, tolerance, sweep cap) and the typed tape of
 * `omai-engine/message-passing` (`epContract` and `vmpContract`: per-sweep change, the evidence or bound, `message`
 * events and the marginals by variable id), plus the budget and snapshot knobs every inference method has. A bound spec
 * is compiled once to a factor graph (`toFactorGraph`: a regression on a weight vector over a plate is one plated factor,
 * other plates are expanded, ids as in `ids.ts`); a step is one sweep (or one
 * factor, or one message, by `granularity`); the state is the messages, so a run serialises, resumes, seeks and forks; and
 * `posterior(state)` reads any state as the library's `Posterior` keyed by site name, in the spec's shapes, which is what
 * an anytime snapshot shows. `check` says whether the model has a factor graph at all and why not, naming the node and
 * the reason (an opaque `custom` call, a family with no message family, a random scale, a non-affine operator), and
 * whether the method has a message rule for every operator of the graph (VMP has none for a hard comparison, `greater-than`).
 * Nothing chooses among the three: the user does, and a method that does not apply refuses with that reason. Within
 * expectation propagation `epRoute` (`ep-route.ts`) chooses how it runs a model: plain, or inside a grid over hyperparameters
 * (`ep-grid.ts`), or not at all.
 */

import { OmaiError } from '../../base/index.ts'
import {
  bpMethod,
  type CompiledGraph,
  defaultsOf,
  type EpPath,
  epAlgorithm,
  epDefaults,
  epMethod,
  type MpState,
  messagePassingKnobs,
  mpVerdict,
  projectionOf,
  type ScheduleKind,
  takesPower,
  type VmpPath,
  vmpMethod,
  withPowers,
} from '../../message-passing/index.ts'
import { dict, real } from '../../space/index.ts'
import { type Config, type Contract, defineContract, derived, Log, runAlgorithm } from '../../trace/index.ts'
import { factorGraphPosterior, factorGraphVerdict, type MessageRule } from '../model/factors.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import {
  guard,
  isProgramAnalysis,
  type MethodModel,
  type ModelAnalysis,
  no,
  posteriorJson,
  prepare,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths an expectation propagation or belief propagation method logs: the engine's, and the posterior of a state as JSON. */
export type EpSpecPath = EpPath | 'mp/posterior'

/** An expectation propagation method on a model spec. */
export type EpSpecMethod = InferenceMethod<MpState, Config, EpSpecPath, ModelAnalysis>

/** The paths a variational message passing method logs: the engine's, and the posterior of a state as JSON. */
export type VmpSpecPath = VmpPath | 'mp/posterior'

const TITLE: Readonly<Record<MessageRule, string>> = {
  ep: 'expectation propagation',
  vmp: 'variational message passing',
  bp: 'belief propagation',
}

/** What the compiler of a spec method is told: whether the rows of a regression become one plated factor or a factor each. */
export interface MpSpecOptions {
  /** The name of the method (default `ep`, `vmp` or `bp`). */
  readonly name?: string
  /**
   * How the rows of a plate are compiled (see `toFactorGraph`): `vectorise` (default) makes an observed node that is one
   * regression on a weight vector a single plated factor, whose message is computed over all the rows in one call and whose
   * graph does not grow with the data; `expand` makes a factor and a variable per row (more messages per sweep, so
   * finer anytime steps, and a graph that draws row by row).
   */
  readonly plates?: 'vectorise' | 'expand'
  /**
   * Whether the scalar normal latents an observed node reads together (`a + b * x`) are compiled as one joint Gaussian vector
   * (see `toFactorGraph`). Default true for expectation propagation: as separate scalars the loop between them is loopy
   * Gaussian belief propagation, whose marginal spreads come out two to three times too small; as one vector the correlation is
   * kept. Default false for the others (a variational method over a joint vector is not mean-field).
   */
  readonly packScalars?: boolean
}

/** What the analysis alone says, before the graph is built: the features a message-passing method needs. */
function staticVerdict(a: ModelAnalysis, rule: MessageRule, compiles: boolean): Verdict | undefined {
  if (isProgramAnalysis(a)) return undefined
  const title = TITLE[rule]
  // A custom function of known values, or affine in one random vector, is read by the compiler: a model that compiled has nothing opaque left.
  if (a.opaque.length > 0 && !compiles)
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
  options: MpSpecOptions,
): InferenceMethod<MpState, Config, P, ModelAnalysis> {
  const name = options.name ?? rule
  const title = TITLE[rule]
  const prepared = prepare(model, options.plates ?? 'vectorise', options.packScalars ?? rule === 'ep')
  const base = messagePassingKnobs(rule)
  // Expectation propagation also takes the power of fractional EP, on every factor whose operator supports one (the likelihood
  // links, the softmax site, the shape and concentration sites, unary functions).
  const knobs =
    rule === 'ep'
      ? {
          ...base,
          config: dict({
            ...base.config.spaces,
            power: real(0.05, 2, {
              default: 1,
              label: 'power',
              description:
                'fractional EP: the power of every factor that supports one (1 is EP). Below 1 the spreads of variables that many factors share come out wider: where one Gaussian per variable leaves them too narrow (the class slopes of a softmax classifier), a power near 5 / rows restores the coverage of their intervals',
            }),
          }),
        }
      : base
  const powered = new Map<number, CompiledGraph | undefined>()
  /** The graph at the configuration's power (the prepared one at 1): the factors that take a power get it, the others stay at 1. */
  const graphAt = (config: Config): CompiledGraph | undefined => {
    const g = prepared?.graph
    const p = typeof config['power'] === 'number' ? config['power'] : 1
    if (g === undefined || p === 1 || rule !== 'ep') return g
    if (!powered.has(p))
      powered.set(
        p,
        withPowers(g, (f) => (takesPower(f.operator, f.params) ? p : 1)),
      )
    return powered.get(p)
  }
  const check = (a: ModelAnalysis): Verdict => {
    if (prepared === undefined)
      return no(
        `${title} runs on a model spec: a closure program has no factor graph (lifting a program to a spec is not provided)`,
        ['a model spec'],
        'write the model with defineModel',
      )
    const s = staticVerdict(a, rule, prepared.graph !== undefined)
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
  // The evidence of a state whose plated factor has an improper cavity, from the same model with a factor per row, by state.
  const rowEvidence = new WeakMap<MpState, { readonly value: number; readonly method: string } | null>()
  const evidenceByRows = (
    state: MpState,
    config: Config,
  ): { readonly value: number; readonly method: string } | null => {
    const known = rowEvidence.get(state)
    if (known !== undefined) return known
    let found: { readonly value: number; readonly method: string } | null = null
    const rows = prepare(model, 'expand', options.packScalars ?? rule === 'ep')
    if (rows?.graph !== undefined) {
      const run = runAlgorithm(
        epAlgorithm(rows.graph, {
          tol: tol(config),
          maxSweeps: 400,
          damping: typeof config['damping'] === 'number' ? config['damping'] : defaultsOf(rule).damping,
        }),
        { maxSteps: 402, seed: 1 },
      )
      const v = factorGraphPosterior(rows.compiled, rows.graph, run.state as MpState, {
        engine: name,
        rule,
      }).logEvidence
      if (v !== undefined && Number.isFinite(v.value) && run.status === 'converged')
        found = { value: v.value, method: 'ep-energy (rows expanded)' }
    }
    rowEvidence.set(state, found)
    return found
  }
  const posterior = (state: MpState, config: Config): Posterior => {
    if (prepared?.graph === undefined)
      return definePosterior({
        form: 'variational',
        engine: name,
        sites: [],
        status: 'failed',
        diagnostics: { reason: why() },
      })
    const graph = graphAt(config) ?? prepared.graph
    const verdict = mpVerdict(graph, state, {
      tol: tol(config),
      maxSweeps: Number.POSITIVE_INFINITY,
      schedule: (typeof config['schedule'] === 'string'
        ? config['schedule']
        : defaultsOf(rule).schedule) as ScheduleKind,
      damping: typeof config['damping'] === 'number' ? config['damping'] : defaultsOf(rule).damping,
      restarts: typeof config['restarts'] === 'number' ? config['restarts'] : defaultsOf(rule).restarts,
      projection: projectionOf(rule),
    })
    const status = verdict.status === 'converged' ? 'converged' : 'stopped'
    const reason = verdict.reason === undefined ? {} : { reason: verdict.reason }
    const own = factorGraphPosterior(prepared.compiled, graph, state, {
      engine: name,
      rule,
      status,
      ...reason,
    })
    // A converged run whose plated factor has an improper cavity has no EP energy of its own: the evidence of the same model with a
    // factor per row (each cavity proper) stands for it, and the diagnostics say so.
    if (
      rule !== 'ep' ||
      status !== 'converged' ||
      (options.plates ?? 'vectorise') === 'expand' ||
      own.logEvidence !== undefined ||
      !Array.isArray(own.diagnostics['evidenceUndefinedAt']) ||
      own.diagnostics['evidenceUndefinedAt'].length === 0
    )
      return own
    const evidence = evidenceByRows(state, config)
    return evidence === null
      ? own
      : factorGraphPosterior(prepared.compiled, graph, state, {
          engine: name,
          rule,
          status,
          ...reason,
          evidence,
        })
  }
  const tape = knobs.contract as Contract<MpState, string>
  const contract = (config: Config): Contract<MpState, P> =>
    defineContract({
      ...tape,
      'mp/posterior': derived<MpState>({
        cost: 'dear',
        description:
          'the posterior of the state as JSON: the marginal of every site, in the spec shapes, with the evidence or bound',
        compute: (s) => posteriorJson(posterior(s, config)),
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
      const g = graphAt(config)
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
 * @param options - A name for the method (default `ep`), how the plates are compiled (`plates`) and whether scalars sharing a predictor are packed (`packScalars`, on by default).
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
export function epSpecMethod(model: MethodModel, options: MpSpecOptions = {}): EpSpecMethod {
  return specMethod<EpSpecPath>('ep', model, options)
}

/**
 * Variational message passing on a bound spec: the expected-log message of every factor, so each variable's belief is the
 * mean-field distribution that minimises KL(q || p), one sweep per step, with the variational bound on the tape. It applies
 * where every operator of the graph has a variational rule: Gaussian and conjugate models, affine means (a regression on
 * scalar or vector weights, a hierarchy with means that are affine in latents; a defined value is the pushforward of
 * its inputs under the mean field), the probit link by the Albert-Chib augmentation and the logistic link by the Jaakkola
 * bound. `check` names the operators that have none and why (a hard comparison, `greater-than`).
 *
 * @param model - A bound spec.
 * @param options - A name for the method (default `vmp`) and how the plates are compiled (`plates`).
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
  options: MpSpecOptions = {},
): InferenceMethod<MpState, Config, VmpSpecPath, ModelAnalysis> {
  return specMethod<VmpSpecPath>('vmp', model, options)
}

/**
 * Belief propagation on a bound spec of finite discrete variables: Bernoulli nodes with tables of their parents' states
 * as factors, sum-product messages (exact on a tree, `schedule: 'tree'`), the Bethe estimate of the evidence. Continuous
 * latents are refused with the reason.
 *
 * @param model - A bound spec.
 * @param options - A name for the method (default `bp`) and how the plates are compiled (`plates`).
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
  options: MpSpecOptions = {},
): InferenceMethod<MpState, Config, EpSpecPath, ModelAnalysis> {
  return specMethod<EpSpecPath>('bp', model, options)
}
