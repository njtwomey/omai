/**
 * Expectation propagation from several starts, the runs weighted by their evidence, as an inference method on a model spec. One
 * EP run commits to one reading of a posterior that has several modes (which steps belong to a regime that is rarely visited,
 * which component is which), and reports the spread of that mode alone. Here the discrete variables of the graph (a regime
 * chain, a switch, a mixture's labels) are started from random messages, and from messages that rule out one category, EP runs
 * from each, the runs that reach the same fixed point are one mode, and the posterior is the mixture of the modes' beliefs
 * weighted by their EP evidence, which for a mode is its share of the posterior mass in the Gaussian (or gamma, beta)
 * approximation. A step is one start (the first is plain EP), so a snapshot is the mixture of the modes found so far.
 */

import { OmaiError } from '../../base/index.ts'
import {
  beliefOf,
  type CompiledGraph,
  epAlgorithm,
  type InitialMessage,
  type MpState,
  warmupOf,
} from '../../message-passing/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import { bool, choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  runAlgorithm,
  scalar,
} from '../../trace/index.ts'
import { factorGraphPosterior } from '../model/factor-graph/readback.ts'
import { factorGraphVerdict } from '../model/factor-graph/verdict.ts'
import type { Posterior } from '../posterior.ts'
import { definePosterior } from '../posterior.ts'
import { canonicalSites, moveAxis, orderingSite, permutations } from './_relabel.ts'
import { recordOf, type SiteRecord, siteMixturePosterior } from './_site-mixture.ts'
import {
  flag,
  isProgramAnalysis,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  type Prepared,
  pick,
  posteriorJson,
  prepare,
  yes,
} from './_support.ts'
import { atPower, epRunKnobs, epRunSettings, powerOf } from './ep-knobs.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths the multi-start method logs. */
export type EpMultiStartPath =
  | 'ep-multistart/starts'
  | 'ep-multistart/modes'
  | 'ep-multistart/logEvidence'
  | 'ep-multistart/snapshot'

/** One EP run from one start. */
export interface StartRun {
  /** Which start it was (0 is plain EP). */
  readonly start: number
  /** How it was started: plain EP, a category ruled out, random messages, or the discrete messages of run `of` relabelled by `perm`. */
  readonly origin: 'plain' | 'exclude' | 'random' | 'relabel'
  /** For a relabelled start: the run whose discrete messages were moved, and the permutation (entry `q` goes to `perm[q]`). */
  readonly of?: number
  readonly perm?: readonly number[]
  /**
   * The discrete messages of the fixed point, one array per start port, in the order of the graph's discrete ports: the message on
   * the port, and for the port of a prior the belief of its label (what the relabelled start of that reading sets there).
   */
  readonly discrete: readonly (readonly number[])[]
  /** The EP estimate of log p(data) at the fixed point it reached (negative infinity when it has none). */
  readonly logEvidence: number
  readonly converged: boolean
  readonly sweeps: number
  /** The largest change of a belief over the last sweep (the run's distance from a fixed point). */
  readonly change: number
  readonly sites: Readonly<Record<string, SiteRecord>>
}

/** The state of {@link epMultiStartMethod}: the runs made so far. */
export interface EpMultiStartState {
  readonly runs: readonly StartRun[]
}

/** The knobs of {@link epMultiStartMethod} (the budget and snapshot knobs are added to them). */
export const epMultiStartKnobs: DictSpace = dict({
  starts: int(1, 2000, {
    default: 24,
    label: 'starts',
    description: 'the EP runs: the first is plain EP, then one per excluded category (when asked), then random ones',
  }),
  jitter: real(0, 10, {
    default: 2,
    label: 'start spread',
    description: 'the sd of the random initial log-probabilities of each discrete message',
  }),
  exclusions: bool({
    default: true,
    label: 'exclusion starts',
    description:
      'start also from messages that rule out one category of every discrete variable (a regime nobody visits)',
  }),
  labels: choice(['keep', 'permute', 'sort'], {
    default: 'keep',
    label: 'labels',
    description:
      'keep: the components as the model names them; permute: also start from every relabelling of each reading found (the labellings the priors allow, each with its evidence); sort: the same, read with the components sorted by the first vector of parameters, so the relabellings of a reading are one reading',
  }),
  merge: real(0.001, 1, {
    log: true,
    default: 0.05,
    label: 'merge tolerance',
    description: 'two runs are one mode when every site mean differs by less than this share of the larger sd',
  }),
  // The knobs of each EP run (schedule, damping, improper, tol, maxSweeps, restarts, power): the shared declaration.
  ...epRunKnobs('ep-multistart').spaces,
})

/** Options of {@link epMultiStartMethod}. */
export interface EpMultiStartOptions {
  /** The name of the method (default `ep-multistart`). */
  readonly name?: string
  /** How the rows of a plate are compiled (see `toFactorGraph`; default `vectorise`). */
  readonly plates?: 'vectorise' | 'expand'
  /** Whether scalars sharing a predictor are packed (default true). */
  readonly packScalars?: boolean
}

/** A port that carries a discrete or Bernoulli message: its place, the size of the message, its edge and the family. */
interface StartPort {
  readonly factor: number
  readonly port: number
  readonly size: number
  readonly edge: number
  readonly kind: string
  /** The variable the port is on. */
  readonly variable: number
  /** Whether the factor is the prior of the variable (the only message a label with no neighbour in the graph but its row's likelihood is cavity-free on). */
  readonly prior: boolean
}

/**
 * The ports of the graph that carry a discrete or Bernoulli message: where a start is set. A prior's port is one: the cavity of the
 * likelihood of a row on its label is the prior's message and nothing else (the label has no other neighbour), so a start that
 * would not be set there changes nothing in a mixture whose labels are independent given the parameters.
 */
function discretePorts(g: CompiledGraph): StartPort[] {
  const out: StartPort[] = []
  g.factors.forEach((f, factor) => {
    f.ports.forEach((p, port) => {
      if (p.edge >= 0 && (p.family.kind === 'discrete' || p.family.kind === 'bernoulli'))
        out.push({
          factor,
          port,
          size: p.family.size,
          edge: p.edge,
          kind: p.family.kind,
          variable: g.edges[p.edge]?.variable ?? -1,
          prior: f.operator.name === 'prior',
        })
    })
  })
  return out
}

/** The start messages of start number `s`: nothing for plain EP, else random log-probabilities, or ones that rule out category `off`. */
function startMessages(
  ports: readonly StartPort[],
  key: Key,
  jitter: number,
  off: number | undefined,
): InitialMessage[] {
  const total = ports.reduce((a, p) => a + p.size, 0)
  const noise = toFlat(normal(key, [total]))
  let at = 0
  return ports.map(({ factor, port, size }) => {
    const eta = Array.from({ length: size }, (_, q) => {
      const draw = (noise[at + q] ?? 0) * jitter
      return off === undefined ? draw : (q === off ? -6 : 0) + 0.25 * draw
    })
    at += size
    return { factor, port, eta }
  })
}

/** Whether two fixed points agree: every site mean within `merge` of the larger sd. */
function sameSites(
  a: Readonly<Record<string, SiteRecord>>,
  b: Readonly<Record<string, SiteRecord>>,
  merge: number,
): boolean {
  return Object.entries(a).every(([site, ra]) => {
    const rb = b[site]
    if (rb === undefined) return false
    return ra.mean.every((m, i) => {
      const sd = Math.max(Math.sqrt(ra.variance[i] ?? 0), Math.sqrt(rb.variance[i] ?? 0), 1e-6)
      return Math.abs(m - (rb.mean[i] ?? 0)) <= merge * sd + 1e-9
    })
  })
}

/** The runs that count (those that converged when any did, with an evidence), best evidence first. */
function usableRuns(runs: readonly StartRun[]): StartRun[] {
  const usable = runs.filter((r) => Number.isFinite(r.logEvidence))
  const converged = usable.filter((r) => r.converged)
  return (converged.length > 0 ? converged : usable).slice().sort((a, b) => b.logEvidence - a.logEvidence)
}

/** The runs grouped into modes: the best-evidence run of each set whose site means agree within `merge` of the larger sd. */
function modesOf(runs: readonly StartRun[], merge: number): StartRun[] {
  const modes: StartRun[] = []
  for (const r of usableRuns(runs)) if (!modes.some((m) => sameSites(m.sites, r.sites, merge))) modes.push(r)
  return modes
}

/** One component of the posterior: what it believes of every site, and its log weight (its EP evidence). */
interface Mode {
  readonly sites: Readonly<Record<string, SiteRecord>>
  readonly logEvidence: number
}

/** The log of a sum of exponentials. */
function logSumExp(xs: readonly number[]): number {
  const top = Math.max(...xs)
  return top + Math.log(xs.reduce((a, v) => a + Math.exp(v - top), 0))
}

/**
 * The fixed points as the components of the posterior, each with its EP evidence as its weight. With relabelling each is put in
 * canonical form first (the components sorted by the mean of the first vector of parameters, `key`): the labellings of a reading
 * are different fixed points under asymmetric priors, each with its evidence, and read in one order they are one set of components.
 */
function posteriorModes(
  runs: readonly StartRun[],
  merge: number,
  canonical: { key: string; k: number } | undefined,
): Mode[] {
  const raw = modesOf(runs, merge)
  const modes = raw.map((r) => ({
    sites: canonical === undefined ? r.sites : canonicalSites(r.sites, canonical.key, canonical.k),
    logEvidence: r.logEvidence,
  }))
  return modes
}

/** The category counts the discrete messages of the graph have (two for a Bernoulli switch), the largest first. */
function categoryCounts(ports: readonly StartPort[]): number[] {
  const sizes = new Set(ports.filter((p) => p.kind === 'discrete').map((p) => p.size))
  if (ports.some((p) => p.kind === 'bernoulli')) sizes.add(2)
  return [...sizes].sort((a, b) => b - a)
}

/** The damped sweeps a start is held for before the run is let go (see {@link warmed}). */
const WARM_SWEEPS = 3

/**
 * The start messages carried through a few damped sweeps, as the messages the run starts from. A start set on the port of a label's
 * prior is a bias of the label before any row has been read, and the prior sends its own message again at its first update, so a run
 * with no damping would never see it; held by damping for `WARM_SWEEPS` sweeps the bias reaches the messages the rows send to the
 * parameters (which are not reset), and the run goes on from those at its own settings. The messages of every edge are returned.
 */
function warmed(g: CompiledGraph, initial: readonly InitialMessage[]): InitialMessage[] {
  const r = runAlgorithm(
    epAlgorithm(g, {
      schedule: 'serial',
      damping: 0.5,
      improper: 'damp',
      tol: 0,
      maxSweeps: WARM_SWEEPS,
      restarts: 0,
      initial,
    }),
    { maxSteps: WARM_SWEEPS, seed: 1 },
  )
  const messages = (r.state as MpState).messages
  return g.edges.map((e, i) => ({ factor: e.factor, port: e.port, eta: messages[i] ?? [] }))
}

/** The most components whose relabellings are all tried: 4 gives 24 labellings of every mode. */
const MAX_RELABELLED = 4

/** The discrete messages with the components relabelled: entry `q` of every message over `k` categories goes to `perm[q]`, a Bernoulli logit changes sign when the two are swapped. */
function relabelMessages(
  ports: readonly StartPort[],
  messages: readonly (readonly number[])[],
  perm: readonly number[],
  k: number,
): InitialMessage[] {
  return ports.map((p, i) => {
    const eta = messages[i] ?? []
    const moved =
      p.kind === 'discrete' && p.size === k
        ? moveAxis(eta, perm)
        : p.kind === 'bernoulli' && k === 2 && perm[0] === 1
          ? eta.map((v) => -v)
          : eta.slice()
    return { factor: p.factor, port: p.port, eta: moved }
  })
}

/**
 * Expectation propagation from many starts, the distinct fixed points weighted by their evidence: the answer for a model whose
 * posterior over discrete structure has several modes (a regime that is hardly visited, the labels of mixture components), where
 * one run reports one mode with a spread that is far too small. The discrete messages of the graph are started at random and,
 * when `exclusions` is on, at messages that rule out one category; runs that end at the same fixed point are one mode; the
 * posterior is the mixture of the modes' beliefs with weights proportional to their EP evidence, and the evidence of the model is
 * the log sum of the modes' (the modes found: a mode that no start reaches is not in it). With `labels: 'permute'` every reading found is
 * also started from each relabelling of its discrete messages, and with `labels: 'sort'` the posterior is read with the components
 * sorted by the mean of the first vector of parameters of their number (the relabellings of a reading become one reading, as wide as
 * the sampler's where every component is visited). A start is set on the port of each label's prior and held for a few damped
 * sweeps before the run goes on at its own settings: in a mixture or a clutter model whose labels are independent given the
 * parameters the likelihood of a row reads its label's cavity, which is the prior's message and nothing else, so a start set
 * anywhere else is divided out and every start would reach the same reading. The weights are the EP evidence of the fixed points,
 * which for well separated modes is their share of the mass (`diagnostics.modeWeights`: against the enumeration of a two-cluster
 * mixture with unequal priors, 0.559 against 0.559 exactly; against the quadrature of the clutter problem, the evidence within 0.006
 * nats and the mean and sd of the location within 0.03 sd); modes that overlap (the means of two clusters that are not apart) are
 * one fixed point, and its evidence is the single Gaussian's. The runs are weighted by their EP evidence, so a model with none at its fixed
 * point (a factor whose cavity is improper) gives no posterior.
 *
 * @param model - A bound spec.
 * @param options - The name and how plates are compiled.
 * @returns The method, with `check` (the model compiles and has a discrete latent) and `posterior`.
 * @concept expectation-propagation
 * @index estimator Expectation propagation from several starts, the fixed points weighted by their evidence.
 * @example
 * ```ts
 * import { epMultiStartMethod } from 'omai-engine/probprog/inference'
 * import { defineModel, ops } from 'omai-engine/probprog/model'
 * import { bernoulli, normal } from 'omai-engine/probprog/model/dists'
 * import { session } from 'omai-engine/trace'
 * // A coin picks which of two means a reading is drawn around: a reading between them has two explanations.
 * const spec = defineModel('m', (b) => {
 *   const side = b.latent('side', bernoulli({ probs: 0.5 }))
 *   b.observe('y', normal(ops.where(side, 3, -3), 2), b.data('y'))
 * })
 * const m = epMultiStartMethod(spec.bind({ y: 0.5 }))
 * const run = session(m, { steps: 10, mode: 'eager', config: { starts: 6 } })
 * [run.status, Number(m.posterior?.(run.state, run.config).diagnostics['modes']) >= 1] // => ['completed', true]
 * ```
 */
export function epMultiStartMethod(
  model: MethodModel,
  options: EpMultiStartOptions = {},
): InferenceMethod<EpMultiStartState, Config, EpMultiStartPath, ModelAnalysis> {
  const name = options.name ?? 'ep-multistart'
  const prepared: Prepared | undefined =
    typeof model === 'function' ? undefined : prepare(model, options.plates ?? 'vectorise', options.packScalars ?? true)
  const check = (a: ModelAnalysis): Verdict => {
    if (prepared === undefined || isProgramAnalysis(a))
      return no(
        `${name} runs on a model spec: a closure program has no factor graph`,
        ['a model spec'],
        'write the model with defineModel',
      )
    if (prepared.failure !== undefined)
      return no(`the factor graph does not compile: ${prepared.failure}`, ['a valid factor graph'])
    const v = factorGraphVerdict(prepared.compiled, prepared.graph, 'ep')
    if (!v.ok) return no(v.reason, v.missing, 'see toFactorGraph(model).issues')
    if (prepared.graph === undefined || discretePorts(prepared.graph).length === 0)
      return no(
        `${name} starts the discrete messages of the graph from different values, and the model has no discrete or Bernoulli variable`,
        ['a discrete latent'],
        'use ep',
      )
    return yes(
      'expectation propagation from several starts of the discrete messages, the distinct fixed points weighted by their evidence',
    )
  }

  /**
   * The components of the model and the site that sorts them: the largest category count (at most the number whose relabellings are
   * all tried) for which some continuous site is a vector of that length (a rate or a location per component), or undefined.
   */
  const componentsOf = (runs: readonly StartRun[]): { key: string; k: number } | undefined => {
    if (prepared?.graph === undefined) return undefined
    const first = runs.find((r) => r.converged) ?? runs[0]
    if (first === undefined) return undefined
    for (const k of categoryCounts(discretePorts(prepared.graph))) {
      if (k < 2 || k > MAX_RELABELLED) continue
      const key = orderingSite(first.sites, k)
      if (key !== undefined) return { key, k }
    }
    return undefined
  }
  const canonicalOf = (runs: readonly StartRun[], labels: string): { key: string; k: number } | undefined =>
    labels === 'sort' ? componentsOf(runs) : undefined

  const posterior = (state: EpMultiStartState, config: Config): Posterior => {
    if (prepared?.graph === undefined || state.runs.length === 0)
      return definePosterior({
        form: 'variational',
        engine: name,
        sites: [],
        status: 'failed',
        diagnostics: { reason: 'no run yet' },
      })
    const modes = posteriorModes(state.runs, num(config, 'merge'), canonicalOf(state.runs, pick(config, 'labels')))
    if (modes.length === 0)
      return definePosterior({
        form: 'variational',
        engine: name,
        sites: [],
        status: 'failed',
        diagnostics: { reason: 'no start reached a fixed point with an evidence' },
      })
    const top = Math.max(...modes.map((m) => m.logEvidence))
    const weights = modes.map((m) => Math.exp(m.logEvidence - top))
    const total = weights.reduce((a, b) => a + b, 0)
    const logEvidence = top + Math.log(total)
    return siteMixturePosterior({
      engine: name,
      status: state.runs.length >= num(config, 'starts') ? 'completed' : 'stopped',
      logEvidence: { value: logEvidence, method: 'ep-multistart' },
      diagnostics: {
        starts: state.runs.length,
        modes: modes.length,
        convergedStarts: state.runs.filter((r) => r.converged).length,
        modeEvidence: modes.map((m) => m.logEvidence),
        // Each mode's share of the posterior mass: its evidence over the evidence of the model (the weights of the mixture, summing to one).
        modeWeights: weights.map((w) => w / total),
      },
      weights,
      runs: modes.map((m) => m.sites),
    })
  }

  return defineInferenceMethod<EpMultiStartState, Config, EpMultiStartPath, ModelAnalysis>({
    name,
    family: 'message-passing',
    description:
      'expectation propagation from several starts of the discrete messages, the distinct fixed points mixed by their evidence',
    config: epMultiStartKnobs,
    check,
    posterior,
    defaults: { 'ep-multistart/starts': Log(), 'ep-multistart/modes': Log(), 'ep-multistart/logEvidence': Log() },
    contract: (config): Contract<EpMultiStartState, EpMultiStartPath> =>
      defineContract({
        'ep-multistart/starts': scalar({ description: 'EP runs made so far' }),
        'ep-multistart/modes': scalar({ description: 'the distinct fixed points found so far' }),
        'ep-multistart/logEvidence': scalar({ description: "the log sum of the modes' EP evidence" }),
        'ep-multistart/snapshot': derived<EpMultiStartState>({
          cost: 'dear',
          description: 'the evidence-weighted mixture of the modes found so far as JSON',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key): Algorithm<EpMultiStartState, EpMultiStartPath> => {
      const starts = num(config, 'starts')
      const jitter = num(config, 'jitter')
      const exclusions = flag(config, 'exclusions')
      const settings = epRunSettings(config)
      const maxSweeps = num(config, 'maxSweeps')
      const power = powerOf(config)
      const graph = (): CompiledGraph => {
        if (prepared?.graph === undefined)
          throw new OmaiError('value', `${name}: ${prepared?.failure ?? 'the model has no factor graph'}`)
        return atPower(prepared.graph, power)
      }
      const merge = num(config, 'merge')
      const labels = pick(config, 'labels')
      /** One EP run from the given start messages. */
      const run = (
        index: number,
        origin: StartRun['origin'],
        initial: readonly InitialMessage[],
        extra: { of?: number; perm?: readonly number[] } = {},
      ): StartRun => {
        const g = graph()
        const r = runAlgorithm(
          epAlgorithm(g, {
            ...settings,
            ...warmupOf(config),
            ...(initial.length === 0 ? {} : { initial: warmed(g, initial) }),
          }),
          {
            maxSteps: maxSweeps + 2,
            seed: 1,
          },
        )
        const final = r.state as MpState
        const post = factorGraphPosterior((prepared as Prepared).compiled, g, final, { engine: name })
        const sites: Record<string, SiteRecord> = {}
        for (const s of post.sites) sites[s] = recordOf(post.marginal(s))
        const ev = post.logEvidence?.value
        return {
          start: index,
          origin,
          ...extra,
          // The message of a port; for the port of a prior, the belief of the label (the prior's own message is the model's, the belief is the reading).
          discrete: discretePorts(g).map(
            (p) => (p.prior ? beliefOf(g, final.messages, p.variable) : final.messages[p.edge]) ?? [],
          ),
          logEvidence: ev === undefined || !Number.isFinite(ev) ? Number.NEGATIVE_INFINITY : ev,
          converged: r.status === 'converged',
          sweeps: r.steps,
          change: final.change,
          sites,
        }
      }
      /**
       * The next run. First the relabellings of every new reading (a converged fixed point whose canonical form no earlier one has,
       * that was not itself a relabelling): its discrete messages with the components permuted, every permutation once. Then the
       * base starts: plain EP, one per excluded category (when asked), then random ones.
       */
      const next = (runs: readonly StartRun[]): StartRun => {
        const g = graph()
        const ports = discretePorts(g)
        const canon = labels !== 'keep' ? componentsOf(runs) : undefined
        if (canon !== undefined) {
          const k = canon.k
          const perms = permutations(k)
          const seen: Readonly<Record<string, SiteRecord>>[] = []
          for (const [i, r] of runs.entries()) {
            if (!r.converged || !Number.isFinite(r.logEvidence)) continue
            const sites = canonicalSites(r.sites, canon.key, canon.k)
            if (seen.some((q) => sameSites(q, sites, merge))) continue
            seen.push(sites)
            if (r.origin === 'relabel') continue
            for (const perm of perms.slice(1)) {
              if (runs.some((q) => q.of === i && q.perm?.every((v, j) => v === perm[j]) === true)) continue
              return run(runs.length, 'relabel', relabelMessages(ports, r.discrete, perm, k), { of: i, perm })
            }
          }
        }
        const base = runs.filter((r) => r.origin !== 'relabel').length
        if (base === 0) return run(runs.length, 'plain', [])
        const categories = Math.max(0, ...ports.map((p) => p.size))
        // With exclusions, base starts 1..K rule out category 0..K-1; the rest are random.
        const off = exclusions && base <= categories ? base - 1 : undefined
        return run(
          runs.length,
          off === undefined ? 'random' : 'exclude',
          startMessages(ports, foldIn(key, base), jitter, off),
        )
      }
      return {
        name: 'probprog/ep-multistart-method',
        config: { starts, jitter, exclusions, labels, merge, ...settings, power },
        init: (): EpMultiStartState => {
          graph()
          return { runs: [] }
        },
        step: (s, ctx) => {
          const state: EpMultiStartState = { runs: [...s.runs, next(s.runs)] }
          const modes = posteriorModes(state.runs, merge, canonicalOf(state.runs, labels))
          ctx.log('ep-multistart/starts', state.runs.length)
          ctx.log('ep-multistart/modes', modes.length)
          if (modes.length > 0) ctx.log('ep-multistart/logEvidence', logSumExp(modes.map((m) => m.logEvidence)))
          return state
        },
        done: (s) => (s.runs.length >= starts ? 'completed' : undefined),
        serialise: (s) => JSON.parse(JSON.stringify(s)),
        deserialise: (j) => j as unknown as EpMultiStartState,
      }
    },
  })
}
