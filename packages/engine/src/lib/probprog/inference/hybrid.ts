/**
 * The explicit `hybrid` plan: a user-written assignment of blocks of latent sites to kernels, run as one sampler with a
 * composite state (NIMBLE's sampler assignment, Turing.jl's `Gibbs`, NumPyro's `HMCGibbs` and `MixedHMC`, Bean Machine's blocks).
 * A block is a set of sites and the kernel that moves them given every other site: a Hamiltonian kernel (`hmc`, `nuts`, `mala`,
 * `barker`, `rwm`, window-adapted on the block's own conditional), slice sampling, an exact Gibbs update from the conjugate and
 * discrete conditionals of the Gibbs engine (the way a discrete site is moved in `HMCGibbs`), or a kernel the user
 * writes (`custom`). A continuous block may also sum finite discrete sites out of its density (`marginalise`, as NUTS does and
 * `MixedHMC` approximates), and those sites are drawn from their conditional when a draw is kept. Nothing here chooses the plan:
 * the user names every block, and the method says what is wrong with a plan that leaves a site unassigned. One step is one sweep of
 * every block (in the plan's order or a random one) for every chain, so a run cancels within one sweep, resumes from its
 * serialised state and publishes anytime snapshots; the tape has the acceptance and the step size of every block.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../../random/index.ts'
import { type HamiltonianParameters, type LogDensity, type MCMCKernel, sliceSampler } from '../../sampling/index.ts'
import { choice, type DictSpace, dict, int } from '../../space/index.ts'
import { fromStorage, primalOf, type Tensor, toFlat, type Value } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { findInitialParams, initToUniform, potentialEnergy, type UnconstrainedModel } from '../density.ts'
import { gibbsAlgorithm } from '../engines/gibbs.ts'
import { inferDiscrete } from '../engines/nuts.ts'
import { condition, seed, traceOf } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { assess } from '../trace-ops.ts'
import type { Warmer } from './_mcmc-driver.ts'
import { hamiltonianFamilies, windowWarmer } from './_mcmc-families.ts'
import {
  guard,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  posteriorJson,
  programOf,
  restorable,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** A Hamiltonian, Langevin, Barker or random-walk kernel for a block, window-adapted on the block's conditional density. */
export interface HamiltonianBlockKernel {
  readonly kind: 'hmc' | 'nuts' | 'mala' | 'barker' | 'rwm'
  /** Leapfrog steps of `hmc` (default 10). */
  readonly numSteps?: number
  /** The tree depth cap of `nuts` (default 10). */
  readonly maxTreeDepth?: number
  /** The acceptance rate dual averaging aims for (default the kernel's usual: 0.8, 0.574 for MALA, 0.4 Barker, 0.234 random walk). */
  readonly targetAcceptance?: number
  /** A dense mass matrix (not for Barker and the random walk). */
  readonly dense?: boolean
}

/** Slice sampling for a block, coordinate by coordinate, with a fixed bracket width. */
export interface SliceBlockKernel {
  readonly kind: 'slice'
  readonly width?: number
}

/**
 * The exact update of the Gibbs engine for the block's sites: a discrete site from its conditional by enumeration, a site with a
 * conjugate conditional from it, a collapsible Dirichlet or beta site from the Dirichlet-multinomial, and a slice step on the
 * unconstrained scale for the rest.
 */
export interface GibbsBlockKernel {
  readonly kind: 'gibbs'
  readonly collapse?: boolean
  readonly sliceWidth?: number
}

/** A kernel the user writes: from the block's target (log density and gradient on its unconstrained coordinates) to a Markov kernel. */
export interface CustomBlockKernel {
  readonly kind: 'custom'
  readonly name: string
  readonly kernelFor: (target: LogDensity) => MCMCKernel
}

/** The kernels a block can have. */
export type BlockKernel = HamiltonianBlockKernel | SliceBlockKernel | GibbsBlockKernel | CustomBlockKernel

/**
 * The constructors of block kernels, so a plan reads as `{ sites: ['mu', 'tau'], kernel: blockKernels.nuts() }`.
 *
 * @concept hybrid-sampling
 * @index table The kernels a block of a hybrid plan can have: hmc, nuts, mala, barker, rwm, slice, gibbs, custom.
 * @example
 * ```ts
 * import { blockKernels } from 'omai-engine/probprog/inference'
 * [blockKernels.hmc({ numSteps: 5 }).kind, blockKernels.gibbs().kind] // => ['hmc', 'gibbs']
 * ```
 */
export const blockKernels: {
  readonly hmc: (options?: Omit<HamiltonianBlockKernel, 'kind' | 'maxTreeDepth'>) => HamiltonianBlockKernel
  readonly nuts: (options?: Omit<HamiltonianBlockKernel, 'kind' | 'numSteps'>) => HamiltonianBlockKernel
  readonly mala: (options?: Pick<HamiltonianBlockKernel, 'targetAcceptance' | 'dense'>) => HamiltonianBlockKernel
  readonly barker: (options?: Pick<HamiltonianBlockKernel, 'targetAcceptance'>) => HamiltonianBlockKernel
  readonly rwm: (options?: Pick<HamiltonianBlockKernel, 'targetAcceptance'>) => HamiltonianBlockKernel
  readonly slice: (options?: Omit<SliceBlockKernel, 'kind'>) => SliceBlockKernel
  readonly gibbs: (options?: Omit<GibbsBlockKernel, 'kind'>) => GibbsBlockKernel
  readonly custom: (name: string, kernelFor: (target: LogDensity) => MCMCKernel) => CustomBlockKernel
} = {
  hmc: (o = {}) => ({ kind: 'hmc', ...o }),
  nuts: (o = {}) => ({ kind: 'nuts', ...o }),
  mala: (o = {}) => ({ kind: 'mala', ...o }),
  barker: (o = {}) => ({ kind: 'barker', ...o }),
  rwm: (o = {}) => ({ kind: 'rwm', ...o }),
  slice: (o = {}) => ({ kind: 'slice', ...o }),
  gibbs: (o = {}) => ({ kind: 'gibbs', ...o }),
  custom: (name, kernelFor) => ({ kind: 'custom', name, kernelFor }),
}

/** A block of the plan: the sites it moves and the kernel that moves them given every other site. */
export interface HybridBlock {
  /** The latent sites this block moves (each latent site is in exactly one block, or is summed out by one). */
  readonly sites: readonly string[]
  readonly kernel: BlockKernel
  /**
   * Finite discrete latent sites summed out of this block's density by enumeration (not moved by it): the block's target is the
   * marginal over them, as NUTS does for the whole model. They must be moved by a `gibbs` block, or they are drawn from their
   * conditional given the rest when a draw is kept.
   */
  readonly marginalise?: readonly string[]
}

/** The plan of a hybrid sampler: the blocks, in the order a sweep visits them. */
export interface HybridPlan {
  readonly blocks: readonly HybridBlock[]
}

/** The paths the hybrid method logs. */
export type HybridPath =
  | 'init/tries'
  | 'hybrid/stage'
  | 'hybrid/logJoint'
  | 'hybrid/accept'
  | 'hybrid/stepSize'
  | 'hybrid/order'
  | 'hybrid/kept'
  | 'hybrid/snapshot'

/** One block's state in one chain: its warm-up while it adapts, its tuned parameters afterwards, and what its last move did. */
export interface HybridBlockSlot {
  readonly warm: unknown | null
  readonly tuned: HamiltonianParameters | null
  readonly accept: number
  readonly stepSize: number
}

/** One chain of a hybrid sampler: the value of every latent site, the blocks' states and the draws kept. */
export interface HybridChainState {
  readonly values: Readonly<Record<string, Value>>
  readonly blocks: readonly HybridBlockSlot[]
  readonly logJoint: number
  /** The draws kept so far: the flat values of every latent site in the order of the sites. */
  readonly kept: readonly Float64Array[]
}

/** The state of a hybrid sampler. */
export interface HybridRunState {
  readonly stage: 'init' | 'warmup' | 'sample' | 'done'
  readonly iteration: number
  readonly chains: readonly HybridChainState[]
  readonly failure: string | null
}

/** The knobs of {@link hybridMethod} (the budget and snapshot knobs are added to them). */
export const hybridKnobs: DictSpace = dict({
  numChains: int(1, 10_000, { default: 2, label: 'chains' }),
  numWarmup: int(0, 1_000_000, { default: 300, label: 'warm-up sweeps' }),
  numDraws: int(1, 1_000_000, { default: 500, label: 'draws per chain' }),
  thin: int(1, 1000, { default: 1, label: 'thinning' }),
  order: choice(['systematic', 'random'], {
    default: 'systematic',
    label: 'block order',
    description: "the plan's order every sweep, or a random order",
  }),
})

const product = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

/** What the plan says about a program: every latent site is moved by a block, the blocks are well formed. */
function planVerdict(
  plan: HybridPlan,
  a: { readonly latent: readonly string[]; readonly continuous: readonly string[] },
): Verdict {
  if (plan.blocks.length === 0) return no('the plan has no blocks', ['blocks'], 'give every latent site to a block')
  const latent = new Set(a.latent)
  const moved = new Map<string, number>()
  for (const [i, b] of plan.blocks.entries()) {
    if (b.sites.length === 0) return no(`block ${i} has no sites`, ['sites'])
    for (const s of b.sites) {
      if (!latent.has(s))
        return no(
          `block ${i} names '${s}', which is not a latent site`,
          ['latent sites'],
          `the latent sites are ${a.latent.join(', ')}`,
        )
      if (moved.has(s)) return no(`'${s}' is in block ${moved.get(s)} and block ${i}`, ['disjoint blocks'])
      moved.set(s, i)
    }
    for (const s of b.marginalise ?? [])
      if (!latent.has(s)) return no(`block ${i} marginalises '${s}', which is not a latent site`, ['latent sites'])
    const continuous = a.continuous
    if (b.kernel.kind !== 'gibbs' && b.sites.some((s) => !continuous.includes(s)))
      return no(
        `block ${i} moves a discrete site with a ${b.kernel.kind} kernel`,
        ['continuous sites'],
        'move discrete sites with a gibbs block',
      )
  }
  // A site some block sums out is covered: it is drawn from its conditional when a draw is kept.
  const summed = new Set(plan.blocks.flatMap((b) => b.marginalise ?? []))
  const missing = a.latent.filter((s) => !moved.has(s) && !summed.has(s))
  if (missing.length > 0)
    return no(
      `no block moves ${missing.join(', ')}`,
      ['a block for every latent site'],
      'add a block (a gibbs block moves discrete and conjugate sites)',
    )
  return yes(`${plan.blocks.length} blocks cover the ${a.latent.length} latent sites`)
}

/** The statistics a block logs: acceptance and step size. */
const blockStats = (
  info: { readonly acceptanceRate: number; readonly stepSize?: number | undefined } | undefined,
): [number, number] => [info?.acceptanceRate ?? 1, info?.stepSize ?? Number.NaN]

/**
 * The hybrid sampler as a method: the user's plan of blocks, each a set of latent sites with the kernel that moves them given
 * every other site, run as one sampler with a composite state. A sweep moves every block once, in the plan's order (or a random
 * one); a block's kernel targets the conditional of its sites given the current values of the others (a Hamiltonian block on its
 * unconstrained coordinates with the bijectors of the sites, the discrete sites it `marginalise`s summed out by enumeration; a
 * `gibbs` block by the exact conditionals of the Gibbs engine), and every adaptive block tunes its own step size and mass matrix in
 * warm-up with Stan's windows. Every sweep of every chain is one step, so a run cancels within one sweep, resumes exactly from its
 * serialised state and publishes the draws so far as a posterior of every latent site (discrete ones included), with R-hat and
 * effective sample sizes. The tape adds, per chain and block, the acceptance and the step size, and the log joint of the state.
 * The plan must give every latent site to exactly one block (`check` says which is missing); a conjugate block combined with a NUTS
 * block samples the same posterior as NUTS on the whole model.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param plan - The blocks.
 * @param options - The method's name (default `hybrid`).
 * @returns The method.
 * @throws OmaiError with code `value` (when a session is made) for a plan that does not cover the latent sites.
 * @concept hybrid-sampling
 * @index estimator The user's plan of blocks, each with its own kernel (NUTS, HMC, slice, Gibbs, custom), run as one sampler.
 * @example
 * ```ts
 * import { blockKernels, hybridMethod } from 'omai-engine/probprog/inference'
 * import { gamma, normal } from 'omai-engine/dists'
 * import { observe, plate, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => {
 *   const mu = sample('mu', normal(0, 2))
 *   const sigma = sample('sigma', gamma(2, 2))
 *   plate('n', 3, () => observe('y', normal(mu, sigma), [0.2, 1.1, -0.4]))
 * }
 * const m = hybridMethod(model, { blocks: [{ sites: ['mu'], kernel: blockKernels.gibbs() }, { sites: ['sigma'], kernel: blockKernels.nuts() }] })
 * const s = session(m, { config: { numChains: 1, numWarmup: 20, numDraws: 10 }, steps: 200, mode: 'eager' })
 * [s.status, m.posterior?.(s.state, s.config).sites] // => ['converged', ['mu', 'sigma']]
 * ```
 */
export function hybridMethod(
  model: MethodModel,
  plan: HybridPlan,
  options: { readonly name?: string } = {},
): InferenceMethod<HybridRunState, Config, HybridPath, ModelAnalysis> {
  const name = options.name ?? 'hybrid'
  const program: Model = programOf(model)
  const check = (a: ModelAnalysis): Verdict => planVerdict(plan, { latent: a.latent, continuous: a.continuous })
  const ensure = guard(name, model, check)

  // The latent sites and the shapes of their values, from one run of the program (only shapes are taken from it).
  const proto = traceOf(seed(program, makeKey(0)), undefined)
  const siteNames: string[] = []
  const shapes: Record<string, readonly number[]> = {}
  const protoValues: Record<string, Value> = {}
  for (const s of proto.sites.values())
    if (s.type === 'sample' && !s.isObserved && !s.isFactor) {
      siteNames.push(s.name)
      shapes[s.name] = primalOf(s.value).shape
      protoValues[s.name] = primalOf(s.value)
    }
  const sizes = siteNames.map((n) => product(shapes[n] ?? []))
  const offsets = sizes.map((_, i) => sizes.slice(0, i).reduce((a, b) => a + b, 0))
  const total = sizes.reduce((a, b) => a + b, 0)

  const rowOf = (values: Readonly<Record<string, Value>>): Float64Array => {
    const row = new Float64Array(total)
    siteNames.forEach((n, i) => {
      row.set(toFlat(primalOf(values[n] as Value)), offsets[i] ?? 0)
    })
    return row
  }

  // ---- the blocks at run time: a target with a cell for the conditioned values, or a Gibbs updater ----
  interface Runtime {
    readonly block: HybridBlock
    readonly index: number
    readonly cell: { rest: Readonly<Record<string, Value>> }
    readonly um: UnconstrainedModel | undefined
    readonly gibbs:
      | (Algorithm<{ values: Readonly<Record<string, Value>>; logJoint: number }> & { readonly prime: () => void })
      | undefined
    readonly kernelFor: ((p: HamiltonianParameters) => MCMCKernel) | undefined
    readonly tuned0: HamiltonianParameters | undefined
    readonly adapt: { readonly dense: boolean; readonly targetAcceptance: number } | undefined
  }
  const runtimes = (): Runtime[] =>
    plan.blocks.map((block, index): Runtime => {
      const sites = new Set(block.sites)
      const marginal = new Set(block.marginalise ?? [])
      const cell: { rest: Readonly<Record<string, Value>> } = {
        rest: Object.fromEntries(Object.entries(protoValues).filter(([n]) => !sites.has(n) && !marginal.has(n))),
      }
      if (block.kernel.kind === 'gibbs') {
        const inner = gibbsAlgorithm(program, undefined, {
          sites: block.sites,
          collapse: block.kernel.collapse ?? false,
          ...(block.kernel.sliceWidth === undefined ? {} : { sliceWidth: block.kernel.sliceWidth }),
        })
        return {
          block,
          index,
          cell,
          um: undefined,
          gibbs: restorable(inner, makeKey(0)),
          kernelFor: undefined,
          tuned0: undefined,
          adapt: undefined,
        }
      }
      const blockProgram: Model = (d) =>
        condition(program, (msg) =>
          msg.type === 'sample' && !sites.has(msg.name) && !marginal.has(msg.name) ? cell.rest[msg.name] : undefined,
        )(d)
      const um = potentialEnergy(blockProgram, undefined)
      const layout = um.layout.map((l) => l.name)
      if (layout.length !== block.sites.length || block.sites.some((s) => !layout.includes(s)))
        throw new OmaiError(
          'value',
          `hybrid: block ${index} moves ${block.sites.join(', ')} but its density has the coordinates of ${layout.join(', ') || 'no site'}`,
        )
      const k = block.kernel
      const dim = um.dim
      let kernelFor: (p: HamiltonianParameters) => MCMCKernel
      let adapt: Runtime['adapt']
      if (k.kind === 'slice') {
        const width = k.width ?? 1
        kernelFor = () => sliceSampler(um.logDensity, { width })
      } else if (k.kind === 'custom') {
        kernelFor = () => k.kernelFor(um.logDensity)
      } else {
        const family = hamiltonianFamilies[k.kind]
        if (family === undefined) throw new OmaiError('value', `hybrid: no kernel '${k.kind}'`)
        kernelFor = (p) =>
          family.make(um.logDensity, p, dim, { numSteps: k.numSteps ?? 10, maxTreeDepth: k.maxTreeDepth ?? 10 })
        adapt = { dense: k.dense ?? false, targetAcceptance: k.targetAcceptance ?? family.accept }
      }
      const identity =
        adapt?.dense === true
          ? Float64Array.from({ length: dim * dim }, (_, i) => (i % (dim + 1) === 0 ? 1 : 0))
          : new Float64Array(dim).fill(1)
      return {
        block,
        index,
        cell,
        um,
        gibbs: undefined,
        kernelFor,
        tuned0: { stepSize: 1, inverseMassMatrix: identity },
        adapt,
      }
    })

  const posterior = (state: HybridRunState): Posterior => {
    if (state.failure !== null)
      return definePosterior({
        form: 'samples',
        engine: name,
        sites: [],
        status: 'failed',
        draws: {},
        diagnostics: { reason: state.failure },
      })
    const n = Math.min(...state.chains.map((c) => c.kept.length))
    if (!Number.isFinite(n) || n < 1)
      return definePosterior({
        form: 'samples',
        engine: name,
        sites: [],
        status: 'failed',
        draws: {},
        diagnostics: { reason: 'no draws yet' },
      })
    const numChains = state.chains.length
    const draws: Record<string, Tensor> = {}
    siteNames.forEach((site, j) => {
      const size = sizes[j] ?? 1
      const data = new Float64Array(numChains * n * size)
      state.chains.forEach((c, ci) => {
        for (let d = 0; d < n; d++)
          data.set((c.kept[d] as Float64Array).subarray(offsets[j] ?? 0, (offsets[j] ?? 0) + size), (ci * n + d) * size)
      })
      draws[site] = fromStorage(data, [numChains * n, ...(shapes[site] ?? [])], 'float64')
    })
    const done = state.stage === 'done'
    return definePosterior({
      form: 'samples',
      engine: name,
      sites: siteNames,
      draws,
      chainShape: [numChains, n],
      status: done ? 'completed' : 'stopped',
      diagnostics: {
        partial: !done,
        iterations: state.iteration,
        draws: n,
        blocks: plan.blocks.map((b) => `${b.kernel.kind}: ${b.sites.join(', ')}`),
      },
    })
  }

  return defineInferenceMethod<HybridRunState, Config, HybridPath, ModelAnalysis>({
    name,
    family: 'composite',
    description: 'the plan of blocks the user wrote, each with its own kernel, one sweep of every block per step',
    config: hybridKnobs,
    check,
    posterior: (s) => posterior(s),
    defaults: {
      'hybrid/stage': Log(),
      'hybrid/logJoint': Log(),
      'hybrid/accept': Log(),
      'hybrid/stepSize': Log(),
      'hybrid/kept': Log(),
    },
    contract: (): Contract<HybridRunState, HybridPath> =>
      defineContract({
        'init/tries': tensor({
          role: 'output',
          shape: ['chains'],
          description: "tries the search for each chain's start made",
        }),
        'hybrid/stage': scalar({ description: '1 while adapting, 2 while drawing' }),
        'hybrid/logJoint': tensor({
          role: 'output',
          shape: ['chains'],
          description: 'the log joint density of every chain now',
        }),
        'hybrid/accept': tensor({
          role: 'output',
          shape: ['chains', 'blocks'],
          description: "the acceptance probability of each block's last move in each chain (1 for an exact update)",
        }),
        'hybrid/stepSize': tensor({
          role: 'output',
          shape: ['chains', 'blocks'],
          description: "the step size of each block's last move in each chain (NaN where the kernel has none)",
        }),
        'hybrid/order': tensor({
          role: 'output',
          shape: ['blocks'],
          description: 'the order the blocks were visited in this sweep',
        }),
        'hybrid/kept': scalar({ description: 'draws kept per chain so far' }),
        'hybrid/snapshot': derived<HybridRunState>({
          cost: 'dear',
          description: 'the posterior so far as JSON: means, sds, R-hat and ESS of every latent site',
          compute: (s) => posteriorJson(posterior(s)),
        }),
      }) as unknown as Contract<HybridRunState, HybridPath>,
    make: (config) => {
      ensure()
      const c = {
        numChains: num(config, 'numChains'),
        numWarmup: num(config, 'numWarmup'),
        numDraws: num(config, 'numDraws'),
        thin: num(config, 'thin'),
        random: config['order'] === 'random',
      }
      const blocks = runtimes()
      const recover = siteNames.filter(
        (n) =>
          plan.blocks.some((b) => b.marginalise?.includes(n) === true) && !plan.blocks.some((b) => b.sites.includes(n)),
      )
      const warmers: (Warmer<HamiltonianParameters> | undefined)[] = blocks.map((r) =>
        r.um === undefined || r.adapt === undefined || r.kernelFor === undefined
          ? undefined
          : windowWarmer(r.um.dim, r.kernelFor, {
              numWarmup: c.numWarmup,
              dense: r.adapt.dense,
              targetAcceptance: r.adapt.targetAcceptance,
            }),
      )
      const inner = (ctx: StepContext<HybridPath>, t: number, key: Key): StepContext => ({
        t,
        key,
        run: ctx.run,
        log: () => undefined,
        wants: () => false,
        emit: () => undefined,
        taps: ctx.taps,
      })
      const stepBlock = (
        r: Runtime,
        slot: HybridBlockSlot,
        values: Readonly<Record<string, Value>>,
        ctx: StepContext<HybridPath>,
        iteration: number,
        key: Key,
        warming: boolean,
      ): { readonly values: Readonly<Record<string, Value>>; readonly slot: HybridBlockSlot } => {
        if (r.gibbs !== undefined) {
          r.gibbs.prime()
          const next = r.gibbs.step({ values, logJoint: 0 }, inner(ctx, iteration, key) as never)
          return { values: { ...values, ...next.values }, slot: { ...slot, accept: 1, stepSize: Number.NaN } }
        }
        const um = r.um as UnconstrainedModel
        const sites = new Set(r.block.sites)
        r.cell.rest = Object.fromEntries(Object.entries(values).filter(([n]) => !sites.has(n)))
        const picked = Object.fromEntries(r.block.sites.map((s) => [s, values[s] as Value]))
        const u0 = um.unconstrain(picked)
        let position: Float64Array
        let accept: number
        let stepSize: number
        let next: HybridBlockSlot
        const warmer = warmers[r.index]
        if (warming && warmer !== undefined && slot.warm !== null) {
          const w = warmer.step(warmer.restart?.(slot.warm, u0) ?? slot.warm, inner(ctx, iteration, key))
          position = warmer.chain(w).position
          ;[accept, stepSize] = blockStats(warmer.info(w))
          next = { ...slot, warm: w, accept, stepSize }
        } else {
          const tuned = slot.tuned ?? (r.tuned0 as HamiltonianParameters)
          const kernel = (r.kernelFor as (p: HamiltonianParameters) => MCMCKernel)(tuned)
          const out = kernel.step(key, kernel.init(u0))
          position = out.state.position
          ;[accept, stepSize] = blockStats(out.info)
          next = { ...slot, accept, stepSize }
        }
        const constrained = um.constrain(position)
        const updated: Record<string, Value> = { ...values }
        for (const s of r.block.sites) updated[s] = primalOf(constrained[s] as Value)
        return { values: updated, slot: next }
      }
      const orderOf = (key: Key): number[] => {
        const order = blocks.map((_, i) => i)
        if (!c.random) return order
        // Fisher-Yates from the sweep's key.
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.min(i, Math.floor((uniform(foldIn(foldIn(key, 77), i), [1]).data[0] ?? 0) * (i + 1)))
          ;[order[i], order[j]] = [order[j] as number, order[i] as number]
        }
        return order
      }
      const recoverDiscrete = (values: Readonly<Record<string, Value>>, key: Key): Readonly<Record<string, Value>> => {
        if (recover.length === 0) return values
        const given = Object.fromEntries(Object.entries(values).filter(([n]) => !recover.includes(n)))
        const drawn = inferDiscrete(program, undefined, given, key)
        const out: Record<string, Value> = { ...values }
        for (const n of recover) if (drawn[n] !== undefined) out[n] = primalOf(drawn[n] as Tensor)
        return out
      }
      const sweep = (
        chain: HybridChainState,
        ctx: StepContext<HybridPath>,
        iteration: number,
        ci: number,
        warming: boolean,
        keep: boolean,
      ): HybridChainState => {
        const key = foldIn(ctx.key, ci)
        let values = chain.values
        const slots = [...chain.blocks]
        for (const bi of orderOf(key)) {
          const r = blocks[bi] as Runtime
          const out = stepBlock(
            r,
            slots[bi] as HybridBlockSlot,
            values,
            ctx,
            iteration,
            foldIn(foldIn(key, 1), bi),
            warming,
          )
          values = out.values
          slots[bi] = out.slot
        }
        const kept = keep ? [...chain.kept, rowOf(recoverDiscrete(values, foldIn(key, 2)))] : chain.kept
        return { values, blocks: slots, logJoint: assess(program, undefined, values), kept }
      }
      const logTape = (
        ctx: StepContext<HybridPath>,
        stage: number,
        chains: readonly HybridChainState[],
        order: readonly number[],
      ): void => {
        ctx.log('hybrid/stage', stage)
        ctx.log(
          'hybrid/logJoint',
          chains.map((ch) => ch.logJoint),
        )
        ctx.log(
          'hybrid/accept',
          chains.flatMap((ch) => ch.blocks.map((b) => b.accept)),
        )
        ctx.log(
          'hybrid/stepSize',
          chains.flatMap((ch) => ch.blocks.map((b) => b.stepSize)),
        )
        ctx.log('hybrid/order', [...order])
        ctx.log('hybrid/kept', chains[0]?.kept.length ?? 0)
      }
      const algorithm: Algorithm<HybridRunState, HybridPath> = {
        name: `probprog/${name}`,
        config: { numChains: c.numChains, numWarmup: c.numWarmup, numDraws: c.numDraws },
        init: () => {
          ensure()
          return { stage: 'init', iteration: 0, chains: [], failure: null }
        },
        step: (s, ctx) => {
          if (s.stage === 'init') {
            const full = potentialEnergy(program, undefined)
            const chains: HybridChainState[] = []
            const tries: number[] = []
            for (let i = 0; i < c.numChains; i++) {
              const start = findInitialParams(full, program, undefined, {
                key: foldIn(foldIn(ctx.key, 1), i),
                strategy: initToUniform(),
              })
              tries.push(start.tries)
              if (!start.valid) {
                ctx.log('init/tries', tries)
                return { ...s, stage: 'done', failure: 'no starting point with a finite log density and gradient' }
              }
              // The continuous sites at the start; the discrete sites from their conditional given them (a valid state, not a posterior draw).
              const continuous = Object.fromEntries(
                siteNames.flatMap((n) =>
                  start.constrained[n] === undefined ? [] : [[n, primalOf(start.constrained[n] as Value)] as const],
                ),
              )
              const t = traceOf(seed(condition(program, continuous), foldIn(foldIn(ctx.key, 2), i)), undefined)
              const values: Record<string, Value> = {}
              for (const n of siteNames) {
                const site = t.sites.get(n)
                values[n] = primalOf((continuous[n] ?? (site?.type === 'sample' ? site.value : undefined)) as Value)
              }
              const slots: HybridBlockSlot[] = blocks.map((r, bi) => {
                const warmer = warmers[bi]
                if (warmer === undefined || c.numWarmup === 0 || r.um === undefined)
                  return { warm: null, tuned: r.tuned0 ?? null, accept: Number.NaN, stepSize: Number.NaN }
                // The first position of the block's chain is set when its target is built at the first sweep.
                const sites = new Set(r.block.sites)
                r.cell.rest = Object.fromEntries(Object.entries(values).filter(([n]) => !sites.has(n)))
                const u0 = r.um.unconstrain(Object.fromEntries(r.block.sites.map((n) => [n, values[n] as Value])))
                return {
                  warm: warmer.init(u0, foldIn(foldIn(ctx.key, 3), bi * 1000 + i)),
                  tuned: null,
                  accept: Number.NaN,
                  stepSize: Number.NaN,
                }
              })
              chains.push({ values, blocks: slots, logJoint: assess(program, undefined, values), kept: [] })
            }
            ctx.log('init/tries', tries)
            return { ...s, chains, stage: c.numWarmup > 0 ? 'warmup' : 'sample', iteration: 0 }
          }
          if (s.stage === 'warmup') {
            const i = s.iteration
            const chains = s.chains.map((ch, ci) => sweep(ch, ctx, i, ci, true, false))
            logTape(ctx, 1, chains, orderOf(foldIn(ctx.key, 0)))
            if (i + 1 < c.numWarmup) return { ...s, chains, iteration: i + 1 }
            const sampling = chains.map(
              (ch): HybridChainState => ({
                ...ch,
                blocks: ch.blocks.map((b, bi) => {
                  const warmer = warmers[bi]
                  return warmer === undefined || b.warm === null
                    ? b
                    : { ...b, warm: null, tuned: warmer.result(b.warm) }
                }),
              }),
            )
            return { ...s, chains: sampling, stage: 'sample', iteration: 0 }
          }
          const j = s.iteration
          const keep = (j + 1) % c.thin === 0
          const chains = s.chains.map((ch, ci) => sweep(ch, ctx, j, ci, false, keep))
          logTape(ctx, 2, chains, orderOf(foldIn(ctx.key, 0)))
          const full = (chains[0]?.kept.length ?? 0) >= c.numDraws
          return { ...s, chains, iteration: j + 1, stage: full ? 'done' : 'sample' }
        },
        done: (s) => (s.stage !== 'done' ? undefined : s.failure === null ? 'converged' : 'failed'),
        serialise: (s) => ({
          stage: s.stage,
          iteration: s.iteration,
          failure: s.failure,
          chains: s.chains.map((ch) => ({
            values: stateToJson(ch.values),
            logJoint: stateToJson(ch.logJoint),
            blocks: ch.blocks.map((b, bi) => ({
              warm: b.warm === null ? null : (warmers[bi]?.toJson(b.warm) ?? null),
              tuned: b.tuned === null ? null : stateToJson(b.tuned),
              accept: stateToJson(b.accept),
              stepSize: stateToJson(b.stepSize),
            })),
            kept: stateToJson(
              ch.kept.length === 0 ? new Float64Array(0) : Float64Array.from(ch.kept.flatMap((r) => Array.from(r))),
            ),
            rows: ch.kept.length,
          })),
        }),
        deserialise: (j) => {
          const o = j as unknown as {
            stage: HybridRunState['stage']
            iteration: number
            failure: string | null
            chains: Record<string, Json>[]
          }
          return {
            stage: o.stage,
            iteration: o.iteration,
            failure: o.failure,
            chains: o.chains.map((ch): HybridChainState => {
              const flat = stateFromJson(ch['kept'] as Json) as Float64Array
              const rows = ch['rows'] as number
              return {
                values: stateFromJson(ch['values'] as Json) as Record<string, Value>,
                logJoint: stateFromJson(ch['logJoint'] as Json) as number,
                blocks: (ch['blocks'] as Record<string, Json>[]).map(
                  (b, bi): HybridBlockSlot => ({
                    warm: b['warm'] === null ? null : (warmers[bi]?.fromJson(b['warm'] as Json) ?? null),
                    tuned: b['tuned'] === null ? null : (stateFromJson(b['tuned'] as Json) as HamiltonianParameters),
                    accept: stateFromJson(b['accept'] as Json) as number,
                    stepSize: stateFromJson(b['stepSize'] as Json) as number,
                  }),
                ),
                kept: Array.from({ length: rows }, (_, r) => flat.slice(r * total, (r + 1) * total)),
              }
            }),
          }
        },
      }
      return algorithm
    },
  })
}
