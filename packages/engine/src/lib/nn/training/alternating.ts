/**
 * Training several modules against one another: `alternatingAlgorithm` steps any list of players (a module, its
 * optimiser and its loss, which may read every player's parameters) through a schedule of rounds, with each player's
 * loss and gradient norm on the tape; `ganAlgorithm` is the generator and discriminator (or critic) pair on top of it,
 * with the discriminator stepped `nCritic` times per generator step, the objectives of `losses/generative` (minimax,
 * non-saturating, Wasserstein, hinge, least squares), the gradient penalty of WGAN-GP, and the discriminator's accuracy,
 * the Wasserstein estimate and generated samples as tracks; `modeCoverage` measures how many modes of a known mixture
 * the samples reach. One round is one session step. The single-module trainer of the learn base is
 * `training/fit.ts`; this is its counterpart for alternating objectives.
 */

import { valueAndGrad } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import { discriminatorLoss, type GanObjective, generatorLoss, gradientPenalty } from '../../losses/index.ts'
import { applyUpdates, type GradientTransformation, globalNorm } from '../../optim/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import {
  add,
  asTensor,
  cast,
  greater,
  item,
  less,
  mean,
  mul,
  primalOf,
  reshape,
  sub,
  type Tensor,
  treeLeaves,
  treeMap,
  type Value,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../../trace/index.ts'
import { concreteTree, type Module, type Params } from '../module/module.ts'

/** What a player's loss may use besides its own parameters: applying any player's module with its buffers and a key. */
export interface PlayerScope {
  /** Apply the module of a player to an input with the given parameters (training mode, the step's key, the buffers). */
  apply(name: string, params: Params, x: Value): Value
  /** A key for this player's step (interpolation points, noise). */
  readonly key: Key
}

/** One participant: a module, the optimiser that trains it, and the loss it minimises. */
export interface Player<B> {
  readonly name: string
  readonly module: Module<Value, Value>
  readonly optimizer: GradientTransformation
  /**
   * The loss of this player's parameters `own` given the current parameters of every player (`all`, with `own` in its
   * place, so a loss differentiates through its own parameters only) and the batch of this substep.
   */
  loss(own: Params, all: Readonly<Record<string, Params>>, batch: B, scope: PlayerScope): Value
}

/** One entry of the schedule: a player and how many consecutive updates it takes in the round. */
export interface ScheduleEntry {
  readonly player: string
  readonly times?: number
}

const hasTrainable = (mask: Params): boolean => treeLeaves(mask).some((m) => item(m) !== 0)

/** The state of an alternating run. */
export interface AlternatingState {
  readonly params: Readonly<Record<string, Params>>
  readonly buffers: Readonly<Record<string, Params>>
  readonly optStates: Readonly<Record<string, unknown>>
  /** Rounds completed. */
  readonly round: number
  /** Updates taken by each player. */
  readonly updates: Readonly<Record<string, number>>
  /** Each player's loss and gradient norm at its last update. */
  readonly losses: Readonly<Record<string, number>>
  readonly gradNorms: Readonly<Record<string, number>>
}

/** The paths every alternating run logs for players `N`. */
export type AlternatingPath<N extends string> = `player/${N}/loss` | `player/${N}/gradNorm` | 'schedule/round'

/** Options of {@link alternatingAlgorithm}. */
export interface AlternatingOptions<B, N extends string, K extends string> {
  readonly name?: string
  readonly players: readonly (Player<B> & { readonly name: N })[]
  /** What each round does, in order (default: every player once, in the order given). */
  readonly schedule?: readonly ScheduleEntry[]
  /** A fresh batch for a substep, from a key. */
  readonly batch: (key: Key) => B
  readonly rounds: number
  /** Extra tracks computed from the state after each round. */
  readonly extra?: (state: AlternatingState, ctx: StepContext<AlternatingPath<N> | K>) => void
}

/**
 * Train a list of modules in a schedule (the multi-module trainer): each round runs the schedule's entries in order, a
 * player taking `times` updates, each on a fresh batch and with a fresh key, on its own loss, which sees every player's
 * current parameters. Frozen parameters (`requiresGrad(false)`) stay put. The state serialises. A GAN is two players
 * with the discriminator stepped several times per generator step; actor-critic, a teacher and a student, or an
 * adversary and a defender fit the same shape.
 *
 * @param options - Players, schedule, batch source, rounds and extra tracks.
 * @returns The algorithm; one step is one round.
 * @concept alternating-optimisation
 * @example
 * ```ts
 * import { alternatingAlgorithm, Linear } from 'omai-engine/nn'
 * import { sgd } from 'omai-engine/optim'
 * import { sum, square } from 'omai-engine/tensor'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const a = new Linear(1, 1)
 * const alg = alternatingAlgorithm({
 *   players: [{ name: 'a', module: a, optimizer: sgd({ learningRate: 0.1 }), loss: (own, _all, _b, scope) => sum(square(scope.apply('a', own, [[1]] as never))) }],
 *   batch: () => 0,
 *   rounds: 3,
 * })
 * runAlgorithm(alg, { maxSteps: 10 }).state.round // => 3
 * ```
 */
export function alternatingAlgorithm<B, N extends string, K extends string = never>(
  options: AlternatingOptions<B, N, K>,
): Algorithm<AlternatingState, AlternatingPath<N> | K> {
  const players = new Map<string, Player<B>>(options.players.map((p) => [p.name, p]))
  const schedule: readonly ScheduleEntry[] = options.schedule ?? options.players.map((p) => ({ player: p.name }))
  for (const e of schedule)
    if (!players.has(e.player)) throw new OmaiError('value', `alternatingAlgorithm: no player ${e.player}`)
  const modules = new Map<string, Module<Value, Value>>(options.players.map((p) => [p.name, p.module]))
  const names = [...players.keys()]
  const masks = new Map<string, Params>(options.players.map((p) => [p.name, p.module.trainableMask()]))
  // A player with nothing to train takes no updates (a frozen generator while a critic is fitted to it).
  const frozen = new Set<string>(
    options.players.filter((p) => !hasTrainable(masks.get(p.name) as Params)).map((p) => p.name),
  )

  const update = (s: AlternatingState, player: Player<B>, key: Key): AlternatingState => {
    if (frozen.has(player.name)) return s
    const batch = options.batch(foldIn(key, 0))
    const captured: Record<string, Params> = {}
    const scope: PlayerScope = {
      key: foldIn(key, 1),
      apply: (name, params, x) => {
        const m = modules.get(name)
        if (m === undefined) throw new OmaiError('value', `no player ${name}`)
        return m.apply(params, x, {
          train: true,
          key: foldIn(key, 2 + names.indexOf(name)),
          buffers: captured[name] ?? s.buffers[name] ?? {},
          onBuffers: (b) => {
            captured[name] = b
          },
        })
      },
    }
    const own = s.params[player.name] as Params
    const { value, grad } = valueAndGrad((p: Params) =>
      player.loss(p, { ...s.params, [player.name]: p }, batch, scope),
    )(own)
    const r = player.optimizer.update(grad, s.optStates[player.name], own)
    const masked = treeMap((u, m) => mul(u, m), r.updates as Params, masks.get(player.name) as Params)
    return {
      ...s,
      params: { ...s.params, [player.name]: applyUpdates(own, masked) as Params },
      buffers: { ...s.buffers, ...Object.fromEntries(Object.entries(captured).map(([k, v]) => [k, concreteTree(v)])) },
      optStates: { ...s.optStates, [player.name]: r.state },
      updates: { ...s.updates, [player.name]: (s.updates[player.name] ?? 0) + 1 },
      losses: { ...s.losses, [player.name]: item(primalOf(value)) },
      gradNorms: { ...s.gradNorms, [player.name]: item(globalNorm(grad)) },
    }
  }

  return {
    name: options.name ?? 'alternating',
    config: {
      players: names,
      schedule: schedule.map((e) => ({ player: e.player, times: e.times ?? 1 })),
      rounds: options.rounds,
    },
    init: () => ({
      params: Object.fromEntries(options.players.map((p) => [p.name, p.module.params])),
      buffers: Object.fromEntries(options.players.map((p) => [p.name, p.module.buffers])),
      optStates: Object.fromEntries(options.players.map((p) => [p.name, p.optimizer.init(p.module.params)])),
      round: 0,
      updates: Object.fromEntries(names.map((n) => [n, 0])),
      losses: Object.fromEntries(names.map((n) => [n, Number.NaN])),
      gradNorms: Object.fromEntries(names.map((n) => [n, Number.NaN])),
    }),
    step: (s, ctx) => {
      let state = s
      let sub = 0
      for (const entry of schedule) {
        const player = players.get(entry.player) as Player<B>
        for (let t = 0; t < (entry.times ?? 1); t++) state = update(state, player, foldIn(ctx.key, sub++))
      }
      state = { ...state, round: s.round + 1 }
      for (const n of names) {
        if (frozen.has(n)) continue
        ctx.log(`player/${n}/loss` as never, state.losses[n] as number)
        ctx.log(`player/${n}/gradNorm` as never, state.gradNorms[n] as number)
      }
      ctx.log('schedule/round', state.round)
      options.extra?.(state, ctx)
      return state
    },
    done: (s) => (s.round >= options.rounds ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as AlternatingState,
    observe: (s) => ({
      metrics: Object.fromEntries(
        names.flatMap((n) => (Number.isNaN(s.losses[n]) ? [] : [[`${n}.loss`, s.losses[n] as number]])),
      ),
    }),
  }
}

/** The players of a GAN. */
export type GanPlayer = 'generator' | 'discriminator'

/** The paths a GAN run logs besides the per-player ones. */
export type GanPath = 'gan/discriminatorAccuracy' | 'gan/wasserstein' | 'gan/penalty' | 'gan/samples'

/** The paths of {@link ganAlgorithm}. */
export type GanRunPath = AlternatingPath<GanPlayer> | GanPath

/** What a GAN run logs. */
export const ganContract: Contract<AlternatingState, GanRunPath> = defineContract({
  'player/generator/loss': scalar({ description: 'the generator’s loss at its last update' }),
  'player/generator/gradNorm': scalar({ description: 'the norm of the generator’s gradient' }),
  'player/discriminator/loss': scalar({
    description: 'the discriminator’s (critic’s) loss at its last update, penalty included',
  }),
  'player/discriminator/gradNorm': scalar({ description: 'the norm of the discriminator’s gradient' }),
  'schedule/round': scalar({ description: 'rounds completed' }),
  'gan/discriminatorAccuracy': scalar({
    description: 'the discriminator’s accuracy on a fresh real and generated batch',
  }),
  'gan/wasserstein': scalar({
    description: 'the critic’s estimate of the Wasserstein distance, E D(real) - E D(fake)',
  }),
  'gan/penalty': scalar({ description: 'the gradient penalty (weight included) on a fresh batch' }),
  'gan/samples': tensorSource({ role: 'state', description: 'generated samples from a fixed noise batch' }),
})

/** Options of {@link ganAlgorithm}. */
export interface GanOptions {
  readonly generator: Module<Value, Value>
  readonly discriminator: Module<Value, Value>
  readonly generatorOptimizer: GradientTransformation
  readonly discriminatorOptimizer: GradientTransformation
  /** The noise dimension. */
  readonly noiseDim: number
  /** Real samples `[n, ...]` from a key. */
  readonly sampleReal: (n: number, key: Key) => Value
  readonly batchSize: number
  readonly rounds: number
  /** Default Wasserstein. */
  readonly objective?: GanObjective
  /** Discriminator updates per generator update (default 5 for Wasserstein, else 1). */
  readonly nCritic?: number
  /** Weight of the WGAN-GP gradient penalty (default 10 for Wasserstein, else 0). */
  readonly gradientPenalty?: number
  /** Samples used for the tracks (default 256). */
  readonly evalBatch?: number
  /** Generated samples kept per tracked round (default 256). */
  readonly sampleCount?: number
  /** Compute the tracks (accuracy, Wasserstein estimate, penalty, samples) every this many rounds (default 1). */
  readonly trackEvery?: number
}

interface GanBatch {
  readonly real: Value
  readonly z: Value
}

/**
 * A GAN as an alternating run: the discriminator (a critic for the Wasserstein objective) takes `nCritic` updates on
 * fresh real and generated batches, with the gradient penalty at interpolates (WGAN-GP) when asked, then the generator
 * takes one on the discriminator's current opinion. Logs both losses and gradient norms, the discriminator's accuracy,
 * the critic's Wasserstein estimate E D(real) - E D(fake), the penalty, and generated samples from fixed noise. Wrap the
 * discriminator's layers in `SpectralNorm` for the spectrally normalised alternative to the penalty.
 *
 * @param options - Modules, optimisers, data, objective and schedule knobs.
 * @returns The algorithm; one step is one round.
 * @concept generative-adversarial-networks
 * @example
 * ```ts
 * import { ganAlgorithm, mlp } from 'omai-engine/nn'
 * import { adam } from 'omai-engine/optim'
 * import { key, normal } from 'omai-engine/random'
 * ganAlgorithm({
 *   generator: mlp([2, 8, 2], { key: key(0) }), discriminator: mlp([2, 8, 1], { key: key(1) }),
 *   generatorOptimizer: adam({ learningRate: 1e-3 }), discriminatorOptimizer: adam({ learningRate: 1e-3 }),
 *   noiseDim: 2, sampleReal: (n, k) => normal(k, [n, 2]), batchSize: 16, rounds: 1,
 * }).name // => 'gan'
 * ```
 */
export function ganAlgorithm(options: GanOptions): Algorithm<AlternatingState, GanRunPath> {
  const objective = options.objective ?? 'wasserstein'
  const wasserstein = objective === 'wasserstein'
  const nCritic = options.nCritic ?? (wasserstein ? 5 : 1)
  const penalty = options.gradientPenalty ?? (wasserstein ? 10 : 0)
  const evalN = options.evalBatch ?? 256
  const sampleN = options.sampleCount ?? 256
  const trackEvery = Math.max(1, options.trackEvery ?? 1)
  const critic =
    (scope: PlayerScope, params: Params) =>
    (x: Value): Value =>
      reshape(scope.apply('discriminator', params, x), [-1])
  const discriminator: Player<GanBatch> & { name: 'discriminator' } = {
    name: 'discriminator',
    module: options.discriminator,
    optimizer: options.discriminatorOptimizer,
    loss: (own, all, batch, scope) => {
      const fake = scope.apply('generator', all['generator'] as Params, batch.z)
      const d = critic(scope, own)
      let l = discriminatorLoss(d(batch.real), d(fake), { objective })
      if (penalty > 0) l = add(l, gradientPenalty(d, batch.real, fake, { weight: penalty, key: scope.key }))
      return l
    },
  }
  const generator: Player<GanBatch> & { name: 'generator' } = {
    name: 'generator',
    module: options.generator,
    optimizer: options.generatorOptimizer,
    loss: (own, all, batch, scope) => {
      const fake = scope.apply('generator', own, batch.z)
      return generatorLoss(critic(scope, all['discriminator'] as Params)(fake), { objective })
    },
  }
  return alternatingAlgorithm<GanBatch, GanPlayer, GanPath>({
    name: 'gan',
    players: [discriminator, generator],
    schedule: [{ player: 'discriminator', times: nCritic }, { player: 'generator' }],
    batch: (key) => ({
      real: options.sampleReal(options.batchSize, foldIn(key, 0)),
      z: normal(foldIn(key, 1), [options.batchSize, options.noiseDim]),
    }),
    rounds: options.rounds,
    extra: (s, ctx) => {
      if (s.round % trackEvery !== 0) return
      const evalKey = foldIn(ctx.key, 1 << 20)
      const real = options.sampleReal(evalN, foldIn(evalKey, 0))
      const g = options.generator
      const d = options.discriminator
      const fake = g.apply(s.params['generator'] as Params, normal(foldIn(evalKey, 1), [evalN, options.noiseDim]), {
        train: false,
        buffers: s.buffers['generator'] ?? {},
      })
      const score = (x: Value): Value =>
        reshape(
          d.apply(s.params['discriminator'] as Params, x, { train: false, buffers: s.buffers['discriminator'] ?? {} }),
          [-1],
        )
      const dr = score(real)
      const df = score(fake)
      const midpoint = wasserstein ? (item(mean(dr)) + item(mean(df))) / 2 : 0
      ctx.log(
        'gan/discriminatorAccuracy',
        0.5 * (item(mean(cast(greater(dr, midpoint), 'float64'))) + item(mean(cast(less(df, midpoint), 'float64')))),
      )
      ctx.log('gan/wasserstein', item(sub(mean(dr), mean(df))))
      if (penalty > 0 && ctx.wants('gan/penalty')) {
        ctx.log('gan/penalty', item(gradientPenalty(score, real, fake, { weight: penalty, key: foldIn(evalKey, 2) })))
      }
      if (ctx.wants('gan/samples')) {
        ctx.log(
          'gan/samples',
          g.apply(s.params['generator'] as Params, normal(foldIn(ctx.key, 0xfeed), [sampleN, options.noiseDim]), {
            train: false,
            buffers: s.buffers['generator'] ?? {},
          }) as Tensor,
        )
      }
    },
  })
}

/** How a generated sample set covers a known mixture. */
export interface ModeCoverage {
  /** Modes with at least `minShare` of the samples assigned to them. */
  readonly covered: number
  /** Samples assigned to each mode (within `radius` of it). */
  readonly counts: readonly number[]
  /** The share of samples within `radius` of some mode. */
  readonly highQuality: number
}

/**
 * Mode coverage of generated samples against the means of a known mixture: each sample within `radius` of its nearest
 * mean is assigned to it; a mode is covered when at least `minShare` of all samples are. High-quality samples are
 * those near some mode. The standard check for mode collapse on a mixture of Gaussians.
 *
 * @param samples - Generated samples `[n, d]`.
 * @param means - The mixture's means.
 * @param radius - The distance within which a sample counts as at a mode (for example three standard deviations).
 * @param minShare - The share of samples that makes a mode covered (default 0.5 / number of modes).
 * @returns Modes covered, per-mode counts and the high-quality share.
 * @concept generative-adversarial-networks
 * @example
 * ```ts
 * import { modeCoverage } from 'omai-engine/nn'
 * modeCoverage([[0, 0], [0.1, 0], [5, 5]], [[0, 0], [5, 5]], 0.5).covered // => 2
 * ```
 */
export function modeCoverage(
  samples: Tensor | readonly (readonly number[])[],
  means: readonly (readonly number[])[],
  radius: number,
  minShare: number = 0.5 / means.length,
): ModeCoverage {
  const t = asTensor(samples as never) as Tensor
  const n = t.shape[0] ?? 0
  const d = t.shape[1] ?? 0
  const counts = means.map(() => 0)
  let near = 0
  for (let i = 0; i < n; i++) {
    let best = -1
    let bestDistance = Number.POSITIVE_INFINITY
    means.forEach((m, j) => {
      let s = 0
      for (let k = 0; k < d; k++) s += ((t.data[i * d + k] as number) - (m[k] as number)) ** 2
      if (s < bestDistance) {
        bestDistance = s
        best = j
      }
    })
    if (Math.sqrt(bestDistance) <= radius) {
      counts[best] = (counts[best] ?? 0) + 1
      near++
    }
  }
  return {
    covered: counts.filter((c) => c / Math.max(1, n) >= minShare).length,
    counts,
    highQuality: near / Math.max(1, n),
  }
}
