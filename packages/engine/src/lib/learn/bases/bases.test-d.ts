import { describe, expectTypeOf, it } from 'vitest'
import { qLearning } from '../../agents/index.ts'
import { lars } from '../../linear/index.ts'
import type { TrainState as NetState } from '../../nn/index.ts'
import type { TrainState } from '../../rl/index.ts'
import { dict } from '../../space/index.ts'
import { svc } from '../../svm/index.ts'
import { asTensor, full, type Tensor, type TensorLike } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  defineMethod,
  Log,
  type Method,
  type Session,
  scalar,
} from '../../trace/index.ts'
import { decisionTreeClassifier } from '../../trees/index.ts'
import type { LearnData } from '../estimator.ts'
import { type AgentMethod, type AgentPath, defineAgentMethod } from './agent.ts'
import {
  type IterativeEstimator,
  type IterativePath,
  type IterativeSession,
  type IterativeSessionOptions,
  iterative,
} from './iterative.ts'
import { type RacePath, race, type SweepPath, type SweepSpec, type SweepState, sweep, sweepMethod } from './sweep.ts'
import { defineTrainer, type TrainerEstimator, type TrainerPath, type TrainerSession } from './trainer.ts'

const data: LearnData = { x: [[0], [1]], y: [0, 1] }

interface MeanState {
  readonly i: number
  readonly m: number
}
type MeanParams = { readonly mean: number }
type MeanVerbs = { readonly predict: (x: TensorLike) => Tensor }
type MeanPath = 'extra/gap' | 'extra/seen'

const meanVerbs = (p: MeanParams): MeanVerbs => ({
  predict: (x) => full([asTensor(x).shape[0] ?? 0], p.mean),
})

// A typed estimator: the paths are inferred from the annotated algorithm and contract, no type arguments.
const typed = iterative({
  name: 'typed-mean',
  kind: 'regressor',
  defaults: {},
  algorithm: (): Algorithm<MeanState, MeanPath> => ({
    name: 'mean',
    config: {},
    init: () => ({ i: 0, m: 0 }),
    step: (s, ctx) => {
      ctx.log('extra/gap', 1)
      if (ctx.wants('extra/seen')) ctx.log('extra/seen', s.i)
      return { i: s.i + 1, m: s.m }
    },
  }),
  params: (s) => ({ mean: s.m }),
  verbs: meanVerbs,
  contract: (): Contract<MeanState, MeanPath> => defineContract({ 'extra/gap': scalar(), 'extra/seen': scalar() }),
  logs: { 'extra/gap': Log(), 'solver/anything': Log() },
})

// An untyped one, as every estimator written before typed paths: any string goes.
const untyped = iterative<{ readonly a: number }, MeanState, MeanParams, MeanVerbs>({
  name: 'untyped-mean',
  kind: 'regressor',
  defaults: { a: 1 },
  algorithm: () => ({
    name: 'mean',
    config: {},
    init: () => ({ i: 0, m: 0 }),
    step: (s, ctx) => {
      ctx.log('whatever/you/like', 1)
      return s
    },
  }),
  params: (s) => ({ mean: s.m }),
  verbs: meanVerbs,
  contract: () => ({ 'whatever/you/like': scalar() }),
  logs: { 'solver/nothing': Log() },
})

type PathsOfSession<T> = T extends Session<infer _S, infer _C, infer K> ? K : never
type PathsOfMethod<T> = T extends Method<infer _S, infer _C, infer K> ? K : never

describe('the iterative base', () => {
  it('a typed estimator carries its paths with the base`s, and does not widen them', () => {
    expectTypeOf(typed).toExtend<IterativeEstimator<object, MeanState, MeanParams, MeanVerbs, MeanPath>>()
    expectTypeOf<PathsOfMethod<ReturnType<typeof typed.method>>>().toEqualTypeOf<IterativePath<MeanPath>>()
    expectTypeOf<PathsOfSession<ReturnType<typeof typed.session>>>().toEqualTypeOf<IterativePath<MeanPath>>()
    // The particular's paths, the base's and the solver's channels; not a bare string.
    expectTypeOf<'extra/gap'>().toExtend<IterativePath<MeanPath>>()
    expectTypeOf<'model/params'>().toExtend<IterativePath<MeanPath>>()
    expectTypeOf<'fit/iteration/end'>().toExtend<IterativePath<MeanPath>>()
    expectTypeOf<'solver/loss'>().toExtend<IterativePath<MeanPath>>()
    expectTypeOf<string>().not.toExtend<IterativePath<MeanPath>>()
    expectTypeOf<'extra/gapp'>().not.toExtend<IterativePath<MeanPath>>()
    // `with` keeps the paths.
    expectTypeOf(typed.with({})).toEqualTypeOf<typeof typed>()
  })

  it('a typo in the logs or the probe of a typed estimator`s session is a type error', () => {
    typed.session(data, { logs: { 'extra/gap': Log(), 'model/*': Log(), 'solver/mean': Log(), '**/seen': Log() } })
    typed.session(data, { logs: 'all' })
    // @ts-expect-error not a path of the estimator
    typed.session(data, { logs: { 'extra/gapp': Log() } })
    const s = typed.session(data)
    s.probe(0, 'model/params')
    // @ts-expect-error not a path of the estimator
    s.probe(0, 'model/parms')
    // @ts-expect-error not a path of the estimator
    s.stepsOf('fit/iteration/stop')
    // A fork is a typed session too, and so is the fitted-at session.
    expectTypeOf(s.fork(0)).toExtend<IterativeSession<MeanState, MeanParams, MeanVerbs, IterativePath<MeanPath>>>()
    expectTypeOf(s.fittedAt(0).params).toEqualTypeOf<MeanParams>()
  })

  it('a typo in a typed estimator`s algorithm or contract is a type error', () => {
    iterative({
      name: 'typo',
      kind: 'regressor',
      defaults: {},
      algorithm: (): Algorithm<MeanState, MeanPath> => ({
        name: 'mean',
        config: {},
        init: () => ({ i: 0, m: 0 }),
        step: (s, ctx) => {
          // @ts-expect-error not a path of the algorithm
          ctx.log('extra/gapp', 1)
          // @ts-expect-error not a path of the algorithm
          ctx.emit('extra/gap/end')
          return s
        },
      }),
      params: (s) => ({ mean: s.m }),
      verbs: meanVerbs,
      contract: (): Contract<MeanState, MeanPath> => defineContract({ 'extra/gap': scalar(), 'extra/seen': scalar() }),
    })
    iterative({
      name: 'missing',
      kind: 'regressor',
      defaults: {},
      algorithm: (): Algorithm<MeanState, MeanPath> => ({
        name: 'mean',
        config: {},
        init: () => ({ i: 0, m: 0 }),
        step: (s) => s,
      }),
      params: (s) => ({ mean: s.m }),
      verbs: meanVerbs,
      // @ts-expect-error the contract lacks 'extra/seen'
      contract: (): Contract<MeanState, MeanPath> => defineContract({ 'extra/gap': scalar() }),
    })
    iterative({
      name: 'typo-logs',
      kind: 'regressor',
      defaults: {},
      algorithm: (): Algorithm<MeanState, MeanPath> => ({
        name: 'mean',
        config: {},
        init: () => ({ i: 0, m: 0 }),
        step: (s) => s,
      }),
      params: (s) => ({ mean: s.m }),
      verbs: meanVerbs,
      contract: (): Contract<MeanState, MeanPath> => defineContract({ 'extra/gap': scalar(), 'extra/seen': scalar() }),
      // @ts-expect-error not a path of the estimator
      logs: { 'extra/gapp': Log() },
    })
  })

  it('an untyped estimator accepts any string, as before', () => {
    expectTypeOf(untyped).toExtend<IterativeEstimator<{ readonly a: number }, MeanState, MeanParams, MeanVerbs>>()
    expectTypeOf<IterativePath>().toEqualTypeOf<string>()
    expectTypeOf<PathsOfMethod<ReturnType<typeof untyped.method>>>().toEqualTypeOf<string>()
    untyped.session(data, { logs: { 'whatever/you/like': Log(), 'also/this': Log() } })
    untyped.session(data).probe(0, 'any/probe')
    expectTypeOf<IterativeSessionOptions>().toEqualTypeOf<IterativeSessionOptions<string>>()
    // The base's paths include template literals (`solver/${string}`), yet a typed session still goes where the render
    // package wants a `Session<S, Config>`.
    const plain: Session<MeanState, Config> = typed.session(data)
    expectTypeOf(plain).not.toBeAny()
    const plainMethod: Method<MeanState, Config> = typed.method(data)
    expectTypeOf(plainMethod).not.toBeAny()
    // A typed estimator is accepted where an untyped one is wanted (a list of estimators, a pipeline step).
    expectTypeOf(typed).toExtend<IterativeEstimator<typeof typed.hyper, MeanState, MeanParams, MeanVerbs>>()
    const widened: IterativeEstimator<typeof typed.hyper, MeanState, MeanParams, MeanVerbs> = typed
    expectTypeOf(widened).not.toBeAny()
  })

  it('the migrated estimators of the library are typed', () => {
    svc.session(data, { logs: { 'smo/alphaChanged': Log(), 'solver/dual': Log(), 'model/support-vectors': Log() } })
    // @ts-expect-error not a path of the SVC
    svc.session(data, { logs: { 'smo/alpha': Log() } })
    lars().session(data, { logs: { 'solver/alpha': Log(), 'model/params': Log() } })
    // @ts-expect-error not a path of LARS
    lars().session(data, { logs: { 'lars/alpha': Log() } })
    decisionTreeClassifier().session(data, {
      logs: { 'tree/split': Log(), 'eval/accuracy': Log(), 'model/structure': Log() },
    })
    // @ts-expect-error not a path of the tree
    decisionTreeClassifier().session(data, { logs: { 'tree/splits': Log() } })
  })
})

describe('the trainer base', () => {
  type CriticPath = 'gan/critic'
  const net = null as unknown as TrainerEstimator<{ readonly lr: number }, CriticPath>

  it('carries the base`s paths with the particular`s', () => {
    expectTypeOf<'train/loss'>().toExtend<TrainerPath<CriticPath>>()
    expectTypeOf<'eval/accuracy'>().toExtend<TrainerPath<CriticPath>>()
    expectTypeOf<'layer/3/weights'>().toExtend<TrainerPath<CriticPath>>()
    expectTypeOf<'gan/critic'>().toExtend<TrainerPath<CriticPath>>()
    expectTypeOf<'train/lossy'>().not.toExtend<TrainerPath<CriticPath>>()
    expectTypeOf<'layer/x/weights'>().not.toExtend<TrainerPath<CriticPath>>()
    expectTypeOf<TrainerPath>().toEqualTypeOf<string>()
    expectTypeOf(net.session(data)).toEqualTypeOf<TrainerSession<TrainerPath<CriticPath>>>()
    net.session(data, {
      logs: { 'gan/critic': Log(), 'train/gradNorm': Log(), 'layer/1/activations': Log(), 'eval/f1': Log() },
    })
    // @ts-expect-error not a path of the trainer
    net.session(data, { logs: { 'train/gradnorm': Log() } })
  })

  it('infers the particular`s paths from its contract', () => {
    const t = defineTrainer({
      name: 'gan',
      kind: 'other',
      defaults: { lr: 0.1 },
      model: () => {
        throw new Error('unused')
      },
      objective: () => () => {
        throw new Error('unused')
      },
      optimiser: () => {
        throw new Error('unused')
      },
      settings: () => ({ learningRate: 0.1 }),
      contract: (): Contract<NetState, CriticPath> => defineContract({ 'gan/critic': scalar() }),
    })
    expectTypeOf(t).toExtend<TrainerEstimator<{ readonly lr: number }, CriticPath>>()
    t.session(data, { logs: { 'gan/critic': Log() } })
    // @ts-expect-error not a path of the trainer
    t.session(data, { logs: { 'gan/critic2': Log() } })
  })
})

describe('the agent base', () => {
  type Q = ReturnType<ReturnType<typeof qLearning>['init']>

  it('infers the particular`s paths from its contract, and checks the sessions', () => {
    const make = (): AgentMethod<object, Q, number, number, 'q/extra'> =>
      defineAgentMethod({
        name: 'q',
        defaults: {},
        environment: () => 'gridWorld',
        agent: (env) => qLearning(env, { alpha: 0.3, gamma: 0.95, epsilon: 0.2 }),
        schedule: () => ({ episodes: 2 }),
        contract: (): Contract<TrainState<Q, unknown, number>, 'q/extra'> => defineContract({ 'q/extra': scalar() }),
      })
    const q = make()
    q.session({ logs: { 'q/extra': Log(), 'episode/return': Log(), 'agent/epsilon': Log(), 'agent/*': Log() } })
    // @ts-expect-error not a path of the agent
    q.session({ logs: { 'episode/returns': Log() } })
    // @ts-expect-error not a path of the agent
    q.train({ logs: { 'q/extras': Log() } })
    expectTypeOf<'eval/return'>().toExtend<AgentPath<'q/extra'>>()
    expectTypeOf<AgentPath>().toEqualTypeOf<string>()
  })
})

describe('the sweep and the race', () => {
  const inner = defineMethod({
    name: 'inner',
    config: dict({}),
    make: () => ({
      name: 'inner',
      config: {},
      init: () => 0,
      step: (s: number, ctx) => {
        ctx.log('inner/loss', s)
        return s + 1
      },
    }),
    contract: defineContract({ 'inner/loss': scalar(), 'model/curve': scalar() }),
  })

  it('checks the metrics, the final paths and the logs of a sweep against the inner method`s paths', () => {
    sweep({ method: inner, metrics: ['inner/loss'], final: ['model/curve'], logs: { 'inner/*': Log() } })
    // @ts-expect-error not a path of the inner method
    sweep({ method: inner, metrics: ['inner/lss'] })
    // @ts-expect-error not a path of the inner method
    sweep({ method: inner, metrics: ['inner/loss'], final: ['model/curv'] })
    // @ts-expect-error not a path of the inner method
    sweep({ method: inner, metrics: ['inner/loss'], logs: { 'inner/lss': Log() } })
    const untypedInner: Method<number, Config> = null as unknown as Method<number, Config>
    sweep({ method: untypedInner, metrics: ['any/thing'] })
    expectTypeOf(sweep({ method: inner, metrics: ['inner/loss'] })).toEqualTypeOf<Algorithm<SweepState, SweepPath>>()
    expectTypeOf<SweepSpec<number, Config>['metrics']>().toEqualTypeOf<readonly string[]>()
  })

  it('the sweep method`s paths are the sweep`s; the race mounts the inner paths under the variants', () => {
    const m = sweepMethod({ method: inner, metrics: ['inner/loss'] })
    expectTypeOf(m).toExtend<Method<SweepState, Config, SweepPath>>()
    expectTypeOf<'trial/best'>().toExtend<SweepPath>()
    expectTypeOf<'trial/anything'>().toExtend<SweepPath>()
    expectTypeOf<'config/inner/loss/mean'>().toExtend<SweepPath>()
    expectTypeOf<'trial'>().not.toExtend<SweepPath>()
    const r = race(inner, { a: {}, b: {} })
    expectTypeOf<PathsOfMethod<typeof r>>().toEqualTypeOf<RacePath<'inner/loss' | 'model/curve'>>()
    expectTypeOf<RacePath<'a' | 'b'>>().toEqualTypeOf<`race/${string}/a` | `race/${string}/b`>()
    session3(r)
  })
})

/** Sessions of a race check their subscriptions against the mounted paths. */
function session3(r: ReturnType<typeof race<number, Config, 'inner/loss' | 'model/curve'>>): void {
  const opts = (logs: IterativeSessionOptions<RacePath<'inner/loss' | 'model/curve'>>['logs']): unknown => logs
  opts({ 'race/a/inner/loss': Log() })
  // @ts-expect-error not a path of the race
  opts({ 'race/a/inner/lss': Log() })
  // @ts-expect-error not a path of the race
  opts({ 'inner/loss': Log() })
  expectTypeOf(r.name).toEqualTypeOf<string>()
}
