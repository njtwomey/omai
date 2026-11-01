/**
 * The interface every sequence mixer shares, so that the layers of a sequence model (a state-space layer, Mamba,
 * linear attention, RetNet, RWKV, softmax attention) are interchangeable in one stack: a mixer maps `[B, L, D]` to
 * `[B, L, D]` mixing information along the sequence, causally, and carries a state (a tree of tensors: the
 * recurrent state, a convolution window, a key-value cache) from one call to the next. Called on a whole sequence
 * without a state it is the parallel form used in training; called on one token with the state of the previous call
 * it is the recurrent form used in generation, and the two give the same outputs.
 */

import { foldIn, type Key } from '../../random/index.ts'
import { asTensor, isValue, type TensorLike, type Value } from '../../tensor/index.ts'
import { Composite } from '../module/containers.ts'
import type { ApplyContext, Module, Params } from '../module/module.ts'

/** The state a mixer carries between calls: a tree of tensors (empty for a mixer with no memory). */
export type MixerState = Params

/** The input of a mixer: the sequence `[B, L, D]` and, optionally, the state left by the earlier calls. */
export interface MixerInput {
  readonly x: TensorLike
  readonly state?: MixerState
}

/** The output of a mixer: the sequence `[B, L, D]` and the state after its last position. */
export interface MixerOutput {
  readonly output: Value
  readonly state: MixerState
}

/** The eigenvalues of a mixer's state transition: of `x' = A x` (continuous time) or of `x_t = A x_{t-1}` (discrete). */
export interface TransitionSpectrum {
  readonly domain: 'continuous' | 'discrete'
  readonly re: readonly number[]
  readonly im: readonly number[]
}

/**
 * The key a mixer draws its own parameters from, given the key of the module: constructors and `initParams` both use
 * it, so a re-initialised mixer has the weights a freshly built one has.
 *
 * @param key - The module's key.
 * @returns The key for the mixer's own parameters.
 * @example
 * ```ts
 * import { ownKey } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * ownKey(key(0)).words.length // => 2
 * ```
 */
export function ownKey(key: Key): Key {
  return foldIn(key, 1_000_003)
}

/**
 * A causal sequence mixer: the base of the layers that replace attention in a sequence model. A concrete mixer says
 * how to mix a sequence from a state (`mix`), what the state is before anything is read (`initialState`), and its own
 * parameters beside those of its parts (`ownParams`).
 *
 * @concept sequence-models
 */
export abstract class SequenceMixer extends Composite<MixerInput | Value, MixerOutput> {
  /** The width D of the sequence the mixer reads and writes. */
  abstract readonly dim: number

  protected constructor(parts: Readonly<Record<string, Module<never, unknown> | undefined>>, own: Params = {}) {
    super(parts, own)
  }

  /** The state before the first position of a batch. */
  abstract initialState(batch: number): MixerState

  /** The mixer's output and state for a sequence, from a state (or `undefined`: the start of a sequence). */
  protected abstract mix(params: Params, x: Value, state: MixerState | undefined, ctx: ApplyContext): MixerOutput

  /**
   * The eigenvalues of the mixer's state transition for given parameters, where it has a fixed one (state-space
   * layers, retention): what a training run shows to explain how far back the layer remembers. `undefined` for a mixer
   * whose transition depends on the input or that has no state transition.
   */
  transitionSpectrum(_params: Params = this.params): TransitionSpectrum | undefined {
    return undefined
  }

  /** The parameters the mixer owns beside its parts' (drawn from `key`, which is already {@link ownKey} of the module's). */
  protected ownParams(_key: Key): Params {
    return {}
  }

  override initParams(key: Key): Params {
    return { ...super.initParams(key), ...this.ownParams(ownKey(key)) }
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? { x: asTensor(x as TensorLike) } : x
  }

  apply(params: Params, input: MixerInput | Value, ctx: ApplyContext = {}): MixerOutput {
    const inp: MixerInput = isValue(input) ? { x: input } : input
    return this.mix(params, asTensor(inp.x), inp.state, ctx)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }
}
