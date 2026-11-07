/**
 * The supervised learner behind the imitation agents (imitation.ts), private to the agents: an MLP classifier for
 * discrete actions or a regressor for box actions, trained on expert demonstrations with lib/losses.
 */

import { argmaxOf, MdmlError } from '../base/index.ts'
import { crossEntropy, mse } from '../losses/index.ts'
import type { ActivationName, Params } from '../nn/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import { toFlat, type Value } from '../tensor/index.ts'
import { observationSize, rowsTensor, type Spaces, vectorTensor } from './_common.ts'
import { mlpNet } from './_networks.ts'
import type { PolicyAction } from './_on-policy.ts'

/** One expert decision: an observation and the expert's action. */
export interface Demonstration<A> {
  readonly observation: readonly number[]
  readonly action: A
}

/** A learner network: logits over discrete actions, or a regressor of continuous ones. */
export interface Learner {
  readonly params: Params
  readonly discrete: boolean
  readonly d: number
  readonly k: number
  /** The learner's action: the most likely class, or the regression output. */
  predict(params: Params, observation: readonly number[]): PolicyAction
  /** Cross-entropy (discrete) or squared error (box) on demonstrations. */
  loss(params: Params, data: readonly Demonstration<PolicyAction>[]): Value
  /** Agreement with the demonstrations (accuracy, or minus the mean squared error). */
  agreement(params: Params, data: readonly Demonstration<PolicyAction>[]): number
}

export function learner(
  env: Spaces,
  options: { readonly hidden?: readonly number[]; readonly activation?: ActivationName; readonly key?: Key },
): Learner {
  const d = observationSize(env.observationSpace)
  const discrete = env.actionSpace.kind === 'discrete'
  const k =
    env.actionSpace.kind === 'discrete'
      ? env.actionSpace.n
      : env.actionSpace.kind === 'box'
        ? env.actionSpace.shape.reduce((a, b) => a * b, 1)
        : 0
  if (k === 0) throw new MdmlError('value', 'imitation: needs a discrete or box action space')
  const head = mlpNet([d, ...(options.hidden ?? [32]), k], options.activation ?? 'relu', options.key ?? makeKey(0))
  const predict = (p: Params, obs: readonly number[]): PolicyAction => {
    const out = toFlat(head.apply(p, rowsTensor([obs], d)))
    return discrete ? argmaxOf(out) : out
  }
  const loss = (p: Params, data: readonly Demonstration<PolicyAction>[]): Value => {
    const x = rowsTensor(
      data.map((e) => e.observation),
      d,
    )
    const out = head.apply(p, x)
    return discrete
      ? crossEntropy(out, vectorTensor(data.map((e) => e.action as number)))
      : mse(
          out,
          rowsTensor(
            data.map((e) => e.action as readonly number[]),
            k,
          ),
        )
  }
  const agreement = (p: Params, data: readonly Demonstration<PolicyAction>[]): number => {
    if (data.length === 0) return 0
    const x = rowsTensor(
      data.map((e) => e.observation),
      d,
    )
    const out = toFlat(head.apply(p, x))
    let total = 0
    for (const [i, e] of data.entries()) {
      const row = out.slice(i * k, (i + 1) * k)
      if (discrete) total += argmaxOf(row) === e.action ? 1 : 0
      else for (const [j, a] of (e.action as readonly number[]).entries()) total -= ((row[j] ?? 0) - a) ** 2 / k
    }
    return total / data.length
  }
  return { params: head.params, predict, loss, agreement, discrete, d, k }
}
