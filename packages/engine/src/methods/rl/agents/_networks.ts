// Private network plumbing of the policy-gradient, actor-critic, safe and imitation agents (methods/rl/agents): MLPs
// from lib/nn as pure functions of parameter trees.
import { type ActivationName, activationModule, mlp, type Params } from '../../../lib/nn/index.ts'
import { foldIn, type Key } from '../../../lib/random/index.ts'
import { mul, type Tensor, treeMap, type Value } from '../../../lib/tensor/index.ts'

/** An MLP as a pure function of its parameters. */
export interface Mlp {
  readonly params: Params
  apply(params: Params, x: Value): Value
}

/**
 * An MLP `sizes[0] -> ... -> sizes[n-1]` with the activation between layers and a linear output, initialised as PyTorch
 * initialises linear layers; `outputScale` shrinks the last layer (stable-baselines3 starts policy heads near zero).
 */
export function mlpNet(sizes: readonly number[], activationName: ActivationName, key: Key, outputScale = 1): Mlp {
  const net = mlp(sizes, { activation: () => activationModule(activationName), key })
  let params = net.params
  if (outputScale !== 1) {
    const last = String(2 * (sizes.length - 2))
    const layer = params[last] as Readonly<Record<string, Tensor>>
    params = { ...params, [last]: treeMap((p) => mul(p, outputScale), layer) }
  }
  return { params, apply: (p, x) => net.apply(p, x) }
}

/** Independent initialisation keys for the networks of one agent. */
export function networkKeys(key: Key, n: number): Key[] {
  return Array.from({ length: n }, (_, i) => foldIn(key, 1000 + i))
}
