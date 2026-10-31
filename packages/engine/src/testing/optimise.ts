import { valueAndGrad } from '../lib/autodiff/index.ts'
import type { Params } from '../lib/nn/module/module.ts'
import { adam, applyUpdates } from '../lib/optim/index.ts'
import { item, type Value } from '../lib/tensor/index.ts'

/** Minimise a loss of a parameter tree with Adam for a fixed number of full-batch steps; returns the parameters and every loss. */
export function minimise(
  params: Params,
  loss: (p: Params) => Value,
  steps: number,
  learningRate = 0.01,
): { readonly params: Params; readonly losses: readonly number[] } {
  const optimizer = adam({ learningRate })
  let state = optimizer.init(params)
  let current = params
  const losses: number[] = []
  for (let i = 0; i < steps; i++) {
    const { value, grad } = valueAndGrad(loss)(current)
    losses.push(item(value))
    const r = optimizer.update(grad, state, current)
    state = r.state
    current = applyUpdates(current, r.updates) as Params
  }
  return { params: current, losses }
}
