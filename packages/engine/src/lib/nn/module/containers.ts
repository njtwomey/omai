import { foldIn, type Key } from '../../random/index.ts'
import type { Value } from '../../tensor/index.ts'
import { Linear, ReLU } from './layers.ts'
import { type ApplyContext, Module, type Params } from './module.ts'

/**
 * Modules applied in order (torch.nn.Sequential). Parameters are keyed by position, as PyTorch's state dict
 * (`0.weight`, `2.bias`); each child gets its own key for stochastic layers.
 *
 * @concept modules
 * @example
 * ```ts
 * import { Sequential, Linear, ReLU } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * const net = new Sequential(new Linear(2, 8, { key: key(0) }), new ReLU(), new Linear(8, 1, { key: key(1) }))
 * Object.keys(net.stateDict()) // => ['0.bias', '0.weight', '2.bias', '2.weight']
 * ```
 */
export class Sequential extends Module {
  readonly kind = 'Sequential'
  readonly layers: readonly Module[]

  constructor(...layers: readonly Module[]) {
    super(Object.fromEntries(layers.flatMap((l, i) => (l.parameterCount() > 0 ? [[String(i), l.params]] : []))))
    this.layers = layers
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    return this.layers.reduce(
      (h, layer, i) =>
        layer.apply(
          (params[String(i)] ?? {}) as Params,
          h,
          ctx.key === undefined ? ctx : { ...ctx, key: foldIn(ctx.key, i) },
        ),
      x,
    )
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.layers.reduce<number[]>((s, layer) => layer.outputShape(s), [...inputShape])
  }

  override children(): readonly (readonly [string, Module])[] {
    return this.layers.map((l, i) => [String(i), l] as const)
  }

  /** Replace the parameters and hand each child its part, so the children stay in step with the container. */
  override setParams(params: Params): this {
    this.params = params
    this.layers.forEach((l, i) => {
      const p = params[String(i)]
      if (p !== undefined) l.setParams(p as Params)
    })
    return this
  }
}

/**
 * A multilayer perceptron: Linear layers of the given widths with an activation between them.
 *
 * @param widths - Layer widths from input to output (at least two).
 * @param options - The hidden activation (default ReLU) and an initialisation key.
 * @returns The network.
 * @example
 * ```ts
 * import { mlp } from 'whitebox-engine/nn'
 * mlp([4, 16, 16, 1]).parameterCount() // => 369
 * ```
 */
export function mlp(
  widths: readonly number[],
  options: { readonly activation?: () => Module; readonly key?: Key } = {},
): Sequential {
  const layers: Module[] = []
  for (let i = 0; i + 1 < widths.length; i++) {
    layers.push(
      new Linear(widths[i] ?? 0, widths[i + 1] ?? 0, options.key === undefined ? {} : { key: foldIn(options.key, i) }),
    )
    if (i + 2 < widths.length) layers.push(options.activation === undefined ? new ReLU() : options.activation())
  }
  return new Sequential(...layers)
}
