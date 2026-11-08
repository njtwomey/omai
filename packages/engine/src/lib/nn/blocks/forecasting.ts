/**
 * Forecasting blocks: N-BEATS (Oreshkin et al. 2020), a stack of fully connected blocks in which each block reads the
 * residual of the lookback window, expands two small coefficient vectors over a basis into a backcast (what it
 * explains of the window) and a forecast, and passes the window minus its backcast on, with the forecasts summed. The
 * basis is generic (learned), trend (polynomials in time) or seasonality (Fourier harmonics), the latter two making
 * each block's contribution interpretable.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  matmul,
  stack,
  sub,
  type Tensor,
  type TensorLike,
  tensor,
  type Value,
} from '../../tensor/index.ts'
import { ReLU } from '../module/activations.ts'
import { Composite, Sequential } from '../module/containers.ts'
import { Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, type Module, type Params } from '../module/module.ts'

/** What a block's coefficients are expanded over. */
export type NBeatsBasis = 'generic' | 'trend' | 'seasonality'

/**
 * The trend basis: polynomials of time, `T[i, t] = (t / length)^i` for `i = 0 .. degree` and `t = 0 .. length - 1`.
 *
 * @param degree - The highest power.
 * @param length - The number of time steps.
 * @returns `[degree + 1, length]`.
 * @example
 * ```ts
 * import { trendBasis } from 'omai-engine/nn/blocks'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(trendBasis(2, 4)) // => [[1, 1, 1, 1], [0, 0.25, 0.5, 0.75], [0, 0.0625, 0.25, 0.5625]]
 * ```
 */
export function trendBasis(degree: number, length: number): Tensor {
  return tensor(Array.from({ length: degree + 1 }, (_, i) => Array.from({ length }, (_, t) => (t / length) ** i)))
}

/**
 * The seasonality basis: `cos(2 pi k t / length)` for `k = 0 .. harmonics - 1` and `sin(2 pi k t / length)` for
 * `k = 1 .. harmonics`, so the dimension is `2 harmonics`.
 *
 * @param harmonics - The number of harmonics.
 * @param length - The number of time steps.
 * @returns `[2 harmonics, length]`.
 * @example
 * ```ts
 * import { seasonalityBasis } from 'omai-engine/nn/blocks'
 * seasonalityBasis(3, 12).shape // => [6, 12]
 * ```
 */
export function seasonalityBasis(harmonics: number, length: number): Tensor {
  const rows: number[][] = []
  for (let k = 0; k < harmonics; k++)
    rows.push(Array.from({ length }, (_, t) => Math.cos((2 * Math.PI * k * t) / length)))
  for (let k = 1; k <= harmonics; k++)
    rows.push(Array.from({ length }, (_, t) => Math.sin((2 * Math.PI * k * t) / length)))
  return tensor(rows)
}

/** Options of {@link NBeatsBlock}. */
export interface NBeatsBlockOptions {
  /** The basis (default generic). */
  readonly basis?: NBeatsBasis
  /** Width of the fully connected layers (default 128). */
  readonly width?: number
  /** Number of fully connected layers (default 4). */
  readonly layers?: number
  /** Degree of the trend polynomial (default 3) or the number of seasonal harmonics (default 4); for the generic basis the coefficient size (default 32). */
  readonly size?: number
  readonly key?: Key
}

/** The output of a block: what it explains of the window and its forecast. */
export interface NBeatsBlockOutput {
  readonly backcast: Value
  readonly forecast: Value
}

/**
 * An N-BEATS block: fully connected layers with ReLU on the window, two linear maps to coefficient vectors, and the
 * basis expansion of each into the backcast (over the lookback) and the forecast (over the horizon). Trend and
 * seasonality bases are fixed, so only the coefficients are learned; the generic basis is a learned linear map.
 *
 * @reference Oreshkin, Carpov, Chapados and Bengio (2020), N-BEATS: neural basis expansion analysis for interpretable time series forecasting.
 * @example
 * ```ts
 * import { NBeatsBlock } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new NBeatsBlock(12, 4, { basis: 'trend' }).forward(zeros([2, 12])).forecast.shape // => [2, 4]
 * ```
 */
export class NBeatsBlock extends Composite<Value, NBeatsBlockOutput> {
  readonly kind = 'NBeatsBlock'
  readonly lookback: number
  readonly horizon: number
  readonly basis: NBeatsBasis
  /** The fixed backcast basis `[size, lookback]` (absent for the generic basis). */
  readonly backcastBasis: Tensor | undefined
  /** The fixed forecast basis `[size, horizon]` (absent for the generic basis). */
  readonly forecastBasis: Tensor | undefined

  constructor(lookback: number, horizon: number, options: NBeatsBlockOptions = {}) {
    const key = initKey('NBeatsBlock', options.key)
    const basis = options.basis ?? 'generic'
    const width = options.width ?? 128
    const layers = options.layers ?? 4
    const size = options.size ?? (basis === 'generic' ? 32 : basis === 'trend' ? 3 : 4)
    const dim = basis === 'seasonality' ? 2 * size : basis === 'trend' ? size + 1 : size
    const fc: Module[] = []
    for (let i = 0; i < layers; i++) {
      fc.push(new Linear(i === 0 ? lookback : width, width, { key: foldIn(key, i) }), new ReLU())
    }
    super({
      fc: new Sequential(...fc),
      thetaBackcast: new Linear(layers === 0 ? lookback : width, dim, { key: foldIn(key, 20) }),
      thetaForecast: new Linear(layers === 0 ? lookback : width, dim, { key: foldIn(key, 21) }),
      ...(basis === 'generic'
        ? {
            backcast: new Linear(dim, lookback, { key: foldIn(key, 22) }),
            forecast: new Linear(dim, horizon, { key: foldIn(key, 23) }),
          }
        : {}),
    })
    this.lookback = lookback
    this.horizon = horizon
    this.basis = basis
    if (basis === 'trend') {
      this.backcastBasis = trendBasis(size, lookback)
      this.forecastBasis = trendBasis(size, horizon)
    } else if (basis === 'seasonality') {
      this.backcastBasis = seasonalityBasis(size, lookback)
      this.forecastBasis = seasonalityBasis(size, horizon)
    }
  }

  /** The coefficients the block would expand for a window (before the basis). */
  coefficients(params: Params, x: Value, ctx: ApplyContext = {}): { backcast: Value; forecast: Value } {
    const p = this.parts(params, ctx)
    const h = p.run<Value, Value>('fc', x)
    const out = { backcast: p.run<Value, Value>('thetaBackcast', h), forecast: p.run<Value, Value>('thetaForecast', h) }
    p.done()
    return out
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): NBeatsBlockOutput {
    const theta = this.coefficients(params, x, ctx)
    if (this.basis === 'generic') {
      const p = this.parts(params, ctx)
      const out = {
        backcast: p.run<Value, Value>('backcast', theta.backcast),
        forecast: p.run<Value, Value>('forecast', theta.forecast),
      }
      p.done()
      return out
    }
    return {
      backcast: matmul(theta.backcast, this.backcastBasis as Tensor),
      forecast: matmul(theta.forecast, this.forecastBasis as Tensor),
    }
  }

  outputShape(s: readonly number[]): number[] {
    return [...s.slice(0, -1), this.horizon]
  }
}

/** One stack of an N-BEATS model: a basis and how many blocks it has. */
export interface NBeatsStack extends Omit<NBeatsBlockOptions, 'key'> {
  /** The number of blocks (default 3). */
  readonly blocks?: number
}

/** The output of {@link NBeats}. */
export interface NBeatsOutput {
  /** The forecast, the sum over every block `[B, horizon]`. */
  readonly forecast: Value
  /** What is left of the window after every block's backcast was subtracted `[B, lookback]`. */
  readonly residual: Value
  /** Each block's forecast `[blocks, B, horizon]`. */
  readonly contributions: Value
  /** Each block's backcast `[blocks, B, lookback]`. */
  readonly backcasts: Value
}

/** Options of {@link NBeats}. */
export interface NBeatsOptions {
  readonly stacks: readonly NBeatsStack[]
  readonly key?: Key
}

/**
 * The N-BEATS model: stacks of blocks in a doubly residual arrangement. Block `l` reads the residual `r_l`, its
 * backcast is subtracted, `r_{l+1} = r_l - backcast_l`, and its forecast is added to the output. Each block's forecast
 * and backcast are returned, so the contribution of a trend or seasonality block can be drawn.
 *
 * @example
 * ```ts
 * import { NBeats } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * const model = new NBeats(12, 4, { stacks: [{ basis: 'trend', blocks: 2 }, { basis: 'seasonality', blocks: 2 }] })
 * model.forward(zeros([3, 12])).contributions.shape // => [4, 3, 4]
 * ```
 */
export class NBeats extends Composite<TensorLike, NBeatsOutput> {
  readonly kind = 'NBeats'
  readonly lookback: number
  readonly horizon: number
  readonly blockCount: number

  constructor(lookback: number, horizon: number, options: NBeatsOptions) {
    const key = initKey('NBeats', options.key)
    if (options.stacks.length === 0) throw new OmaiError('value', 'NBeats: need at least one stack')
    const parts: Record<string, Module<never, unknown>> = {}
    let count = 0
    for (const stackSpec of options.stacks) {
      for (let b = 0; b < (stackSpec.blocks ?? 3); b++) {
        const { blocks: _blocks, ...rest } = stackSpec
        parts[`block${count}`] = new NBeatsBlock(lookback, horizon, { ...rest, key: foldIn(key, count) }) as never
        count += 1
      }
    }
    super(parts as never)
    this.lookback = lookback
    this.horizon = horizon
    this.blockCount = count
  }

  apply(params: Params, window: TensorLike, ctx: ApplyContext = {}): NBeatsOutput {
    const p = this.parts(params, ctx)
    let residual: Value = asTensor(window)
    let forecast: Value | undefined
    const forecasts: Value[] = []
    const backcasts: Value[] = []
    for (let i = 0; i < this.blockCount; i++) {
      const out = p.run<Value, NBeatsBlockOutput>(`block${i}`, residual)
      residual = sub(residual, out.backcast)
      forecast = forecast === undefined ? out.forecast : add(forecast, out.forecast)
      forecasts.push(out.forecast)
      backcasts.push(out.backcast)
    }
    p.done()
    return {
      forecast: forecast as Value,
      residual,
      contributions: stack(forecasts, 0),
      backcasts: stack(backcasts, 0),
    }
  }

  outputShape(s: readonly number[]): number[] {
    return [...s.slice(0, -1), this.horizon]
  }
}
