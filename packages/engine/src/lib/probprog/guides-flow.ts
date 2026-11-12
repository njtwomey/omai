/**
 * Normalising-flow guides beyond the inverse autoregressive flow of `autoIafNormal`: planar flows (Rezende and Mohamed
 * 2015), radial flows and RealNVP affine coupling layers, each a standard normal over the flat unconstrained latent
 * vector pushed through the library's bijectors and an affine layer, then onto each site's support. All start at the
 * identity, so the first steps are those of a mean-field Gaussian, and the flows bend it from there.
 */

import { OmaiError } from '../base/index.ts'
import { bijectors, compose, normal, type Support } from '../dists/index.ts'
import { foldIn, type Key, normal as normalDraws } from '../random/index.ts'
import {
  add,
  arange,
  asTensor,
  div,
  full,
  log,
  matmul,
  mul,
  primalOf,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tanh,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { type AutoGuide, type AutoGuideOptions, jointGuide, pushed } from './guides.ts'
import { type Model, param } from './primitives.ts'

/** The flow layers {@link autoFlowNormal} offers. */
export type FlowKind = 'planar' | 'radial' | 'coupling'

/** Options of {@link autoFlowNormal}. */
export interface AutoFlowOptions extends AutoGuideOptions {
  /** The layer: `planar` (default), `radial` or `coupling` (RealNVP, needs two or more latent dimensions). */
  readonly flow?: FlowKind
  /** The number of layers (default 3). */
  readonly numFlows?: number
  /** The hidden width of a coupling layer's conditioner (default 2 D + 2). */
  readonly hiddenDim?: number
}

/**
 * A normalising-flow guide: a standard normal over the flat unconstrained latent vector pushed through `numFlows` layers
 * of one kind (`planar`, `radial` or `coupling`, from the library's bijectors) and a learned affine map, then onto each
 * site's support (compare {@link autoIafNormal}, which is the inverse autoregressive flow). The layers start at the
 * identity. Parameters: `auto_loc`, `auto_scale`, and `auto_flow_<l>_*` per layer.
 *
 * @param model - The model.
 * @param options - The layer kind, the number of layers, the width and the initialisation.
 * @returns The guide.
 * @throws OmaiError with code `value` for a coupling flow over one dimension.
 * @concept normalizing-flows
 * @index function Normalising-flow guide with planar, radial or RealNVP coupling layers.
 * @example
 * ```ts
 * import { autoFlowNormal, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const guide = autoFlowNormal(() => { sample('x', normal(0, 1), { sampleShape: [2] }) }, { flow: 'planar', numFlows: 2 })
 * traceOf(seed(guide, key(0)), undefined).sites.has('auto_flow_1_u') // => true
 * ```
 */
export function autoFlowNormal<D = void>(model: Model<D>, options: AutoFlowOptions = {}): AutoGuide<D> {
  const kind = options.flow ?? 'planar'
  const numFlows = options.numFlows ?? 3
  const s0 = options.initScale ?? 0.1
  type Get = (name: string, init: (k: Key) => Value, support?: Support) => Value
  const layers = (D: number, get: Get): bijectors.Bijector[] => {
    const parts: bijectors.Bijector[] = []
    for (let l = 0; l < numFlows; l++) {
      const at = (n: string): string => `auto_flow_${l}_${n}`
      if (kind === 'planar') {
        parts.push(
          bijectors.planar(
            get(at('u'), (k) => mul(normalDraws(foldIn(k, 1), [D]), 0.01)),
            get(at('w'), (k) => mul(normalDraws(foldIn(k, 2), [D]), 0.01)),
            get(at('b'), () => zeros([])),
          ),
        )
      } else if (kind === 'radial') {
        parts.push(
          bijectors.radial(
            get(at('center'), () => zeros([D])),
            get(at('alpha'), () => full([], 1), { kind: 'positive' }),
            // softplus(0.5413) = 1: with alpha = 1 the map starts at the identity.
            get(at('beta'), () => full([], Math.log(Math.E - 1))),
          ),
        )
      } else {
        if (D < 2) throw new OmaiError('value', 'autoFlowNormal: a coupling flow needs two or more latent dimensions')
        const split = Math.floor(D / 2)
        const rest = D - split
        const H = options.hiddenDim ?? 2 * D + 2
        const w1 = get(at('w1'), (k) => mul(normalDraws(foldIn(k, 3), [split, H]), 1 / Math.sqrt(split)))
        const b1 = get(at('b1'), () => zeros([H]))
        const ws = get(at('shift'), () => zeros([H, rest]))
        const wl = get(at('log_scale'), () => zeros([H, rest]))
        const bs = get(at('shift_bias'), () => zeros([rest]))
        const bl = get(at('log_scale_bias'), () => zeros([rest]))
        parts.push(
          bijectors.affineCoupling(split, (xa) => {
            const h = tanh(add(matmul(xa, w1), b1))
            return { shift: add(matmul(h, ws), bs), logScale: add(matmul(h, wl), bl) }
          }),
        )
        // Alternate which half passes through.
        parts.push(bijectors.permute(toFlat(arange(D - 1, -1, -1))))
      }
    }
    return parts
  }
  const affineOf = (loc: Value, scale: Value): bijectors.Bijector =>
    bijectors.defineBijector({
      name: 'affineVector',
      inputRank: 1,
      forward: (x) => add(loc, mul(scale, x)),
      inverse: (y) => div(sub(y, loc), scale),
      forwardLogDetJacobian: (x) => add(sum(mul(x, 0), -1), sum(log(scale))),
    })
  const read =
    (params: Readonly<Record<string, TensorLike>>): Get =>
    (name) => {
      const v = params[name]
      if (v === undefined) throw new OmaiError('value', `autoFlowNormal: no parameter ${name}`)
      return primalOf(asTensor(v)) as Tensor
    }
  return jointGuide(
    model,
    `autoFlowNormal`,
    {
      latent: (D, _sites, init) => {
        const loc = param('auto_loc', init)
        const scale = param('auto_scale', full([D], s0), { support: { kind: 'positive' } })
        const flows = layers(D, (name, i, support) => param(name, i, support === undefined ? {} : { support }))
        return pushed(compose.independent(normal(zeros([D]), 1), 1), bijectors.chain([...flows, affineOf(loc, scale)]))
      },
      fitted: (params, D) => {
        const g = read(params)
        return compose.transformed(compose.independent(normal(zeros([D]), 1), 1), [
          ...layers(D, g),
          affineOf(
            g('auto_loc', () => zeros([D])),
            g('auto_scale', () => zeros([D])),
          ),
        ])
      },
      centre: (params) => read(params)('auto_loc', () => zeros([1])),
    },
    options,
  )
}
