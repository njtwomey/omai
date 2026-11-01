/**
 * Structured state-space layers as sequence mixers: S4D (a diagonal complex state per channel, HiPPO-inspired
 * eigenvalues, the layer as one long causal convolution for training and as a recurrence for generation), S4 (the
 * same with HiPPO's normal-plus-low-rank matrix, bilinear discretisation), S5 (one state shared by all channels,
 * complex diagonal, run as a parallel scan) and the linear recurrent unit. Complex parameters are stored as real and
 * imaginary parts. In every layer the convolution (or scan) form and the recurrent form compute the same function,
 * and the state they hand on is the same too.
 */

import { MdmlError } from '../../base/index.ts'
import { solve } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, normal, uniform } from '../../random/index.ts'
import {
  add,
  type ComplexValue,
  cDiv,
  cExp,
  cMul,
  complex,
  concat,
  cSub,
  div,
  einsum,
  exp,
  expandDims,
  eye,
  flip,
  fromStorage,
  matmul,
  minimum,
  mul,
  neg,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  swapaxes,
  type Tensor,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import {
  channelCausalConvolution,
  type Discretisation,
  type LRUParams,
  lruEigenvalues,
  lruInit,
  lruScan,
  parallelScanComplex,
} from '../functional/state-space.ts'
import { type ApplyContext, initKey, type Params } from '../module/module.ts'
import { type DiagonalHippo, nplrHippo, s4dEigenvalues } from './hippo.ts'
import { type MixerOutput, type MixerState, ownKey, SequenceMixer, type TransitionSpectrum } from './mixer.ts'

/** How a state-space layer computes: as a convolution (or parallel scan) over the whole sequence, or step by step. */
export type SsmMode = 'convolution' | 'recurrence'

const f64 = (a: ArrayLike<number>, shape: number[]): Tensor => fromStorage(Float64Array.from(a), shape, 'float64')
const dtInit = (key: Key, shape: number[], dtMin: number, dtMax: number): Tensor =>
  uniform(key, shape, { low: Math.log(dtMin), high: Math.log(dtMax) }) as Tensor
const col = (v: Value): Value => expandDims(v, 1)

/** The complex state `{re, im}` `[B, ...]` of a layer, zeros when there is none. */
function stateOf(
  state: MixerState | undefined,
  shape: number[],
): { readonly re: Value; readonly im: Value; readonly given: boolean } {
  if (state === undefined) return { re: zeros(shape), im: zeros(shape), given: false }
  return { re: state['re'] as Value, im: state['im'] as Value, given: true }
}

/** Options of {@link S4D}. */
export interface S4DOptions {
  /** The state size N per channel (even; N / 2 complex eigenvalues stand for N conjugate-paired states). Default 16. */
  readonly stateSize?: number
  /** The eigenvalue initialisation (default `lin`). */
  readonly init?: DiagonalHippo
  /** The range of the step Delta, log-uniform at initialisation (default 1e-3 to 0.1). */
  readonly dtMin?: number
  readonly dtMax?: number
  readonly discretisation?: Discretisation
  /** `convolution` (default) or `recurrence`; a call that hands in a state always recurs. */
  readonly mode?: SsmMode
  readonly key?: Key
}

function s4dOwn(dim: number, o: Required<Omit<S4DOptions, 'key'>>, key: Key): Params {
  const n2 = o.stateSize / 2
  const lam = s4dEigenvalues(o.init, n2)
  const tile = (v: ArrayLike<number>, f: (x: number) => number): Tensor =>
    f64(
      Array.from({ length: dim * n2 }, (_, i) => f(v[i % n2] ?? 0)),
      [dim, n2],
    )
  const scale = Math.sqrt(0.5)
  return {
    logDt: dtInit(foldIn(key, 0), [dim], o.dtMin, o.dtMax),
    logNegRe: tile(lam.re.data, (x) => Math.log(-x)),
    im: tile(lam.im.data, (x) => x),
    cRe: mul(normal(foldIn(key, 1), [dim, n2]), scale) as Tensor,
    cIm: mul(normal(foldIn(key, 2), [dim, n2]), scale) as Tensor,
    d: normal(foldIn(key, 3), [dim]),
  }
}

/** The discretised diagonal system of S4D: Abar `[H, N/2]` and C Bbar (B = 1 is absorbed into the readout). */
function s4dDiscrete(p: Params, method: Discretisation): { readonly a: ComplexValue; readonly ct: ComplexValue } {
  const dt = col(exp(p['logDt'] as Value))
  const lam: ComplexValue = { re: neg(exp(p['logNegRe'] as Value)), im: p['im'] as Value }
  const dtA: ComplexValue = { re: mul(lam.re, dt), im: mul(lam.im, dt) }
  let a: ComplexValue
  let bbar: ComplexValue
  if (method === 'zoh') {
    a = cExp(dtA)
    bbar = cDiv(cSub(a, complex(1)), lam)
  } else {
    const den: ComplexValue = { re: sub(1, div(dtA.re, 2)), im: neg(div(dtA.im, 2)) }
    a = cDiv({ re: add(1, div(dtA.re, 2)), im: div(dtA.im, 2) }, den)
    bbar = cDiv({ re: add(dt, mul(lam.re, 0)), im: mul(lam.im, 0) }, den)
  }
  return { a, ct: cMul({ re: p['cRe'] as Value, im: p['cIm'] as Value }, bbar) }
}

/** The powers Abar^k, k < L, stacked on a last axis: `{re, im}` `[H, N/2, L]`. */
function powers(a: ComplexValue, length: number): ComplexValue {
  const rs: Value[] = []
  const is: Value[] = []
  let p: ComplexValue = { re: add(mul(a.re, 0), 1), im: mul(a.im, 0) }
  for (let k = 0; k < length; k++) {
    rs.push(p.re)
    is.push(p.im)
    p = cMul(p, a)
  }
  return { re: stack(rs, 2), im: stack(is, 2) }
}

/**
 * The S4D layer (Gu, Gupta, Goel and Re 2022): per channel a diagonal complex state-space model whose conjugate
 * pairs are implicit (the output is twice the real part), `h_t = Abar h_{t-1} + u_t`, `y_t = 2 Re(C Bbar h_t) + D u_t`,
 * with `Abar = exp(Delta lambda)` (zero-order hold) or the bilinear form. Training computes the layer as the causal
 * convolution with the kernel `K_k = 2 Re(C Bbar Abar^k)`; generation runs the recurrence one token at a time; both
 * give the same outputs and the same final state `[B, H, N/2]` (real and imaginary parts).
 *
 * @concept state-space-models
 * @reference Gu, Gupta, Goel and Re (2022), On the parameterization and initialization of diagonal state space models.
 * @example
 * ```ts
 * import { S4D } from 'mdml-engine/nn/blocks'
 * import { zeros } from 'mdml-engine/tensor'
 * new S4D(4, { stateSize: 8 }).forward(zeros([1, 6, 4])).output.shape // => [1, 6, 4]
 * ```
 */
export class S4D extends SequenceMixer {
  readonly kind = 'S4D'
  readonly dim: number
  readonly stateSize: number
  readonly mode: SsmMode
  private readonly options: Required<Omit<S4DOptions, 'key'>>

  constructor(dim: number, options: S4DOptions = {}) {
    const o = {
      stateSize: options.stateSize ?? 16,
      init: options.init ?? 'lin',
      dtMin: options.dtMin ?? 1e-3,
      dtMax: options.dtMax ?? 0.1,
      discretisation: options.discretisation ?? 'zoh',
      mode: options.mode ?? 'convolution',
    } as const
    if (o.stateSize % 2 !== 0) throw new MdmlError('value', 'S4D: the state size must be even')
    super({}, s4dOwn(dim, o, ownKey(initKey('S4D', options.key))))
    this.dim = dim
    this.stateSize = o.stateSize
    this.mode = o.mode
    this.options = o
  }

  protected override ownParams(key: Key): Params {
    return s4dOwn(this.dim, this.options, key)
  }

  override transitionSpectrum(params: Params = this.params): TransitionSpectrum {
    return {
      domain: 'continuous',
      re: Array.from(toFlat(neg(exp(params['logNegRe'] as Value)))),
      im: Array.from(toFlat(params['im'] as Value)),
    }
  }

  initialState(batch: number): MixerState {
    const shape = [batch, this.dim, this.stateSize / 2]
    return { re: zeros(shape), im: zeros(shape) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, _ctx: ApplyContext): MixerOutput {
    const [bs = 1, l = 0, h = 1] = x.shape
    const n2 = this.stateSize / 2
    const { a, ct } = s4dDiscrete(params, this.options.discretisation)
    const skip = mul(x, params['d'] as Value)
    const s0 = stateOf(state, [bs, h, n2])
    if (this.mode === 'convolution' && !s0.given) {
      const pw = powers(a, l)
      const kernel = mul(sum(sub(mul(expandDims(ct.re, 2), pw.re), mul(expandDims(ct.im, 2), pw.im)), 1), 2) // [H, L]
      const y = add(channelCausalConvolution(x, kernel), skip)
      // The state after the last step: sum_k Abar^k u_{L-1-k}.
      const uRev = swapaxes(flip(x, 1), 1, 2) // [B, H, L]
      return {
        output: y,
        state: { re: einsum('hnk,bhk->bhn', pw.re, uRev), im: einsum('hnk,bhk->bhn', pw.im, uRev) },
      }
    }
    let hr = s0.re
    let hi = s0.im
    const ys: Value[] = []
    for (let t = 0; t < l; t++) {
      const u = reshape(sliceAxis(x, 1, t, t + 1), [bs, h, 1])
      const nr = add(sub(mul(a.re, hr), mul(a.im, hi)), u)
      const ni = add(mul(a.re, hi), mul(a.im, hr))
      hr = nr
      hi = ni
      ys.push(mul(sum(sub(mul(ct.re, hr), mul(ct.im, hi)), 2), 2))
    }
    return { output: add(stack(ys, 1), skip), state: { re: hr, im: hi } }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, state=${this.stateSize}, ${this.options.init}, ${this.mode}`
  }
}

/** Options of {@link S4}. */
export interface S4Options {
  /** The state size N (default 16). */
  readonly stateSize?: number
  /** The HiPPO measure whose normal-plus-low-rank matrix initialises the layer (default `legs`). */
  readonly measure?: 'legs' | 'legt' | 'lagt'
  readonly dtMin?: number
  readonly dtMax?: number
  /** The window of `legt` in time units of one step (default 1 / dtMax-ish: `1 / 0.1`). */
  readonly theta?: number
  readonly mode?: SsmMode
  readonly key?: Key
}

function s4Own(dim: number, o: Required<Omit<S4Options, 'key'>>, key: Key): Params {
  const f = nplrHippo(o.measure, o.stateSize, o.measure === 'legt' ? { theta: o.theta } : {})
  const scale = Math.sqrt(0.5)
  return {
    logDt: dtInit(foldIn(key, 0), [dim], o.dtMin, o.dtMax),
    lamRe: f.lambda.re,
    lamIm: f.lambda.im,
    pRe: f.pTilde.re,
    pIm: f.pTilde.im,
    bRe: f.bTilde.re,
    bIm: f.bTilde.im,
    cRe: mul(normal(foldIn(key, 1), [dim, o.stateSize]), scale) as Tensor,
    cIm: mul(normal(foldIn(key, 2), [dim, o.stateSize]), scale) as Tensor,
    d: normal(foldIn(key, 3), [dim]),
  }
}

/**
 * The S4 layer (Gu, Goel and Re 2022): one linear state-space model per channel whose matrix is HiPPO's, in its
 * eigenbasis `A = diag(lambda) - P P^H` (complex, normal plus low rank), shared across channels, with a step Delta,
 * a readout C and a skip D of its own per channel; discretised with the bilinear transform. The convolution kernel is
 * `K_k = Re(C Abar^k Bbar)`; the recurrence `h_t = Abar h_{t-1} + Bbar u_t`, `y_t = Re(C h_t) + D u_t` gives the same
 * outputs and the same final state `[B, H, N]`. Unlike the paper's Cauchy-kernel algorithm, the kernel is computed by
 * repeated application of `Abar` (O(L N^2) per channel), which is simple and exact; the state size is for the full
 * complex state (no conjugate halving).
 *
 * @concept state-space-models
 * @reference Gu, Goel and Re (2022), Efficiently modeling long sequences with structured state spaces.
 * @example
 * ```ts
 * import { S4 } from 'mdml-engine/nn/blocks'
 * import { zeros } from 'mdml-engine/tensor'
 * new S4(2, { stateSize: 4 }).forward(zeros([1, 5, 2])).output.shape // => [1, 5, 2]
 * ```
 */
export class S4 extends SequenceMixer {
  readonly kind = 'S4'
  readonly dim: number
  readonly stateSize: number
  readonly mode: SsmMode
  private readonly options: Required<Omit<S4Options, 'key'>>

  constructor(dim: number, options: S4Options = {}) {
    const o = {
      stateSize: options.stateSize ?? 16,
      measure: options.measure ?? 'legs',
      dtMin: options.dtMin ?? 1e-3,
      dtMax: options.dtMax ?? 0.1,
      theta: options.theta ?? 10,
      mode: options.mode ?? 'convolution',
    } as const
    super({}, s4Own(dim, o, ownKey(initKey('S4', options.key))))
    this.dim = dim
    this.stateSize = o.stateSize
    this.mode = o.mode
    this.options = o
  }

  protected override ownParams(key: Key): Params {
    return s4Own(this.dim, this.options, key)
  }

  override transitionSpectrum(params: Params = this.params): TransitionSpectrum {
    return {
      domain: 'continuous',
      re: Array.from(toFlat(minimum(params['lamRe'] as Value, 0))),
      im: Array.from(toFlat(params['lamIm'] as Value)),
    }
  }

  initialState(batch: number): MixerState {
    const shape = [batch, this.dim, this.stateSize]
    return { re: zeros(shape), im: zeros(shape) }
  }

  /** The bilinear discretisation per channel: Abar `[H, N, N]` and Bbar `[H, N]`, complex. */
  private discretise(p: Params): { readonly a: ComplexValue; readonly b: ComplexValue } {
    const n = this.stateSize
    const lamRe = minimum(p['lamRe'] as Value, 0)
    const lamIm = p['lamIm'] as Value
    const pr = p['pRe'] as Value
    const pi = p['pIm'] as Value
    const eyeN = eye(n)
    // A = diag(lambda) - P P^H.
    const aRe = sub(mul(eyeN, lamRe), add(matmul(pr, swapaxes(pr, 0, 1)), matmul(pi, swapaxes(pi, 0, 1))))
    const aIm = sub(mul(eyeN, lamIm), sub(matmul(pi, swapaxes(pr, 0, 1)), matmul(pr, swapaxes(pi, 0, 1))))
    const bRe = reshape(p['bRe'] as Value, [n, 1])
    const bIm = reshape(p['bIm'] as Value, [n, 1])
    const dt = exp(p['logDt'] as Value)
    const aRs: Value[] = []
    const aIs: Value[] = []
    const bRs: Value[] = []
    const bIs: Value[] = []
    for (let c = 0; c < this.dim; c++) {
      const step = reshape(sliceAxis(dt, 0, c, c + 1), [])
      const half = mul(step, 0.5)
      // (I - Delta/2 A) X = [I + Delta/2 A | Delta B] as one real system on (re, im).
      const mRe = sub(eyeN, mul(aRe, half))
      const mIm = neg(mul(aIm, half))
      const yRe = concat([add(eyeN, mul(aRe, half)), mul(bRe, step)], 1)
      const yIm = concat([mul(aIm, half), mul(bIm, step)], 1)
      const big = concat([concat([mRe, neg(mIm)], 1), concat([mIm, mRe], 1)], 0)
      const x = solve(big, concat([yRe, yIm], 0)) // [2N, N + 1]
      const re = sliceAxis(x, 0, 0, n)
      const im = sliceAxis(x, 0, n, 2 * n)
      aRs.push(sliceAxis(re, 1, 0, n))
      aIs.push(sliceAxis(im, 1, 0, n))
      bRs.push(reshape(sliceAxis(re, 1, n, n + 1), [n]))
      bIs.push(reshape(sliceAxis(im, 1, n, n + 1), [n]))
    }
    return { a: { re: stack(aRs, 0), im: stack(aIs, 0) }, b: { re: stack(bRs, 0), im: stack(bIs, 0) } }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, _ctx: ApplyContext): MixerOutput {
    const [bs = 1, l = 0, h = 1] = x.shape
    const n = this.stateSize
    const { a, b } = this.discretise(params)
    const cRe = params['cRe'] as Value
    const cIm = params['cIm'] as Value
    const skip = mul(x, params['d'] as Value)
    const s0 = stateOf(state, [bs, h, n])
    if (this.mode === 'convolution' && !s0.given) {
      // v_k = Abar^k Bbar, so that the kernel is K_k = Re(C v_k) and the final state is sum_k v_k u_{L-1-k}.
      const vr: Value[] = []
      const vi: Value[] = []
      let r = b.re
      let i = b.im
      for (let k = 0; k < l; k++) {
        vr.push(r)
        vi.push(i)
        const nr = sub(einsum('hij,hj->hi', a.re, r), einsum('hij,hj->hi', a.im, i))
        const ni = add(einsum('hij,hj->hi', a.re, i), einsum('hij,hj->hi', a.im, r))
        r = nr
        i = ni
      }
      const VR = stack(vr, 0) // [L, H, N]
      const VI = stack(vi, 0)
      const kernel = swapaxes(sum(sub(mul(cRe, VR), mul(cIm, VI)), 2), 0, 1) // [H, L]
      const uRev = swapaxes(flip(x, 1), 1, 2)
      return {
        output: add(channelCausalConvolution(x, kernel), skip),
        state: { re: einsum('lhn,bhl->bhn', VR, uRev), im: einsum('lhn,bhl->bhn', VI, uRev) },
      }
    }
    let hr = s0.re
    let hi = s0.im
    const ys: Value[] = []
    for (let t = 0; t < l; t++) {
      const u = reshape(sliceAxis(x, 1, t, t + 1), [bs, h, 1])
      const nr = add(sub(einsum('hij,bhj->bhi', a.re, hr), einsum('hij,bhj->bhi', a.im, hi)), mul(b.re, u))
      const ni = add(add(einsum('hij,bhj->bhi', a.re, hi), einsum('hij,bhj->bhi', a.im, hr)), mul(b.im, u))
      hr = nr
      hi = ni
      ys.push(sum(sub(mul(cRe, hr), mul(cIm, hi)), 2))
    }
    return { output: add(stack(ys, 1), skip), state: { re: hr, im: hi } }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, state=${this.stateSize}, ${this.options.measure}, ${this.mode}`
  }
}

/** Options of {@link S5}. */
export interface S5Options {
  /** The state size P of the one state shared by all channels (default 16). */
  readonly stateSize?: number
  /** The number of HiPPO blocks the state is tiled from (default 1; must divide the state size). */
  readonly blocks?: number
  readonly dtMin?: number
  readonly dtMax?: number
  /** `convolution` (default): a parallel scan over the sequence; `recurrence`: step by step. */
  readonly mode?: SsmMode
  readonly key?: Key
}

function s5Own(dim: number, o: Required<Omit<S5Options, 'key'>>, key: Key): Params {
  const p = o.stateSize
  const m = p / o.blocks
  const f = nplrHippo('legs', m)
  const vr = f.v.re.data
  const vi = f.v.im.data
  const lamRe = new Float64Array(p)
  const lamIm = new Float64Array(p)
  const bRe = new Float64Array(p * dim)
  const bIm = new Float64Array(p * dim)
  const cRe = new Float64Array(dim * p)
  const cIm = new Float64Array(dim * p)
  for (let blk = 0; blk < o.blocks; blk++) {
    const braw = normal(foldIn(key, 10 + blk), [m, dim]).data
    const craw = normal(foldIn(key, 100 + blk), [dim, m]).data
    for (let n = 0; n < m; n++) {
      lamRe[blk * m + n] = f.lambda.re.data[n] ?? 0
      lamIm[blk * m + n] = f.lambda.im.data[n] ?? 0
      for (let j = 0; j < dim; j++) {
        // B~ = V^H B (lecun-scaled), C~ = C V.
        let sr = 0
        let si = 0
        for (let q = 0; q < m; q++) {
          sr += (vr[q * m + n] ?? 0) * (braw[q * dim + j] ?? 0)
          si -= (vi[q * m + n] ?? 0) * (braw[q * dim + j] ?? 0)
        }
        bRe[(blk * m + n) * dim + j] = sr / Math.sqrt(dim)
        bIm[(blk * m + n) * dim + j] = si / Math.sqrt(dim)
        let tr = 0
        let ti = 0
        for (let q = 0; q < m; q++) {
          tr += (craw[j * m + q] ?? 0) * (vr[q * m + n] ?? 0)
          ti += (craw[j * m + q] ?? 0) * (vi[q * m + n] ?? 0)
        }
        cRe[j * p + blk * m + n] = tr / Math.sqrt(m)
        cIm[j * p + blk * m + n] = ti / Math.sqrt(m)
      }
    }
  }
  return {
    logDt: dtInit(foldIn(key, 0), [p], o.dtMin, o.dtMax),
    logNegRe: f64(
      Array.from(lamRe, (x) => Math.log(-x)),
      [p],
    ),
    lamIm: f64(lamIm, [p]),
    bRe: f64(bRe, [p, dim]),
    bIm: f64(bIm, [p, dim]),
    cRe: f64(cRe, [dim, p]),
    cIm: f64(cIm, [dim, p]),
    d: normal(foldIn(key, 3), [dim]),
  }
}

/**
 * The S5 layer (Smith, Warrington and Linderman 2023): a single multi-input multi-output state-space model with a
 * complex diagonal state of size P shared by all H channels (rather than H independent single-input models),
 * initialised from HiPPO-LegS in its eigenbasis (`B~ = V^H B`, `C~ = C V`), a step per state, zero-order hold. The
 * layer is run as a parallel scan (logarithmic depth) or step by step, with the same outputs and final state `[B, P]`.
 *
 * @concept state-space-models
 * @reference Smith, Warrington and Linderman (2023), Simplified state space layers for sequence modeling.
 * @example
 * ```ts
 * import { S5 } from 'mdml-engine/nn/blocks'
 * import { zeros } from 'mdml-engine/tensor'
 * new S5(3, { stateSize: 4 }).forward(zeros([2, 5, 3])).output.shape // => [2, 5, 3]
 * ```
 */
export class S5 extends SequenceMixer {
  readonly kind = 'S5'
  readonly dim: number
  readonly stateSize: number
  readonly mode: SsmMode
  private readonly options: Required<Omit<S5Options, 'key'>>

  constructor(dim: number, options: S5Options = {}) {
    const o = {
      stateSize: options.stateSize ?? 16,
      blocks: options.blocks ?? 1,
      dtMin: options.dtMin ?? 1e-3,
      dtMax: options.dtMax ?? 0.1,
      mode: options.mode ?? 'convolution',
    } as const
    if (o.stateSize % o.blocks !== 0) throw new MdmlError('value', 'S5: the blocks must divide the state size')
    super({}, s5Own(dim, o, ownKey(initKey('S5', options.key))))
    this.dim = dim
    this.stateSize = o.stateSize
    this.mode = o.mode
    this.options = o
  }

  protected override ownParams(key: Key): Params {
    return s5Own(this.dim, this.options, key)
  }

  override transitionSpectrum(params: Params = this.params): TransitionSpectrum {
    return {
      domain: 'continuous',
      re: Array.from(toFlat(neg(exp(params['logNegRe'] as Value)))),
      im: Array.from(toFlat(params['lamIm'] as Value)),
    }
  }

  initialState(batch: number): MixerState {
    return { re: zeros([batch, this.stateSize]), im: zeros([batch, this.stateSize]) }
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, _ctx: ApplyContext): MixerOutput {
    const [bs = 1, l = 0] = x.shape
    const p = this.stateSize
    const dt = exp(params['logDt'] as Value)
    const lam: ComplexValue = { re: neg(exp(params['logNegRe'] as Value)), im: params['lamIm'] as Value }
    const a = cExp({ re: mul(lam.re, dt), im: mul(lam.im, dt) })
    const factor = cDiv(cSub(a, complex(1)), lam) // [P]
    const bRe = params['bRe'] as Value
    const bIm = params['bIm'] as Value
    const bbar = {
      re: sub(mul(bRe, expandDims(factor.re, 1)), mul(bIm, expandDims(factor.im, 1))),
      im: add(mul(bRe, expandDims(factor.im, 1)), mul(bIm, expandDims(factor.re, 1))),
    }
    const bu = {
      re: matmul(x, swapaxes(bbar.re, 0, 1)), // [B, L, P]
      im: matmul(x, swapaxes(bbar.im, 0, 1)),
    }
    const s0 = state === undefined ? undefined : { re: state['re'] as Value, im: state['im'] as Value }
    let states: { readonly re: Value; readonly im: Value }
    if (this.mode === 'convolution') {
      states = parallelScanComplex(a, bu, 1, s0)
    } else {
      let hr = s0?.re ?? zeros([bs, p])
      let hi = s0?.im ?? zeros([bs, p])
      const rs: Value[] = []
      const is: Value[] = []
      for (let t = 0; t < l; t++) {
        const ur = reshape(sliceAxis(bu.re, 1, t, t + 1), [bs, p])
        const ui = reshape(sliceAxis(bu.im, 1, t, t + 1), [bs, p])
        const nr = add(sub(mul(a.re, hr), mul(a.im, hi)), ur)
        const ni = add(add(mul(a.re, hi), mul(a.im, hr)), ui)
        hr = nr
        hi = ni
        rs.push(hr)
        is.push(hi)
      }
      states = { re: stack(rs, 1), im: stack(is, 1) }
    }
    const y = sub(
      matmul(states.re, swapaxes(params['cRe'] as Value, 0, 1)),
      matmul(states.im, swapaxes(params['cIm'] as Value, 0, 1)),
    )
    const last = (v: Value): Value => reshape(sliceAxis(v, 1, l - 1, l), [bs, p])
    return {
      output: add(y, mul(x, params['d'] as Value)),
      state: { re: last(states.re), im: last(states.im) },
    }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, state=${this.stateSize}, ${this.mode}`
  }
}

/** Options of {@link LRUMixer}. */
export interface LRUMixerOptions {
  /** The complex state size N (default 16). */
  readonly stateSize?: number
  /** The ring of initial eigenvalue moduli (default 0 to 1) and the largest initial phase (default 2 pi). */
  readonly rMin?: number
  readonly rMax?: number
  readonly maxPhase?: number
  readonly key?: Key
}

const lruTree = (p: LRUParams): Params => ({ ...p })

/**
 * The linear recurrent unit (Orvieto et al. 2023) as a mixer: complex diagonal recurrence with eigenvalues
 * `exp(-exp(nu) + i exp(theta))` (stable by construction) and the input normalisation `gamma`, over the functional
 * {@link lruScan}. The state is the complex hidden state `[B, N]`.
 *
 * @concept state-space-models
 * @reference Orvieto et al. (2023), Resurrecting recurrent neural networks for long sequences.
 * @example
 * ```ts
 * import { LRUMixer } from 'mdml-engine/nn/blocks'
 * import { zeros } from 'mdml-engine/tensor'
 * new LRUMixer(3, { stateSize: 4 }).forward(zeros([1, 5, 3])).output.shape // => [1, 5, 3]
 * ```
 */
export class LRUMixer extends SequenceMixer {
  readonly kind = 'LRU'
  readonly dim: number
  readonly stateSize: number
  private readonly options: LRUMixerOptions

  constructor(dim: number, options: LRUMixerOptions = {}) {
    const n = options.stateSize ?? 16
    super({}, lruTree(lruInit(ownKey(initKey('LRU', options.key)), n, dim, options)))
    this.dim = dim
    this.stateSize = n
    this.options = options
  }

  protected override ownParams(key: Key): Params {
    return lruTree(lruInit(key, this.stateSize, this.dim, this.options))
  }

  initialState(batch: number): MixerState {
    return { re: zeros([batch, this.stateSize]), im: zeros([batch, this.stateSize]) }
  }

  override transitionSpectrum(params: Params = this.params): TransitionSpectrum {
    const e = this.eigenvalues(params)
    return {
      domain: 'discrete',
      re: Array.from(toFlat(e.re)),
      im: Array.from(toFlat(e.im)),
    }
  }

  /** The eigenvalues of the recurrence (real and imaginary parts, modulus below one). */
  eigenvalues(params: Params = this.params): { readonly re: Value; readonly im: Value } {
    return lruEigenvalues({ nuLog: params['nuLog'] as Value, thetaLog: params['thetaLog'] as Value })
  }

  protected mix(params: Params, x: Value, state: MixerState | undefined, _ctx: ApplyContext): MixerOutput {
    const s = state === undefined ? undefined : { re: state['re'] as Value, im: state['im'] as Value }
    const r = lruScan(params as unknown as LRUParams, x, s)
    return { output: r.y, state: { re: r.state.re, im: r.state.im } }
  }

  override extraRepr(): string {
    return `dim=${this.dim}, state=${this.stateSize}`
  }
}
