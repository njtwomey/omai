/**
 * Multirate filter banks that split a signal into subbands and put it back together: quadrature-mirror filters,
 * two-channel banks built from wavelets with checks of the perfect-reconstruction and power-complementarity conditions,
 * the uniform DFT analysis and synthesis bank, and the wavelet packet decomposition (pywt.WaveletPacket), which splits
 * both approximation and detail at every level.
 */

import { MdmlError } from '../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../numerics/differentiate/index.ts'
import { type ComplexValue, flip, mul, sliceAxis, type TensorLike, type Value } from '../tensor/index.ts'
import { constant, float, samples } from './_shared.ts'
import { istft, stft } from './time-frequency.ts'
import { type DwtOptions, dwt, idwt, type WaveletMode } from './wavelets/dwt.ts'
import { type DiscreteWavelet, type WaveletSpec, wavelet } from './wavelets/families.ts'

// Filter banks: quadrature-mirror filters and the perfect-reconstruction conditions of two-channel banks, uniform DFT
// filter banks (an analysis by windowed DFTs at a decimated hop and synthesis by weighted overlap-add), and wavelet
// packets, the full binary tree of two-channel splits.

/**
 * The quadrature-mirror highpass of a lowpass filter (scipy.signal.qmf, pywt.qmf): h1[k] = (-1)^k h0[N - k], so that
 * |H1(w)| = |H0(pi - w)|.
 *
 * @param h - Lowpass taps.
 * @returns The highpass taps.
 * @concept filter-banks
 * @example
 * ```ts
 * import { qmf } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(qmf([1, 2, 3])) // => [3, -2, 1]
 * ```
 */
export function qmf(h: TensorLike): Value {
  const v = flip(float(h), 0)
  const n = v.shape[0] ?? 0
  return mul(v, constant(Array.from({ length: n }, (_, k) => (k % 2 === 0 ? 1 : -1))))
}

/** The four filters of a two-channel filter bank: analysis h0 (lowpass), h1, synthesis g0, g1. */
export interface TwoChannelBank {
  readonly h0: readonly number[]
  readonly h1: readonly number[]
  readonly g0: readonly number[]
  readonly g1: readonly number[]
}

/**
 * A two-channel bank from a wavelet's filters (decomposition as analysis, reconstruction as synthesis).
 *
 * @param w - The wavelet.
 * @returns The filters.
 * @concept filter-banks
 * @example
 * ```ts
 * import { waveletBank } from 'mdml-engine/signal'
 * waveletBank('db2').h0.length // => 4
 * ```
 */
export function waveletBank(w: WaveletSpec): TwoChannelBank {
  const wv: DiscreteWavelet = wavelet(w)
  return { h0: Array.from(wv.decLo), h1: Array.from(wv.decHi), g0: Array.from(wv.recLo), g1: Array.from(wv.recHi) }
}

function polyMul(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(a.length + b.length - 1).fill(0)
  for (const [i, x] of a.entries()) for (const [j, y] of b.entries()) out[i + j] = (out[i + j] ?? 0) + x * y
  return out
}

function alternate(h: readonly number[]): number[] {
  return h.map((v, k) => (k % 2 === 0 ? v : -v))
}

/** The distortion and aliasing transfer functions of a two-channel bank, and its check for perfect reconstruction. */
export interface ReconstructionCheck {
  /** Coefficients of T(z) = (H0 G0 + H1 G1) / 2 in powers of z^-1. */
  readonly distortion: number[]
  /** Coefficients of A(z) = (H0(-z) G0 + H1(-z) G1) / 2. */
  readonly aliasing: number[]
  /** The delay d when T(z) = z^-d (or -1). */
  readonly delay: number
  /** Whether the aliasing vanishes and the distortion is a pure delay (to 1e-10). */
  readonly perfect: boolean
}

/**
 * The perfect-reconstruction conditions of a two-channel filter bank: no aliasing (H0(-z) G0(z) + H1(-z) G1(z) = 0)
 * and a pure-delay distortion (H0(z) G0(z) + H1(z) G1(z) = 2 z^-d).
 *
 * @param bank - The four filters.
 * @returns The distortion and aliasing polynomials, the delay and the verdict.
 * @concept perfect-reconstruction
 * @example
 * ```ts
 * import { checkReconstruction, waveletBank } from 'mdml-engine/signal'
 * checkReconstruction(waveletBank('bior2.2')).perfect // => true
 * ```
 */
export function checkReconstruction(bank: TwoChannelBank): ReconstructionCheck {
  const add = (a: number[], b: number[]): number[] =>
    Array.from({ length: Math.max(a.length, b.length) }, (_, i) => ((a[i] ?? 0) + (b[i] ?? 0)) / 2)
  const t = add(polyMul(bank.h0, bank.g0), polyMul(bank.h1, bank.g1))
  const a = add(polyMul(alternate(bank.h0), bank.g0), polyMul(alternate(bank.h1), bank.g1))
  const tol = 1e-10
  let delay = -1
  let ok = a.every((v) => Math.abs(v) < tol)
  const nonzero = t.map((v, i) => [v, i] as const).filter(([v]) => Math.abs(v) >= tol)
  if (nonzero.length === 1 && Math.abs(Math.abs(nonzero[0]?.[0] ?? 0) - 1) < tol) delay = nonzero[0]?.[1] ?? -1
  else ok = false
  return { distortion: t, aliasing: a, delay, perfect: ok && delay >= 0 }
}

/** The frequency response magnitude of a filter on n points of [0, pi]. */
function magnitude(h: readonly number[], n: number): number[] {
  return Array.from({ length: n }, (_, j) => {
    const w = (Math.PI * j) / (n - 1)
    let z: ComplexNumber = C.of(0)
    for (const [k, v] of h.entries()) z = C.add(z, C.of(v * Math.cos(w * k), -v * Math.sin(w * k)))
    return Math.hypot(z.re, z.im)
  })
}

/**
 * Whether a bank's analysis filters are power complementary (|H0|^2 + |H1|^2 = 2 on the unit circle), as orthogonal
 * (paraunitary) banks are; returns the largest deviation.
 *
 * @param bank - The filters.
 * @param n - Grid size (default 257).
 * @returns The maximum of ||H0|^2 + |H1|^2 - 2|.
 * @concept perfect-reconstruction
 * @example
 * ```ts
 * import { powerComplementarity, waveletBank } from 'mdml-engine/signal'
 * powerComplementarity(waveletBank('db4')) < 1e-12 // => true
 * ```
 */
export function powerComplementarity(bank: TwoChannelBank, n = 257): number {
  const a = magnitude(bank.h0, n)
  const b = magnitude(bank.h1, n)
  let worst = 0
  for (let j = 0; j < n; j++) worst = Math.max(worst, Math.abs((a[j] ?? 0) ** 2 + (b[j] ?? 0) ** 2 - 2))
  return worst
}

/** Options of the uniform DFT filter bank. */
export interface DftBankOptions {
  /** Number of channels M (the DFT length). */
  readonly channels: number
  /** Decimation: the hop between analysis frames (default M / 2). */
  readonly decimation?: number
  /** The prototype lowpass window of length M (default periodic Hann). */
  readonly prototype?: TensorLike
}

/**
 * Analysis by a uniform DFT filter bank: channel k is the signal modulated by e^{-2 pi i k n / M}, filtered by the
 * prototype and decimated, computed as windowed DFTs at the decimated hop. Differentiable.
 *
 * @param x - Real signal.
 * @param options - Channels, decimation and prototype.
 * @returns The complex subband signals `[M, frames]`.
 * @concept filter-banks
 * @example
 * ```ts
 * import { dftAnalysis } from 'mdml-engine/signal'
 * dftAnalysis(new Array(32).fill(1), { channels: 8 }).re.shape // => [8, 9]
 * ```
 */
export function dftAnalysis(x: TensorLike, options: DftBankOptions): ComplexValue {
  const m = options.channels
  const d = options.decimation ?? Math.floor(m / 2)
  return stft(x, {
    window: options.prototype ?? 'hann',
    nperseg: m,
    noverlap: m - d,
    returnOnesided: false,
    scaling: 'none',
  }).values
}

/**
 * Synthesis of a uniform DFT filter bank by weighted overlap-add, the inverse of {@link dftAnalysis} whenever the
 * prototype's shifts by the decimation do not vanish together (perfect reconstruction).
 *
 * @param subbands - Subband signals `[M, frames]`.
 * @param n - Length of the output.
 * @param options - Channels, decimation and prototype (as in the analysis).
 * @returns The reconstructed signal.
 * @concept perfect-reconstruction
 * @example
 * ```ts
 * import { dftAnalysis, dftSynthesis } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [1, 2, 0, -1, 3, 1, -2, 0, 1, 2, 0, 1, 3, -1, 0, 2]
 * toFlat(dftSynthesis(dftAnalysis(x, { channels: 8, decimation: 2 }), 16, { channels: 8, decimation: 2 })).slice(0, 3).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 2, 0]
 * ```
 */
export function dftSynthesis(subbands: ComplexValue, n: number, options: DftBankOptions): Value {
  const m = options.channels
  const d = options.decimation ?? Math.floor(m / 2)
  const y = istft(subbands, {
    window: options.prototype ?? 'hann',
    nperseg: m,
    noverlap: m - d,
    inputOnesided: false,
    scaling: 'none',
  }).x
  if ((y.shape[0] ?? 0) < n) throw new MdmlError('value', 'dftSynthesis: too few subband frames for this length')
  return sliceAxis(y, 0, 0, n)
}

/** A wavelet packet tree: coefficients of every node by path ('' the root, 'a', 'd', 'aa', 'ad', ...). */
export interface WaveletPacket {
  readonly nodes: ReadonlyMap<string, Value>
  readonly level: number
  readonly wavelet: DiscreteWavelet
  readonly mode: WaveletMode
}

/**
 * The wavelet packet decomposition (pywt.WaveletPacket): split both the approximation and the detail at every
 * level, to the given depth.
 *
 * @param x - Signal.
 * @param w - The wavelet.
 * @param options - Depth (default the maximum level) and extension mode (default 'symmetric').
 * @returns The tree of nodes.
 * @concept wavelet-packets
 * @example
 * ```ts
 * import { waveletPacket } from 'mdml-engine/signal'
 * waveletPacket(new Array(16).fill(1), 'haar', { level: 2 }).nodes.size // => 7
 * ```
 */
export function waveletPacket(
  x: TensorLike,
  w: WaveletSpec,
  options: { readonly level?: number; readonly mode?: WaveletMode } = {},
): WaveletPacket {
  const wv = wavelet(w)
  const data = float(x)
  const n = samples(data, 'waveletPacket').length
  const f = wv.decLo.length
  const level = options.level ?? (f < 2 || n < f - 1 ? 0 : Math.floor(Math.log2(n / (f - 1))))
  const mode = options.mode ?? 'symmetric'
  const nodes = new Map<string, Value>([['', data]])
  let frontier = ['']
  const o: DwtOptions = { mode }
  for (let l = 0; l < level; l++) {
    const next: string[] = []
    for (const p of frontier) {
      const c = dwt(nodes.get(p) as Value, wv, o)
      nodes.set(`${p}a`, c.cA)
      nodes.set(`${p}d`, c.cD)
      next.push(`${p}a`, `${p}d`)
    }
    frontier = next
  }
  return { nodes, level, wavelet: wv, mode }
}

/**
 * The paths of the nodes at a level, in natural (lexicographic) or frequency (Gray-code) order, as
 * pywt.WaveletPacket.get_level.
 *
 * @param level - The level.
 * @param order - 'natural' (default) or 'freq'.
 * @returns The paths.
 * @concept wavelet-packets
 * @example
 * ```ts
 * import { packetPaths } from 'mdml-engine/signal'
 * packetPaths(2, 'freq') // => ['aa', 'ad', 'dd', 'da']
 * ```
 */
export function packetPaths(level: number, order: 'natural' | 'freq' = 'natural'): string[] {
  if (level === 0) return ['']
  if (order === 'natural') {
    let paths = ['a', 'd']
    for (let l = 1; l < level; l++) paths = paths.flatMap((p) => [`${p}a`, `${p}d`])
    return paths
  }
  let gray = ['a', 'd']
  for (let l = 1; l < level; l++) gray = [...gray.map((p) => `a${p}`), ...[...gray].reverse().map((p) => `d${p}`)]
  return gray
}

/**
 * Reconstruct a signal from the nodes of one level of a wavelet packet (pywt.WaveletPacket.reconstruct), merging
 * siblings by idwt up to the root.
 *
 * @param packet - The tree (only the nodes at `level` are used).
 * @param level - The level to reconstruct from (default the tree's depth).
 * @returns The signal.
 * @concept wavelet-packets
 * @example
 * ```ts
 * import { packetReconstruct, waveletPacket } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(packetReconstruct(waveletPacket([3, 1, 4, 1, 5, 9, 2, 6], 'haar', { level: 2 }))).map((v) => Math.round(v * 1e12) / 1e12) // => [3, 1, 4, 1, 5, 9, 2, 6]
 * ```
 */
export function packetReconstruct(packet: WaveletPacket, level: number = packet.level): Value {
  const current = new Map<string, Value>()
  for (const p of packetPaths(level)) {
    const v = packet.nodes.get(p)
    if (v === undefined) throw new MdmlError('value', `packetReconstruct: missing node '${p}'`)
    current.set(p, v)
  }
  for (let l = level; l > 0; l--) {
    for (const parent of packetPaths(l - 1)) {
      const a = current.get(`${parent}a`) as Value
      const d = current.get(`${parent}d`) as Value
      let rec = idwt(a, d, packet.wavelet, { mode: packet.mode })
      // Crop to the node's own length, as pywt.
      const tl = packet.nodes.get(parent)?.shape[0] ?? 0
      if (tl > 0 && (rec.shape[0] ?? 0) > tl) rec = sliceAxis(rec, 0, 0, tl)
      current.set(parent, rec)
    }
  }
  return current.get('') as Value
}
