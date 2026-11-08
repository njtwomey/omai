/**
 * `getWindow`, the window lookup of scipy.signal.get_window: any window of this folder by name, by name with parameters
 * (`['kaiser', 8]`, `['tukey', 0.25]`) or by a Kaiser beta, so analyses can take a window as a simple specification.
 */

import { OmaiError } from '../../base/index.ts'
import type { Tensor } from '../../tensor/index.ts'
import {
  blackman,
  blackmanHarris,
  boxcar,
  flattop,
  generalCosine,
  generalHamming,
  hamming,
  hann,
  nuttall,
} from './cosine.ts'
import { dpss } from './dpss.ts'
import {
  barthann,
  bartlett,
  bohman,
  cosine,
  exponential,
  gaussian,
  generalGaussian,
  kaiser,
  kaiserBesselDerived,
  lanczos,
  parzen,
  triang,
  tukey,
} from './shapes.ts'

/** A window by name, a name with parameters, or a Kaiser beta (as scipy.signal.get_window). */
export type WindowSpec = string | readonly [string, ...number[]] | number

const ALIASES: Readonly<Record<string, string>> = {
  barthann: 'barthann',
  brthan: 'barthann',
  bth: 'barthann',
  bartlett: 'bartlett',
  bart: 'bartlett',
  brt: 'bartlett',
  blackman: 'blackman',
  black: 'blackman',
  blk: 'blackman',
  blackmanharris: 'blackmanharris',
  blackharr: 'blackmanharris',
  bkh: 'blackmanharris',
  bohman: 'bohman',
  bman: 'bohman',
  bmn: 'bohman',
  boxcar: 'boxcar',
  box: 'boxcar',
  ones: 'boxcar',
  rect: 'boxcar',
  rectangular: 'boxcar',
  cosine: 'cosine',
  halfcosine: 'cosine',
  dpss: 'dpss',
  exponential: 'exponential',
  poisson: 'exponential',
  flattop: 'flattop',
  flat: 'flattop',
  flt: 'flattop',
  gaussian: 'gaussian',
  gauss: 'gaussian',
  gss: 'gaussian',
  'general cosine': 'general_cosine',
  general_cosine: 'general_cosine',
  'general gaussian': 'general_gaussian',
  general_gaussian: 'general_gaussian',
  'general gauss': 'general_gaussian',
  general_gauss: 'general_gaussian',
  ggs: 'general_gaussian',
  'general hamming': 'general_hamming',
  general_hamming: 'general_hamming',
  hamming: 'hamming',
  hamm: 'hamming',
  ham: 'hamming',
  hann: 'hann',
  han: 'hann',
  kaiser: 'kaiser',
  ksr: 'kaiser',
  'kaiser bessel derived': 'kbd',
  kaiser_bessel_derived: 'kbd',
  kbd: 'kbd',
  lanczos: 'lanczos',
  sinc: 'lanczos',
  nuttall: 'nuttall',
  nutl: 'nuttall',
  nut: 'nuttall',
  parzen: 'parzen',
  parz: 'parzen',
  par: 'parzen',
  triangle: 'triang',
  triang: 'triang',
  tri: 'triang',
  tukey: 'tukey',
  tuk: 'tukey',
}

/**
 * A window by specification (scipy.signal.get_window): a name (`'hann'`), a name with parameters (`['kaiser', 8]`,
 * `['tukey', 0.25]`, `['gaussian', 3]`, `['dpss', 2.5]`, `['general_cosine', a0, a1, ...]`), or a number (a Kaiser
 * beta). Periodic by default (`fftbins`), for spectral analysis; a `_symmetric` or `_periodic` suffix overrides.
 *
 * @param window - The specification.
 * @param n - Length.
 * @param fftbins - Periodic (default true) or symmetric.
 * @returns The window `[n]`.
 * @throws OmaiError with code `value` for an unknown name or missing parameters.
 * @concept window-functions
 * @example
 * ```ts
 * import { getWindow } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(getWindow('hann', 4)) // => [0, 0.5, 1, 0.5]
 * toArray(getWindow(['tukey', 0], 3)) // => [1, 1, 1]
 * ```
 */
export function getWindow(window: WindowSpec, n: number, fftbins = true): Tensor {
  if (!Number.isInteger(n) || n < 1) throw new OmaiError('value', `getWindow: Nx must be a positive integer, got ${n}`)
  if (typeof window === 'number') return kaiser(n, window, { sym: !fftbins })
  let name = typeof window === 'string' ? window : window[0]
  const args = typeof window === 'string' ? [] : (window.slice(1) as number[])
  let sym = !fftbins
  if (name.endsWith('_symmetric')) {
    sym = true
    name = name.slice(0, -10)
  } else if (name.endsWith('_periodic')) {
    sym = false
    name = name.slice(0, -9)
  }
  const key = ALIASES[name]
  if (key === undefined) throw new OmaiError('value', `getWindow: unknown window '${name}'`)
  const o = { sym }
  const need = (k: number): number => {
    const v = args[k]
    if (v === undefined) throw new OmaiError('value', `getWindow: '${name}' needs parameter ${k + 1}`)
    return v
  }
  const none = (): void => {
    if (args.length > 0) throw new OmaiError('value', `getWindow: '${name}' takes no parameters`)
  }
  switch (key) {
    case 'barthann':
      none()
      return barthann(n, o)
    case 'bartlett':
      none()
      return bartlett(n, o)
    case 'blackman':
      none()
      return blackman(n, o)
    case 'blackmanharris':
      none()
      return blackmanHarris(n, o)
    case 'bohman':
      none()
      return bohman(n, o)
    case 'boxcar':
      none()
      return boxcar(n, o)
    case 'cosine':
      none()
      return cosine(n, o)
    case 'dpss':
      return dpss(n, need(0), o)
    case 'exponential':
      return args.length === 0 ? exponential(n, o) : exponential(n, { ...o, center: need(0), tau: args[1] ?? 1 })
    case 'flattop':
      none()
      return flattop(n, o)
    case 'gaussian':
      return gaussian(n, need(0), o)
    case 'general_cosine':
      need(0)
      return generalCosine(n, args, o)
    case 'general_gaussian':
      return generalGaussian(n, need(0), need(1), o)
    case 'general_hamming':
      return generalHamming(n, need(0), o)
    case 'hamming':
      none()
      return hamming(n, o)
    case 'hann':
      none()
      return hann(n, o)
    case 'kaiser':
      return kaiser(n, need(0), o)
    case 'kbd':
      if (!sym) throw new OmaiError('value', 'getWindow: Kaiser-Bessel-derived windows are only symmetric')
      return kaiserBesselDerived(n, need(0))
    case 'lanczos':
      none()
      return lanczos(n, o)
    case 'nuttall':
      none()
      return nuttall(n, o)
    case 'parzen':
      none()
      return parzen(n, o)
    case 'triang':
      none()
      return triang(n, o)
    default:
      return tukey(n, args[0] ?? 0.5, o)
  }
}
