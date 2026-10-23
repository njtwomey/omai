import { type Json, WhiteboxError } from '../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'

/** A scalar function to minimise. */
export type ScalarFunction = (x: number) => number

/** A bracket a < b < c (or c < b < a) with f(b) below f(a) and f(c), and the evaluations it took. */
export interface Bracket {
  readonly xa: number
  readonly xb: number
  readonly xc: number
  readonly fa: number
  readonly fb: number
  readonly fc: number
  readonly nfev: number
}

/** Options of {@link bracket}. */
export interface BracketSearchOptions {
  /** First point (default 0). */
  readonly xa?: number
  /** Second point (default 1). */
  readonly xb?: number
  /** Largest parabolic extrapolation, as a multiple of the last interval (default 110). */
  readonly growLimit?: number
  /** Iteration cap (default 1000). */
  readonly maxIterations?: number
}

const GOLD = 1.618034

/**
 * Find a bracket of a minimum by moving downhill from two points with golden-ratio and parabolic extrapolation
 * (scipy.optimize.bracket, Numerical Recipes' mnbrak).
 *
 * @param f - The function.
 * @param options - Starting points, growth limit and iteration cap.
 * @returns The bracket.
 * @throws WhiteboxError with code `convergence` when no valid bracket is found.
 * @concept scalar-minimisation
 * @example
 * ```ts
 * import { bracket } from 'whitebox-engine/optim'
 * const b = bracket((x) => (x - 3) ** 2)
 * b.fb < b.fa && b.fb < b.fc // => true
 * ```
 */
export function bracket(f: ScalarFunction, options: BracketSearchOptions = {}): Bracket {
  const growLimit = options.growLimit ?? 110
  const maxiter = options.maxIterations ?? 1000
  let xa = options.xa ?? 0
  let xb = options.xb ?? 1
  let fa = f(xa)
  let fb = f(xb)
  if (fa < fb) {
    ;[xa, xb] = [xb, xa]
    ;[fa, fb] = [fb, fa]
  }
  let xc = xb + GOLD * (xb - xa)
  let fc = f(xc)
  let nfev = 3
  let iter = 0
  while (fc < fb) {
    const tmp1 = (xb - xa) * (fb - fc)
    const tmp2 = (xb - xc) * (fb - fa)
    const val = tmp2 - tmp1
    const denom = Math.abs(val) < 1e-21 ? 2e-21 : 2 * val
    let w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
    const wlim = xb + growLimit * (xc - xb)
    if (iter > maxiter) throw new WhiteboxError('convergence', 'bracket: no valid bracket within the iteration limit')
    iter += 1
    let fw: number
    if ((w - xc) * (xb - w) > 0) {
      fw = f(w)
      nfev += 1
      if (fw < fc) {
        xa = xb
        xb = w
        fa = fb
        fb = fw
        break
      }
      if (fw > fb) {
        xc = w
        fc = fw
        break
      }
      w = xc + GOLD * (xc - xb)
      fw = f(w)
      nfev += 1
    } else if ((w - wlim) * (wlim - xc) >= 0) {
      w = wlim
      fw = f(w)
      nfev += 1
    } else if ((w - wlim) * (xc - w) > 0) {
      fw = f(w)
      nfev += 1
      if (fw < fc) {
        xb = xc
        xc = w
        w = xc + GOLD * (xc - xb)
        fb = fc
        fc = fw
        fw = f(w)
        nfev += 1
      }
    } else {
      w = xc + GOLD * (xc - xb)
      fw = f(w)
      nfev += 1
    }
    xa = xb
    xb = xc
    xc = w
    fa = fb
    fb = fc
    fc = fw
  }
  const valid =
    ((fb < fc && fb <= fa) || (fb < fa && fb <= fc)) &&
    ((xa < xb && xb < xc) || (xc < xb && xb < xa)) &&
    Number.isFinite(xa) &&
    Number.isFinite(xb) &&
    Number.isFinite(xc)
  if (!valid)
    throw new WhiteboxError('convergence', 'bracket: the search ended without a valid bracket', { xa, xb, xc })
  return { xa, xb, xc, fa, fb, fc, nfev }
}

/** Result of a scalar minimisation (scipy's OptimizeResult for minimize_scalar). */
export interface ScalarResult {
  readonly x: number
  readonly fun: number
  readonly iterations: number
  readonly nfev: number
  readonly converged: boolean
  readonly message: string
}

/** The state of the scalar minimisers: the current best point, the interval of uncertainty and method data. */
export interface ScalarState {
  readonly x: number
  readonly fx: number
  /** The interval of uncertainty [a, b]. */
  readonly a: number
  readonly b: number
  readonly nfev: number
  /** The last step kind: initial, golden or parabolic. */
  readonly kind: 'initial' | 'golden' | 'parabolic'
  readonly done: boolean
  readonly internal: Readonly<Record<string, number>>
}

/** Options shared by the scalar minimisers. */
export interface ScalarOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** A bracket: two points to start {@link bracket} from, or three points a < b < c with f(b) below both ends. */
  readonly bracket?: readonly [number, number] | readonly [number, number, number]
  /** The interval of {@link fminbound} (required there). */
  readonly bounds?: readonly [number, number]
  /** The tolerance (Brent: relative xtol 1.48e-8; golden: sqrt(eps); bounded: absolute xatol 1e-5). */
  readonly tol?: number
  /** Iteration cap (Brent 500, golden 5000, bounded 500 evaluations). */
  readonly maxIterations?: number
}

/** A scalar minimisation method as a function value, for {@link minimizeScalar}. */
export type ScalarMethod = (f: ScalarFunction, options?: ScalarOptions) => ScalarResult

const SCHEMA = channels({
  metrics: {
    loss: { description: 'f at the best point', direction: 'lower' },
    x: { description: 'best point' },
    width: { description: 'width of the interval of uncertainty' },
  },
  messages: { step: { description: 'kind of step: golden section or parabolic interpolation' } },
})

const observe = (s: ScalarState) => ({
  metrics: { loss: s.fx, x: s.x, width: Math.abs(s.b - s.a) },
  messages: [{ kind: 'step', payload: s.kind }],
})

function startBracket(f: ScalarFunction, options: ScalarOptions): Bracket {
  const br = options.bracket
  if (br === undefined) return bracket(f)
  if (br.length === 2) return bracket(f, { xa: br[0], xb: br[1] })
  let [xa, xb, xc] = br
  if (xa > xc) [xa, xc] = [xc, xa]
  if (!(xa < xb && xb < xc)) throw new WhiteboxError('value', 'bracket must satisfy xa < xb < xc')
  const fa = f(xa)
  const fb = f(xb)
  const fc = f(xc)
  if (!(fb < fa && fb < fc)) throw new WhiteboxError('value', 'bracket must satisfy f(xb) < f(xa) and f(xb) < f(xc)')
  return { xa, xb, xc, fa, fb, fc, nfev: 3 }
}

function finish(alg: Algorithm<ScalarState>, options: ScalarOptions, steps: number): ScalarResult {
  const { bracket: _b, bounds: _bo, tol: _t, maxIterations: _m, ...run } = options
  const r = runAlgorithm(alg, { ...run, maxSteps: steps })
  const s = r.state
  const ok = s.done && !Number.isNaN(s.x) && !Number.isNaN(s.fx)
  return {
    x: s.x,
    fun: s.fx,
    iterations: r.steps,
    nfev: s.nfev,
    converged: ok,
    message: ok ? 'solution found' : Number.isNaN(s.fx) ? 'NaN encountered' : 'maximum number of iterations reached',
  }
}

/**
 * Brent's method for minimisation as a traced algorithm: golden-section steps with parabolic interpolation through
 * the three best points (scipy.optimize.brent).
 *
 * @param f - The function.
 * @param options - Bracket and relative tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { brentAlgorithm } from 'whitebox-engine/optim'
 * brentAlgorithm((x) => x * x).name // => 'brent'
 * ```
 */
export function brentAlgorithm(f: ScalarFunction, options: ScalarOptions = {}): Algorithm<ScalarState> {
  const tol = options.tol ?? 1.48e-8
  const mintol = 1e-11
  const cg = 0.381966
  const stopped = (s: { x: number; a: number; b: number }): boolean => {
    const tol1 = tol * Math.abs(s.x) + mintol
    return Math.abs(s.x - 0.5 * (s.a + s.b)) < 2 * tol1 - 0.5 * (s.b - s.a)
  }
  return {
    name: 'brent',
    config: { tol } as Json,
    schema: SCHEMA,
    init: () => {
      const br = startBracket(f, options)
      const [a, b] = br.xa < br.xc ? [br.xa, br.xc] : [br.xc, br.xa]
      const s = { x: br.xb, fx: br.fb, a, b }
      return {
        ...s,
        nfev: br.nfev,
        kind: 'initial',
        done: stopped(s),
        internal: { w: br.xb, v: br.xb, fw: br.fb, fv: br.fb, deltax: 0, rat: 0 },
      }
    },
    step: (s) => {
      const { x, fx } = s
      let { a, b } = s
      let { w, v, fw, fv, deltax, rat } = s.internal as Record<'w' | 'v' | 'fw' | 'fv' | 'deltax' | 'rat', number>
      const tol1 = tol * Math.abs(x) + mintol
      const tol2 = 2 * tol1
      const xmid = 0.5 * (a + b)
      let kind: ScalarState['kind'] = 'golden'
      if (Math.abs(deltax) <= tol1) {
        deltax = x >= xmid ? a - x : b - x
        rat = cg * deltax
      } else {
        const tmp1 = (x - w) * (fx - fv)
        let tmp2 = (x - v) * (fx - fw)
        let p = (x - v) * tmp2 - (x - w) * tmp1
        tmp2 = 2 * (tmp2 - tmp1)
        if (tmp2 > 0) p = -p
        tmp2 = Math.abs(tmp2)
        const dxTemp = deltax
        deltax = rat
        if (p > tmp2 * (a - x) && p < tmp2 * (b - x) && Math.abs(p) < Math.abs(0.5 * tmp2 * dxTemp)) {
          rat = p / tmp2
          kind = 'parabolic'
          const u = x + rat
          if (u - a < tol2 || b - u < tol2) rat = xmid - x >= 0 ? tol1 : -tol1
        } else {
          deltax = x >= xmid ? a - x : b - x
          rat = cg * deltax
        }
      }
      const u = Math.abs(rat) < tol1 ? (rat >= 0 ? x + tol1 : x - tol1) : x + rat
      const fu = f(u)
      let nx = x
      let nfx = fx
      if (fu > fx) {
        if (u < x) a = u
        else b = u
        if (fu <= fw || w === x) {
          v = w
          w = u
          fv = fw
          fw = fu
        } else if (fu <= fv || v === x || v === w) {
          v = u
          fv = fu
        }
      } else {
        if (u >= x) a = x
        else b = x
        v = w
        w = x
        nx = u
        fv = fw
        fw = fx
        nfx = fu
      }
      return {
        x: nx,
        fx: nfx,
        a,
        b,
        nfev: s.nfev + 1,
        kind,
        done: stopped({ x: nx, a, b }),
        internal: { w, v, fw, fv, deltax, rat },
      }
    },
    done: (s) => (s.done ? 'converged' : Number.isNaN(s.fx) ? 'diverged' : undefined),
    observe,
  }
}

/**
 * Golden-section search as a traced algorithm: shrink the bracket by the golden ratio each step
 * (scipy.optimize.golden).
 *
 * @param f - The function.
 * @param options - Bracket and tolerance (relative to |x1| + |x2|).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { goldenAlgorithm } from 'whitebox-engine/optim'
 * goldenAlgorithm((x) => x * x).name // => 'golden'
 * ```
 */
export function goldenAlgorithm(f: ScalarFunction, options: ScalarOptions = {}): Algorithm<ScalarState> {
  const tol = options.tol ?? Math.sqrt(Number.EPSILON)
  const gR = 0.61803399
  const gC = 1 - gR
  const best = (i: Record<string, number>): { x: number; fx: number } =>
    (i['f1'] ?? 0) < (i['f2'] ?? 0) ? { x: i['x1'] ?? 0, fx: i['f1'] ?? 0 } : { x: i['x2'] ?? 0, fx: i['f2'] ?? 0 }
  const stopped = (i: Record<string, number>): boolean =>
    Math.abs((i['x3'] ?? 0) - (i['x0'] ?? 0)) <= tol * (Math.abs(i['x1'] ?? 0) + Math.abs(i['x2'] ?? 0))
  const state = (internal: Record<string, number>, nfev: number, kind: ScalarState['kind']): ScalarState => {
    const b = best(internal)
    const lo = Math.min(internal['x0'] ?? 0, internal['x3'] ?? 0)
    const hi = Math.max(internal['x0'] ?? 0, internal['x3'] ?? 0)
    return { ...b, a: lo, b: hi, nfev, kind, done: stopped(internal), internal }
  }
  return {
    name: 'golden',
    config: { tol } as Json,
    schema: SCHEMA,
    init: () => {
      const br = startBracket(f, options)
      const { xa, xb, xc } = br
      const [x1, x2] = Math.abs(xc - xb) > Math.abs(xb - xa) ? [xb, xb + gC * (xc - xb)] : [xb - gC * (xb - xa), xb]
      return state({ x0: xa, x1, x2, x3: xc, f1: f(x1), f2: f(x2) }, br.nfev + 2, 'initial')
    },
    step: (s) => {
      const i = s.internal as Record<'x0' | 'x1' | 'x2' | 'x3' | 'f1' | 'f2', number>
      let { x0, x1, x2, x3, f1, f2 } = i
      if (f2 < f1) {
        x0 = x1
        x1 = x2
        x2 = gR * x1 + gC * x3
        f1 = f2
        f2 = f(x2)
      } else {
        x3 = x2
        x2 = x1
        x1 = gR * x2 + gC * x0
        f2 = f1
        f1 = f(x1)
      }
      return state({ x0, x1, x2, x3, f1, f2 }, s.nfev + 1, 'golden')
    },
    done: (s) => (s.done ? 'converged' : Number.isNaN(s.fx) ? 'diverged' : undefined),
    observe,
  }
}

/**
 * Brent's bounded minimisation on an interval as a traced algorithm (scipy.optimize.fminbound, minimize_scalar with
 * method 'bounded').
 *
 * @param f - The function.
 * @param options - `bounds` (required) and the absolute tolerance xatol (default 1e-5).
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` when the bounds are missing, infinite or reversed.
 * @example
 * ```ts
 * import { fminboundAlgorithm } from 'whitebox-engine/optim'
 * fminboundAlgorithm((x) => x * x, { bounds: [-1, 2] }).name // => 'fminbound'
 * ```
 */
export function fminboundAlgorithm(f: ScalarFunction, options: ScalarOptions = {}): Algorithm<ScalarState> {
  const bounds = options.bounds
  if (bounds === undefined || !Number.isFinite(bounds[0]) || !Number.isFinite(bounds[1]) || bounds[0] > bounds[1])
    throw new WhiteboxError('value', 'fminbound needs finite bounds [lo, hi] with lo <= hi')
  const xatol = options.tol ?? 1e-5
  const sqrtEps = Math.sqrt(2.2e-16)
  const goldenMean = 0.5 * (3 - Math.sqrt(5))
  const stopped = (x: number, a: number, b: number): boolean => {
    const tol1 = sqrtEps * Math.abs(x) + xatol / 3
    return !(Math.abs(x - 0.5 * (a + b)) > 2 * tol1 - 0.5 * (b - a))
  }
  return {
    name: 'fminbound',
    config: { bounds: [bounds[0], bounds[1]], xatol } as Json,
    schema: SCHEMA,
    init: () => {
      const [a, b] = bounds
      const xf = a + goldenMean * (b - a)
      const fx = f(xf)
      return {
        x: xf,
        fx,
        a,
        b,
        nfev: 1,
        kind: 'initial',
        done: stopped(xf, a, b),
        internal: { nfc: xf, fulc: xf, fnfc: fx, ffulc: fx, rat: 0, e: 0 },
      }
    },
    step: (s) => {
      const xf = s.x
      const fx = s.fx
      let { a, b } = s
      let { nfc, fulc, fnfc, ffulc, rat, e } = s.internal as Record<
        'nfc' | 'fulc' | 'fnfc' | 'ffulc' | 'rat' | 'e',
        number
      >
      const xm = 0.5 * (a + b)
      const tol1 = sqrtEps * Math.abs(xf) + xatol / 3
      const tol2 = 2 * tol1
      let golden = true
      let kind: ScalarState['kind'] = 'golden'
      if (Math.abs(e) > tol1) {
        golden = false
        let r = (xf - nfc) * (fx - ffulc)
        let q = (xf - fulc) * (fx - fnfc)
        let p = (xf - fulc) * q - (xf - nfc) * r
        q = 2 * (q - r)
        if (q > 0) p = -p
        q = Math.abs(q)
        r = e
        e = rat
        if (Math.abs(p) < Math.abs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf)) {
          rat = p / q
          kind = 'parabolic'
          const x = xf + rat
          if (x - a < tol2 || b - x < tol2) rat = tol1 * (Math.sign(xm - xf) + (xm - xf === 0 ? 1 : 0))
        } else golden = true
      }
      if (golden) {
        e = xf >= xm ? a - xf : b - xf
        rat = goldenMean * e
      }
      const si = Math.sign(rat) + (rat === 0 ? 1 : 0)
      const x = xf + si * Math.max(Math.abs(rat), tol1)
      const fu = f(x)
      let nx = xf
      let nfx = fx
      if (fu <= fx) {
        if (x >= xf) a = xf
        else b = xf
        fulc = nfc
        ffulc = fnfc
        nfc = xf
        fnfc = fx
        nx = x
        nfx = fu
      } else {
        if (x < xf) a = x
        else b = x
        if (fu <= fnfc || nfc === xf) {
          fulc = nfc
          ffulc = fnfc
          nfc = x
          fnfc = fu
        } else if (fu <= ffulc || fulc === xf || fulc === nfc) {
          fulc = x
          ffulc = fu
        }
      }
      return {
        x: nx,
        fx: nfx,
        a,
        b,
        nfev: s.nfev + 1,
        kind,
        done: stopped(nx, a, b) || Number.isNaN(fu),
        internal: { nfc, fulc, fnfc, ffulc, rat, e },
      }
    },
    done: (s) => (Number.isNaN(s.fx) ? 'diverged' : s.done ? 'converged' : undefined),
    observe,
  }
}

/**
 * Minimise a function of one variable by Brent's method on a bracket (scipy.optimize.minimize_scalar with method
 * 'brent').
 *
 * @param f - The function.
 * @param options - Bracket, relative tolerance (default 1.48e-8), iteration cap (500) and run options.
 * @returns The result.
 * @concept scalar-minimisation
 * @example
 * ```ts
 * import { brent } from 'whitebox-engine/optim'
 * Math.round(brent((x) => (x - 2) ** 2 + 1).x * 1e8) / 1e8 // => 2
 * ```
 */
export function brent(f: ScalarFunction, options: ScalarOptions = {}): ScalarResult {
  return finish(brentAlgorithm(f, options), options, options.maxIterations ?? 500)
}

/**
 * Minimise a function of one variable by golden-section search (scipy.optimize.minimize_scalar with method 'golden').
 *
 * @param f - The function.
 * @param options - Bracket, tolerance (default sqrt(eps)), iteration cap (5000) and run options.
 * @returns The result.
 * @concept scalar-minimisation
 * @example
 * ```ts
 * import { golden } from 'whitebox-engine/optim'
 * Math.round(golden((x) => (x - 2) ** 2 + 1).x * 1e7) / 1e7 // => 2
 * ```
 */
export function golden(f: ScalarFunction, options: ScalarOptions = {}): ScalarResult {
  return finish(goldenAlgorithm(f, options), options, options.maxIterations ?? 5000)
}

/**
 * Minimise a function on an interval by Brent's bounded method (scipy.optimize.fminbound).
 *
 * @param f - The function.
 * @param options - `bounds`, absolute tolerance (default 1e-5), evaluation cap (500) and run options.
 * @returns The result.
 * @concept scalar-minimisation
 * @example
 * ```ts
 * import { fminbound } from 'whitebox-engine/optim'
 * Math.round(fminbound((x) => (x - 2) ** 2, { bounds: [3, 5] }).x * 1e4) / 1e4 // => 3
 * ```
 */
export function fminbound(f: ScalarFunction, options: ScalarOptions = {}): ScalarResult {
  // scipy caps function evaluations (one at init, one per step).
  return finish(fminboundAlgorithm(f, options), options, Math.max(0, (options.maxIterations ?? 500) - 1))
}

/**
 * Minimise a function of one variable (scipy.optimize.minimize_scalar). The method is a function value: `brent`
 * (default), `golden` or `fminbound` (the default when `bounds` is given).
 *
 * @param f - The function.
 * @param options - The method, bracket or bounds, tolerance, iteration cap and run options.
 * @returns The result.
 * @concept scalar-minimisation
 * @example
 * ```ts
 * import { golden, minimizeScalar } from 'whitebox-engine/optim'
 * Math.round(minimizeScalar((x) => Math.cosh(x - 1), { method: golden }).x * 1e6) / 1e6 // => 1
 * Math.round(minimizeScalar((x) => -x, { bounds: [0, 2] }).x * 1e4) / 1e4 // => 2
 * ```
 */
export function minimizeScalar(
  f: ScalarFunction,
  options: ScalarOptions & { readonly method?: ScalarMethod } = {},
): ScalarResult {
  const { method, ...rest } = options
  const m = method ?? (options.bounds === undefined ? brent : fminbound)
  return m(f, rest)
}
