/**
 * Interval line searches used by scipy's quasi-Newton and conjugate gradient methods: More-Thuente (MINPACK-2's
 * `dcsrch`), Hager-Zhang (CG_DESCENT's approximate Wolfe search), and `wolfe12`, scipy's More-Thuente with the strong
 * Wolfe zoom as fallback.
 */

import { runAlgorithm } from '../trace/index.ts'
import { checkConstants, firstStep, probe } from './_line-search.ts'
import { type ExtraCondition, type LineSearch, type LineSearchWarning, strongWolfe } from './line-search.ts'
import { hagerZhangSearch } from './line-search-algorithms.ts'

/** Options of {@link moreThuente}. */
export interface MoreThuenteOptions {
  /** Sufficient-decrease constant (MINPACK's ftol, default 1e-4). */
  readonly c1?: number
  /** Curvature constant (MINPACK's gtol, default 0.9). */
  readonly c2?: number
  /** Relative width of an acceptable bracket (default 1e-14). */
  readonly xtol?: number
  /** Smallest step (default 1e-8). */
  readonly alphaMin?: number
  /** Largest step (default 50). */
  readonly alphaMax?: number
  /** Most evaluations (default 100). */
  readonly maxEvaluations?: number
}

/** The safeguarded step of More and Thuente (MINPACK-2 dcstep). */
function dcstep(
  stx0: number,
  fx0: number,
  dx0: number,
  sty0: number,
  fy0: number,
  dy0: number,
  stp: number,
  fp: number,
  dp: number,
  brackt0: boolean,
  stpmin: number,
  stpmax: number,
): [number, number, number, number, number, number, number, boolean] {
  let stx = stx0
  let fx = fx0
  let dx = dx0
  let sty = sty0
  let fy = fy0
  let dy = dy0
  let brackt = brackt0
  const sgnd = Math.sign(dp) * Math.sign(dx)
  let stpf: number
  if (fp > fx) {
    const theta = (3 * (fx - fp)) / (stp - stx) + dx + dp
    const s = Math.max(Math.abs(theta), Math.abs(dx), Math.abs(dp))
    let gamma = s * Math.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
    if (stp < stx) gamma = -gamma
    const pp = gamma - dx + theta
    const q = gamma - dx + gamma + dp
    const r = pp / q
    const stpc = stx + r * (stp - stx)
    const stpq = stx + (dx / ((fx - fp) / (stp - stx) + dx) / 2) * (stp - stx)
    stpf = Math.abs(stpc - stx) <= Math.abs(stpq - stx) ? stpc : stpc + (stpq - stpc) / 2
    brackt = true
  } else if (sgnd < 0) {
    const theta = (3 * (fx - fp)) / (stp - stx) + dx + dp
    const s = Math.max(Math.abs(theta), Math.abs(dx), Math.abs(dp))
    let gamma = s * Math.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
    if (stp > stx) gamma = -gamma
    const pp = gamma - dp + theta
    const q = gamma - dp + gamma + dx
    const r = pp / q
    const stpc = stp + r * (stx - stp)
    const stpq = stp + (dp / (dp - dx)) * (stx - stp)
    stpf = Math.abs(stpc - stp) > Math.abs(stpq - stp) ? stpc : stpq
    brackt = true
  } else if (Math.abs(dp) < Math.abs(dx)) {
    const theta = (3 * (fx - fp)) / (stp - stx) + dx + dp
    const s = Math.max(Math.abs(theta), Math.abs(dx), Math.abs(dp))
    let gamma = s * Math.sqrt(Math.max(0, (theta / s) ** 2 - (dx / s) * (dp / s)))
    if (stp > stx) gamma = -gamma
    const pp = gamma - dp + theta
    const q = gamma + (dx - dp) + gamma
    const r = pp / q
    const stpc = r < 0 && gamma !== 0 ? stp + r * (stx - stp) : stp > stx ? stpmax : stpmin
    const stpq = stp + (dp / (dp - dx)) * (stx - stp)
    if (brackt) {
      stpf = Math.abs(stpc - stp) < Math.abs(stpq - stp) ? stpc : stpq
      stpf = stp > stx ? Math.min(stp + 0.66 * (sty - stp), stpf) : Math.max(stp + 0.66 * (sty - stp), stpf)
    } else {
      stpf = Math.abs(stpc - stp) > Math.abs(stpq - stp) ? stpc : stpq
      stpf = Math.min(Math.max(stpf, stpmin), stpmax)
    }
  } else if (brackt) {
    const theta = (3 * (fp - fy)) / (sty - stp) + dy + dp
    const s = Math.max(Math.abs(theta), Math.abs(dy), Math.abs(dp))
    let gamma = s * Math.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
    if (stp > sty) gamma = -gamma
    const pp = gamma - dp + theta
    const q = gamma - dp + gamma + dy
    stpf = stp + (pp / q) * (sty - stp)
  } else stpf = stp > stx ? stpmax : stpmin
  if (fp > fx) {
    sty = stp
    fy = fp
    dy = dp
  } else {
    if (sgnd < 0) {
      sty = stx
      fy = fx
      dy = dx
    }
    stx = stp
    fx = fp
    dx = dp
  }
  return [stx, fx, dx, sty, fy, dy, stpf, brackt]
}

/** The messages of More-Thuente's warnings, for people: a solver reads the typed `warning`, not these. */
const WARNING_MESSAGES: Readonly<Record<LineSearchWarning, string>> = {
  rounding: 'rounding errors prevent progress',
  xtol: 'xtol test satisfied',
  alphaMax: 'step at alphaMax',
  alphaMin: 'step at alphaMin',
}

/**
 * The More-Thuente line search (MINPACK-2 dcsrch, scipy's scalar_search_wolfe1 and the default of scipy's BFGS and
 * conjugate gradients): safeguarded cubic and quadratic steps that keep an interval of uncertainty and find a step
 * satisfying the strong Wolfe conditions, using a modified function until a sufficient decrease is found.
 *
 * @param options - c1 (ftol), c2 (gtol), xtol, step bounds and the evaluation cap.
 * @returns The line search.
 * @concept line-search
 * @reference J. J. More and D. J. Thuente, Line search algorithms with guaranteed sufficient decrease, ACM TOMS 20 (1994).
 * @example
 * ```ts
 * import { moreThuente } from 'omai-engine/optim'
 * // phi(alpha) = (alpha - 0.25)^2: the first trial (alpha = 1) overshoots, interpolation finds the minimum
 * const phi = { at: (a: number) => ({ value: (a - 0.25) ** 2, slope: 2 * (a - 0.25) }) }
 * const r = moreThuente()(phi, { value: 0.0625, slope: -0.5 })
 * [r.success, r.alpha, r.trials.length] // => [true, 0.25, 2]
 * ```
 */
export function moreThuente(options: MoreThuenteOptions = {}): LineSearch {
  const ftol = options.c1 ?? 1e-4
  const gtol = options.c2 ?? 0.9
  const xtol = options.xtol ?? 1e-14
  const stpmin = options.alphaMin ?? 1e-8
  const stpmax = options.alphaMax ?? 50
  const maxiter = options.maxEvaluations ?? 100
  checkConstants(ftol, gtol)
  return (phi, start) => {
    const p = probe(phi)
    const finit = start.value
    const ginit = start.slope
    let stp = firstStep(start)
    if (stp < stpmin || stp > stpmax || !(ginit < 0)) {
      return p.finish(
        undefined,
        finit,
        Number.NaN,
        !(ginit < 0) ? 'not a descent direction' : 'first step out of bounds',
      )
    }
    let brackt = false
    let stage = 1
    const gtest = ftol * ginit
    let width = stpmax - stpmin
    let width1 = width / 0.5
    let stx = 0
    let fx = finit
    let gx = ginit
    let sty = 0
    let fy = finit
    let gy = ginit
    let stmin = 0
    let stmax = stp + 4 * stp
    for (let i = 0; i < maxiter; i++) {
      const pt = p.at(stp)
      const f = pt.value
      const g = pt.slope
      const ftest = finit + stp * gtest
      if (stage === 1 && f <= ftest && g >= 0) stage = 2
      let warning: LineSearchWarning | undefined
      if (brackt && (stp <= stmin || stp >= stmax)) warning = 'rounding'
      if (brackt && stmax - stmin <= xtol * stmax) warning = 'xtol'
      if (stp === stpmax && f <= ftest && g <= gtest) warning = 'alphaMax'
      if (stp === stpmin && (f > ftest || g >= gtest)) warning = 'alphaMin'
      if (f <= ftest && Math.abs(g) <= gtol * -ginit) return p.finish(stp, f, g, 'strong Wolfe conditions satisfied')
      // A warning (as scipy, a failure) still reports the last trial step, which L-BFGS-B accepts.
      if (warning !== undefined) return { ...p.finish(stp, f, g, WARNING_MESSAGES[warning]), success: false, warning }
      if (stage === 1 && f <= fx && f > ftest) {
        const r = dcstep(
          stx,
          fx - stx * gtest,
          gx - gtest,
          sty,
          fy - sty * gtest,
          gy - gtest,
          stp,
          f - stp * gtest,
          g - gtest,
          brackt,
          stmin,
          stmax,
        )
        stx = r[0]
        sty = r[3]
        stp = r[6]
        brackt = r[7]
        fx = r[1] + stx * gtest
        fy = r[4] + sty * gtest
        gx = r[2] + gtest
        gy = r[5] + gtest
      } else {
        const r = dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax)
        ;[stx, fx, gx, sty, fy, gy, stp, brackt] = r
      }
      if (brackt) {
        if (Math.abs(sty - stx) >= 0.66 * width1) stp = stx + 0.5 * (sty - stx)
        width1 = width
        width = Math.abs(sty - stx)
        stmin = Math.min(stx, sty)
        stmax = Math.max(stx, sty)
      } else {
        stmin = stp + 1.1 * (stp - stx)
        stmax = stp + 4 * (stp - stx)
      }
      stp = Math.min(Math.max(stp, stpmin), stpmax)
      if ((brackt && (stp <= stmin || stp >= stmax)) || (brackt && stmax - stmin <= xtol * stmax)) stp = stx
      if (!Number.isFinite(stp)) return p.finish(undefined, f, g, 'non-finite step')
    }
    return p.finish(undefined, Number.NaN, Number.NaN, 'evaluation cap reached')
  }
}

/** Options of {@link hagerZhang}. */
export interface HagerZhangOptions {
  /** Sufficient-decrease constant delta (default 0.1). */
  readonly delta?: number
  /** Curvature constant sigma (default 0.9). */
  readonly sigma?: number
  /** Relative error allowed in the approximate Wolfe conditions (default 1e-6). */
  readonly epsilon?: number
  /** Required shrink factor of the bracket per secant step before bisecting (default 0.66). */
  readonly gamma?: number
  /** Expansion factor while bracketing (default 5). */
  readonly rho?: number
  /** Largest step (default Infinity). */
  readonly alphaMax?: number
  /** Most evaluations (default 50). */
  readonly maxEvaluations?: number
}

/**
 * The Hager-Zhang line search (CG_DESCENT): bracket, then secant^2 steps with bisection safeguards, accepting a step
 * that satisfies the Wolfe conditions or the approximate Wolfe conditions
 * (2 delta - 1) phi'(0) >= phi'(alpha) >= sigma phi'(0) with phi(alpha) <= phi(0) + epsilon |phi(0)|, which stay
 * reliable near a minimum where function differences are lost to rounding.
 *
 * @param options - delta, sigma, epsilon, gamma, rho, the largest step and the evaluation cap.
 * @returns The line search.
 * @concept line-search
 * @reference W. W. Hager and H. Zhang, A new conjugate gradient method with guaranteed descent and an efficient line
 * search, SIAM J. Optim. 16 (2005); Algorithm 851: CG_DESCENT, ACM TOMS 32 (2006).
 * @example
 * ```ts
 * import { hagerZhang } from 'omai-engine/optim'
 * // phi(alpha) = (alpha - 0.25)^2: the first trial (alpha = 1) overshoots, interpolation finds the minimum
 * const phi = { at: (a: number) => ({ value: (a - 0.25) ** 2, slope: 2 * (a - 0.25) }) }
 * const r = hagerZhang()(phi, { value: 0.0625, slope: -0.5 })
 * [r.success, r.alpha, r.trials.length] // => [true, 0.25, 2]
 * ```
 */
export function hagerZhang(options: HagerZhangOptions = {}): LineSearch {
  checkConstants(options.delta ?? 0.1, options.sigma ?? 0.9)
  return (phi, start) => {
    const search = hagerZhangSearch(phi, start, options)
    return search.result(runAlgorithm(search.algorithm, { maxSteps: Number.MAX_SAFE_INTEGER }).state)
  }
}

/** Options of {@link wolfe12}. */
export interface Wolfe12Options {
  readonly c1?: number
  readonly c2?: number
  /** The step bounds of the More-Thuente search (default 1e-8 and 50; scipy's BFGS passes 1e-100 and 1e100). */
  readonly alphaMin?: number
  readonly alphaMax?: number
  /** Also bound the fallback search by alphaMax (scipy passes amax on only when the caller gave one). */
  readonly boundFallback?: boolean
  /** An extra acceptance test: a More-Thuente step that fails it is rejected and the zoom search runs with it. */
  readonly extraCondition?: ExtraCondition
}

/**
 * The line search of scipy's BFGS, conjugate gradients and Newton-CG (its `_line_search_wolfe12`): More-Thuente, and
 * when that fails (or its step fails the extra condition), the strong Wolfe zoom.
 *
 * @param options - c1, c2, step bounds and an extra condition.
 * @returns The line search.
 * @concept line-search
 * @example
 * ```ts
 * import { wolfe12 } from 'omai-engine/optim'
 * const phi = { at: (a: number) => ({ value: (a - 0.25) ** 2, slope: 2 * (a - 0.25) }) }
 * wolfe12()(phi, { value: 0.0625, slope: -0.5 }).alpha // => 0.25
 * ```
 */
export function wolfe12(options: Wolfe12Options = {}): LineSearch {
  const c1 = options.c1 ?? 1e-4
  const c2 = options.c2 ?? 0.9
  const mt = moreThuente({
    c1,
    c2,
    ...(options.alphaMin === undefined ? {} : { alphaMin: options.alphaMin }),
    ...(options.alphaMax === undefined ? {} : { alphaMax: options.alphaMax }),
  })
  const zoom = strongWolfe({
    c1,
    c2,
    ...(options.boundFallback === true && options.alphaMax !== undefined ? { alphaMax: options.alphaMax } : {}),
    ...(options.extraCondition === undefined ? {} : { extraCondition: options.extraCondition }),
  })
  const extra = options.extraCondition
  return (phi, start) => {
    const first = mt(phi, start)
    const accepted =
      first.success &&
      (extra === undefined ||
        extra(first.alpha, {
          value: first.value,
          slope: first.slope,
          ...(first.grad === undefined ? {} : { grad: first.grad }),
        }))
    if (accepted) return first
    const second = zoom(phi, start)
    return {
      ...second,
      evaluations: first.evaluations + second.evaluations,
      gradEvaluations: first.gradEvaluations + second.gradEvaluations,
      trials: [...first.trials, ...second.trials],
    }
  }
}
