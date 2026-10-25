/**
 * The front doors to the ODE solvers. `solveIvp` follows scipy's `solve_ivp` (methods by name, dense output, events),
 * also as a traced algorithm; `odeint` returns states at given times as in scipy and torchdiffeq; `diffeqsolve` mirrors
 * diffrax with step-size controllers and saving policies. Also the LSODA-style automatic switch between Adams and BDF
 * when a problem turns stiff.
 */

import { type Json, WhiteboxError } from '../../base/index.ts'
import { brentq } from '../../numerics/roots/index.ts'
import { stack, type TensorLike, type Value, zeros } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { matVec } from './_newton.ts'
import { floatState, numbersOf, rms } from './_vec.ts'
import {
  type AdaptiveOptions,
  adaptiveRk,
  BOGACKI_SHAMPINE,
  CASH_KARP,
  DOP853_TABLEAU,
  DORMAND_PRINCE,
  type EmbeddedTableau,
  FEHLBERG,
  HEUN_EULER,
  TSITOURAS,
} from './adaptive.ts'
import { type ButcherTableau, EULER, explicitRk, HEUN, KUTTA3, MIDPOINT, RALSTON, RK4, RK38 } from './explicit.ts'
import {
  BACKWARD_EULER,
  type BdfState,
  bdf,
  dirk,
  IMPLICIT_MIDPOINT,
  type ImplicitOptions,
  radau,
  rosenbrock23,
  SDIRK2,
  SDIRK3,
  TRAPEZOID,
} from './implicit.ts'
import { adams, adamsBashforth, adamsMoulton, type NordsieckState, nordsieckAdamsL, pece } from './multistep.ts'
import {
  type DenseSegment,
  type DenseSolution,
  denseSolution,
  type EventSpec,
  type OdeSolution,
  type OdeStatus,
  type SolveCounts,
  type Stepper,
  type StepperOptions,
  type StepperState,
  type VectorField,
} from './problem.ts'

/** The state of {@link lsoda}: which method is active and its state. */
export interface LsodaState extends StepperState {
  readonly mode: 'adams' | 'bdf'
  readonly inner: StepperState
  /** The stepper state that took the last step (for dense output after a switch). */
  readonly last: StepperState
  readonly lastMode: 'adams' | 'bdf'
  readonly offset: SolveCounts
  readonly stepsInMode: number
  readonly stiffSteps: number
  /** Method switches so far. */
  readonly switches: number
}

function addCounts(a: SolveCounts, b: SolveCounts): SolveCounts {
  return { nfev: a.nfev + b.nfev, njev: a.njev + b.njev, nlu: a.nlu + b.nlu }
}

function spectralRadius(j: Float64Array, n: number): number {
  // Power iteration from a fixed start: enough to compare h rho with a stability bound.
  let v: Float64Array = Float64Array.from({ length: n }, (_, i) => 1 / Math.sqrt(n) + (i % 2 === 0 ? 0 : 1e-3))
  let rho = 0
  for (let k = 0; k < 20; k++) {
    const w = matVec(j, v, n)
    const norm = rms(w) * Math.sqrt(n)
    if (norm === 0) return 0
    rho = norm / (rms(v) * Math.sqrt(n))
    v = Float64Array.from(w, (x) => x / norm)
  }
  return rho
}

/**
 * Automatic stiffness switching in the manner of LSODA (Petzold 1983): start with variable-order Adams-Moulton
 * ({@link adams}) and move to variable-order BDF when the corrector's Lipschitz estimate shows that stability, not
 * accuracy, limits the step (h L l0 beyond 0.5 for three consecutive steps, or when the functional iteration fails);
 * move back when h rho(J) falls below 0.3. The switching logic and Adams details are this library's, so results agree
 * with scipy's LSODA to the solver tolerance, not to rounding.
 *
 * @param options - Tolerances, step limits and the Jacobian for the stiff phase.
 * @returns The stepper.
 * @concept stiffness
 * @example
 * ```ts
 * import { lsoda } from 'whitebox-engine/dynamics/ode'
 * lsoda().name // => 'lsoda'
 * ```
 */
export function lsoda(options: ImplicitOptions = {}): Stepper<LsodaState> {
  const nonstiff = (first?: number): Stepper<NordsieckState> =>
    adams({ ...options, ...(first === undefined ? {} : { firstStep: first }) })
  const stiff = (first?: number): Stepper<BdfState> =>
    bdf({ ...options, ...(first === undefined ? {} : { firstStep: first }) })
  const steppers = { adams: nonstiff(), bdf: stiff() }
  const firstFor = (s: StepperState): number => Math.min(s.hAbs, Math.abs(s.tBound - s.t))
  const restart = (s: LsodaState, inner: StepperState, mode: 'adams' | 'bdf'): LsodaState => {
    const first = firstFor(inner)
    if (!(first > 0)) return { ...s, inner, mode }
    const next = mode === 'adams' ? stiff(first) : nonstiff(first)
    const fresh = next.init((inner as { f: VectorField } & StepperState).f, inner.t, inner.y, inner.tBound)
    return {
      ...s,
      mode: mode === 'adams' ? 'bdf' : 'adams',
      inner: fresh,
      offset: addCounts(s.offset, inner.counts),
      stepsInMode: 0,
      stiffSteps: 0,
      switches: s.switches + 1,
    }
  }
  return {
    name: 'lsoda',
    order: 12,
    stiff: true,
    init: (f, t0, y0, tBound) => {
      const inner = steppers.adams.init(f, t0, y0, tBound)
      return {
        ...inner,
        mode: 'adams',
        inner,
        last: inner,
        lastMode: 'adams',
        offset: { nfev: 0, njev: 0, nlu: 0 },
        stepsInMode: 0,
        stiffSteps: 0,
        switches: 0,
      }
    },
    step: (s) => {
      if (s.status !== 'running') throw new WhiteboxError('value', 'lsoda: the integration has ended')
      let state = s
      for (let attemptMode = 0; attemptMode < 2; attemptMode++) {
        const stepper = state.mode === 'adams' ? steppers.adams : steppers.bdf
        const next = (stepper.step as unknown as (x: StepperState) => StepperState)(state.inner)
        if (next.status === 'failed' && state.mode === 'adams') {
          // The functional iteration gave up: treat it as stiffness and retry the step with BDF.
          state = restart(state, state.inner, 'adams')
          continue
        }
        let out: LsodaState = {
          ...state,
          t: next.t,
          y: next.y,
          tOld: next.tOld,
          yOld: next.yOld,
          hAbs: next.hAbs,
          status: next.status,
          message: next.message,
          attempts: next.attempts,
          order: next.order,
          inner: next,
          last: next,
          lastMode: state.mode,
          counts: addCounts(state.offset, next.counts),
          stepsInMode: state.stepsInMode + 1,
        }
        if (out.status !== 'running') return out
        const n = numbersOf(next.y).length
        if (state.mode === 'adams') {
          const ns = next as NordsieckState
          const l0 = nordsieckAdamsL(ns.order)[0] ?? 1
          const stiffNow = Math.abs(ns.h) * ns.lipschitz * l0 > 0.5
          const stiffSteps = stiffNow ? state.stiffSteps + 1 : 0
          out = { ...out, stiffSteps }
          if (stiffSteps >= 3 && out.stepsInMode >= 5)
            out = { ...restart(out, next, 'adams'), last: next, lastMode: 'adams' }
        } else if (out.stepsInMode % 10 === 0) {
          const bs = next as BdfState
          if (next.hAbs * spectralRadius(bs.jacobian, n) < 0.3) {
            out = { ...restart(out, next, 'bdf'), last: next, lastMode: 'bdf' }
          }
        }
        return { ...out, counts: addCounts(out.offset, out.inner.counts) }
      }
      return { ...state, status: 'failed', message: 'lsoda: both methods failed' }
    },
    dense: (s) =>
      s.lastMode === 'adams' ? steppers.adams.dense(s.last as NordsieckState) : steppers.bdf.dense(s.last as BdfState),
  }
}

/** The methods available by name (scipy's six and the rest of this folder). */
export type OdeMethodName =
  | 'RK23'
  | 'RK45'
  | 'DOP853'
  | 'Radau'
  | 'BDF'
  | 'LSODA'
  | 'Tsit5'
  | 'HeunEuler'
  | 'Fehlberg45'
  | 'CashKarp'
  | 'Rosenbrock23'
  | 'Adams'
  | 'Euler'
  | 'Heun'
  | 'Midpoint'
  | 'Ralston'
  | 'Kutta3'
  | 'RK4'
  | 'RK38'
  | 'BackwardEuler'
  | 'Trapezoid'
  | 'ImplicitMidpoint'
  | 'SDIRK2'
  | 'SDIRK3'
  | 'AdamsBashforth'
  | 'AdamsMoulton'
  | 'PECE'

const EMBEDDED: Readonly<Partial<Record<OdeMethodName, EmbeddedTableau>>> = {
  RK23: BOGACKI_SHAMPINE,
  RK45: DORMAND_PRINCE,
  DOP853: DOP853_TABLEAU,
  Tsit5: TSITOURAS,
  HeunEuler: HEUN_EULER,
  Fehlberg45: FEHLBERG,
  CashKarp: CASH_KARP,
}
const FIXED: Readonly<Partial<Record<OdeMethodName, ButcherTableau>>> = {
  Euler: EULER,
  Heun: HEUN,
  Midpoint: MIDPOINT,
  Ralston: RALSTON,
  Kutta3: KUTTA3,
  RK4: RK4,
  RK38: RK38,
}

/** Options that choose and configure a method by name. */
export interface MethodOptions extends AdaptiveOptions, Omit<ImplicitOptions, keyof StepperOptions> {
  /** The fixed step of fixed-step methods. */
  readonly dt?: number
  /** The order of the fixed-step multistep methods. */
  readonly order?: 1 | 2 | 3 | 4 | 5
}

/**
 * The stepper for a method name with its options (fixed-step methods need `dt`).
 *
 * @param method - A method name or a stepper (returned as is).
 * @param options - Tolerances, step limits, controller, Jacobian, mass matrix and fixed step.
 * @returns The stepper.
 * @throws WhiteboxError with code `value` for an unknown name or a fixed-step method without `dt`.
 * @example
 * ```ts
 * import { stepperFor } from 'whitebox-engine/dynamics/ode'
 * stepperFor('DOP853').order // => 8
 * ```
 */
export function stepperFor(method: OdeMethodName | Stepper, options: MethodOptions = {}): Stepper {
  if (typeof method !== 'string') return method
  const tableau = EMBEDDED[method]
  if (tableau !== undefined) return adaptiveRk(tableau, options) as unknown as Stepper
  const implicitOptions: ImplicitOptions = {
    ...pick(options, ['rtol', 'atol', 'maxStep', 'firstStep']),
    ...(options.jac === undefined ? {} : { jac: options.jac }),
    ...(options.mass === undefined ? {} : { mass: options.mass }),
  }
  switch (method) {
    case 'Radau':
      return radau(implicitOptions) as unknown as Stepper
    case 'BDF':
      return bdf(implicitOptions) as unknown as Stepper
    case 'LSODA':
      return lsoda(implicitOptions) as unknown as Stepper
    case 'Rosenbrock23':
      return rosenbrock23(implicitOptions) as unknown as Stepper
    case 'Adams':
      return adams(implicitOptions) as unknown as Stepper
    default:
      break
  }
  const dt = options.dt
  if (dt === undefined) throw new WhiteboxError('value', `${method} is a fixed-step method: give dt`)
  const fixed = FIXED[method]
  if (fixed !== undefined) return explicitRk(fixed, { dt }) as unknown as Stepper
  const dirkOptions = { dt, ...(options.jac === undefined ? {} : { jac: options.jac }) }
  const order = options.order ?? 4
  switch (method) {
    case 'BackwardEuler':
      return dirk(BACKWARD_EULER, dirkOptions) as unknown as Stepper
    case 'Trapezoid':
      return dirk(TRAPEZOID, dirkOptions) as unknown as Stepper
    case 'ImplicitMidpoint':
      return dirk(IMPLICIT_MIDPOINT, dirkOptions) as unknown as Stepper
    case 'SDIRK2':
      return dirk(SDIRK2, dirkOptions) as unknown as Stepper
    case 'SDIRK3':
      return dirk(SDIRK3, dirkOptions) as unknown as Stepper
    case 'AdamsBashforth':
      return adamsBashforth({ ...dirkOptions, order }) as unknown as Stepper
    case 'AdamsMoulton':
      return adamsMoulton({ ...dirkOptions, order }) as unknown as Stepper
    case 'PECE':
      return pece({ ...dirkOptions, order }) as unknown as Stepper
    default:
      throw new WhiteboxError('value', `unknown ODE method ${String(method)}`)
  }
}

function pick<T extends object, K extends keyof T>(o: T, keys: readonly K[]): Partial<Pick<T, K>> {
  const out: Partial<Pick<T, K>> = {}
  for (const k of keys) if (o[k] !== undefined) out[k] = o[k]
  return out
}

/** Tracing options (everything of `runAlgorithm` except the step budget and start). */
export type IvpTraceOptions = Partial<Omit<RunOptions, 'maxSteps' | 'start'>>

/** Options of {@link solveIvp}. */
export interface SolveIvpOptions extends MethodOptions, IvpTraceOptions {
  /** The method (default `RK45`). */
  readonly method?: OdeMethodName | Stepper
  /** Times at which to store the solution (within the span, sorted in the direction of integration). */
  readonly tEval?: readonly number[]
  /** Keep the continuous solution. */
  readonly denseOutput?: boolean
  readonly events?: EventSpec | readonly EventSpec[]
  /** The step budget (default 100000): the solve fails with a message when it is used up. */
  readonly maxSteps?: number
}

/** The state of {@link solveIvpAlgorithm}. The output arrays grow in place: one run owns them. */
export interface IvpState {
  readonly stepper: StepperState
  readonly ts: number[]
  readonly ys: Value[]
  readonly segments: DenseSegment[]
  readonly ti: number[]
  readonly g: readonly number[]
  readonly eventCount: readonly number[]
  readonly tEvents: number[][]
  readonly yEvents: Value[][]
  readonly tEvalIndex: number
  readonly status: OdeStatus | undefined
  readonly message: string
  readonly nSteps: number
  readonly nRejected: number
  readonly lastEvents: readonly { readonly index: number; readonly t: number }[]
  /** Every evaluation of the field in this run (steps, step selection and dense output), as scipy's nfev. */
  readonly calls: { n: number }
}

const SCHEMA = channels({
  metrics: {
    t: { description: 'the time reached' },
    h: { description: 'the size of the last accepted step' },
    error: { description: 'the scaled error norm of the last accepted step (1 is the tolerance)' },
    rejected: { description: 'step attempts rejected before the last accepted step' },
    order: { description: 'the order in use' },
    nfev: { description: 'vector field evaluations so far' },
  },
  tensors: { y: { role: 'state', description: 'the state at t' } },
  messages: {
    rejected: { description: 'a rejected step attempt', payload: { h: 'attempted step', error: 'scaled error norm' } },
    event: { description: 'an event located in the last step', payload: { index: 'event index', t: 'event time' } },
  },
})

function findActive(g: readonly number[], gNew: readonly number[], direction: readonly number[]): number[] {
  const out: number[] = []
  for (let i = 0; i < g.length; i++) {
    const a = g[i] ?? 0
    const b = gNew[i] ?? 0
    const up = a <= 0 && b >= 0
    const down = a >= 0 && b <= 0
    const d = direction[i] ?? 0
    if ((up && d > 0) || (down && d < 0) || ((up || down) && d === 0)) out.push(i)
  }
  return out
}

function maxEventsOf(e: EventSpec): number {
  const t = e.terminal
  if (t === undefined || t === false || t === 0) return Number.POSITIVE_INFINITY
  if (t === true) return 1
  if (Number.isInteger(t) && t > 0) return t
  throw new WhiteboxError('value', 'an event is terminal (true or a positive integer) or not')
}

/**
 * {@link solveIvp} as a traced algorithm: one accepted step per algorithm step, with the step size, error norm,
 * rejected attempts and located events as trace events.
 *
 * @param f - The vector field.
 * @param tspan - [t0, t1].
 * @param y0 - The initial state.
 * @param options - As {@link solveIvp}.
 * @returns The algorithm and a function turning its final state into the solution.
 * @example
 * ```ts
 * import { solveIvpAlgorithm } from 'whitebox-engine/dynamics/ode'
 * import { neg } from 'whitebox-engine/tensor'
 * solveIvpAlgorithm((_t, y) => neg(y), [0, 1], [1]).algorithm.name // => 'solveIvp'
 * ```
 */
export function solveIvpAlgorithm(
  f: VectorField,
  tspan: readonly [number, number],
  y0: TensorLike,
  options: SolveIvpOptions = {},
): { readonly algorithm: Algorithm<IvpState>; readonly finish: (state: IvpState) => OdeSolution } {
  const [t0, tf] = tspan
  if (!Number.isFinite(t0) || !Number.isFinite(tf))
    throw new WhiteboxError('value', 'solveIvp: the span must be finite')
  const y0v = floatState(y0)
  const stepper = stepperFor(options.method ?? 'RK45', options)
  const events =
    options.events === undefined ? [] : Array.isArray(options.events) ? options.events : [options.events as EventSpec]
  const maxEvents = events.map(maxEventsOf)
  const eventDir = events.map((e) => e.direction ?? 0)
  const forward = tf >= t0
  let tEval = options.tEval === undefined ? undefined : [...options.tEval]
  if (tEval !== undefined) {
    const lo = Math.min(t0, tf)
    const hi = Math.max(t0, tf)
    if (tEval.some((t) => t < lo || t > hi)) throw new WhiteboxError('value', 'solveIvp: tEval must lie in the span')
    for (let i = 1; i < tEval.length; i++) {
      const d = (tEval[i] ?? 0) - (tEval[i - 1] ?? 0)
      if ((forward && d <= 0) || (!forward && d >= 0)) {
        throw new WhiteboxError('value', 'solveIvp: tEval must be sorted in the direction of integration')
      }
    }
    if (!forward) tEval = tEval.reverse()
  }
  const dense = options.denseOutput === true
  const evalOrder = tEval
  const algorithm: Algorithm<IvpState> = {
    name: 'solveIvp',
    config: {
      method: typeof options.method === 'string' ? options.method : (options.method?.name ?? 'RK45'),
      rtol: (options.rtol ?? 1e-3) as Json,
      atol: (options.atol ?? 1e-6) as Json,
      t0,
      tf,
    },
    schema: SCHEMA,
    init: () => {
      const calls = { n: 0 }
      const counted: VectorField = (t, y) => {
        calls.n += 1
        return f(t, y)
      }
      const st = stepper.init(counted, t0, y0v, tf)
      return {
        stepper: st,
        ts: evalOrder === undefined ? [t0] : [],
        ys: evalOrder === undefined ? [y0v] : [],
        segments: [],
        ti: [t0],
        g: events.map((e) => e.fn(t0, y0v)),
        eventCount: events.map(() => 0),
        tEvents: events.map(() => []),
        yEvents: events.map(() => []),
        tEvalIndex: evalOrder === undefined ? 0 : forward ? 0 : evalOrder.length,
        status: undefined,
        message: '',
        nSteps: 0,
        nRejected: 0,
        lastEvents: [],
        calls,
      }
    },
    step: (s) => {
      const st = stepper.step(s.stepper)
      const rejected = st.attempts.filter((a) => !a.accepted).length
      if (st.status === 'failed') {
        return {
          ...s,
          stepper: st,
          status: 'failed',
          message: st.message,
          nRejected: s.nRejected + rejected,
          lastEvents: [],
        }
      }
      let status: OdeStatus | undefined = st.status === 'finished' ? 'success' : undefined
      let t = st.t
      let y = st.y
      let sol: DenseSegment | undefined = dense ? stepper.dense(st) : undefined
      if (sol !== undefined) s.segments.push(sol)
      const lastEvents: { index: number; t: number }[] = []
      let g = s.g
      let eventCount = s.eventCount
      if (events.length > 0) {
        const gNew = events.map((e) => e.fn(t, y))
        let active = findActive(g, gNew, eventDir)
        if (active.length > 0) {
          const seg = sol ?? stepper.dense(st)
          sol = seg
          eventCount = eventCount.map((c, i) => (active.includes(i) ? c + 1 : c))
          let roots = active.map((i) => {
            const ev = events[i] as EventSpec
            const r = brentq((tt) => ev.fn(tt, seg.at(tt)), st.tOld, t, {
              xtol: 4 * Number.EPSILON,
              rtol: 4 * Number.EPSILON,
            })
            return r.root
          })
          let terminate = false
          if (active.some((i) => (eventCount[i] ?? 0) >= (maxEvents[i] ?? Number.POSITIVE_INFINITY))) {
            const order = roots
              .map((_, k) => k)
              .sort((a, b) => (t > st.tOld ? (roots[a] ?? 0) - (roots[b] ?? 0) : (roots[b] ?? 0) - (roots[a] ?? 0)))
            active = order.map((k) => active[k] as number)
            roots = order.map((k) => roots[k] as number)
            const stop = active.findIndex((i) => (eventCount[i] ?? 0) >= (maxEvents[i] ?? Number.POSITIVE_INFINITY))
            active = active.slice(0, stop + 1)
            roots = roots.slice(0, stop + 1)
            terminate = true
          }
          active.forEach((i, k) => {
            const te = roots[k] as number
            s.tEvents[i]?.push(te)
            s.yEvents[i]?.push(seg.at(te))
            lastEvents.push({ index: i, t: te })
          })
          if (terminate) {
            status = 'event'
            t = roots[roots.length - 1] as number
            y = seg.at(t)
          }
        }
        g = gNew
      }
      let tEvalIndex = s.tEvalIndex
      if (evalOrder === undefined) {
        const skip = s.ts.length > 1 && s.ts[s.ts.length - 1] === t && dense
        if (!skip) {
          s.ts.push(t)
          s.ys.push(y)
        } else s.segments.pop()
      } else {
        let chunk: number[]
        let next: number
        if (forward) {
          next = tEvalIndex
          while (next < evalOrder.length && (evalOrder[next] ?? 0) <= t) next++
          chunk = evalOrder.slice(tEvalIndex, next)
        } else {
          next = tEvalIndex
          while (next > 0 && (evalOrder[next - 1] ?? 0) >= t) next--
          chunk = evalOrder.slice(next, tEvalIndex).reverse()
        }
        if (chunk.length > 0) {
          const seg = sol ?? stepper.dense(st)
          for (const te of chunk) {
            s.ts.push(te)
            s.ys.push(seg.at(te))
          }
          tEvalIndex = next
        }
      }
      if (evalOrder !== undefined && dense) s.ti.push(t)
      return {
        ...s,
        stepper: st,
        g,
        eventCount,
        tEvalIndex,
        status,
        message:
          status === 'event'
            ? 'A termination event occurred.'
            : status === 'success'
              ? 'The solver successfully reached the end of the integration interval.'
              : '',
        nSteps: s.nSteps + 1,
        nRejected: s.nRejected + rejected,
        lastEvents,
      }
    },
    done: (s) =>
      s.status === undefined
        ? undefined
        : s.status === 'failed'
          ? 'failed'
          : s.status === 'event'
            ? 'stopped'
            : 'completed',
    observe: (s) => {
      const acc = s.stepper.attempts.find((a) => a.accepted)
      return {
        metrics: {
          t: s.stepper.t,
          h: acc?.h ?? 0,
          error: acc?.error ?? 0,
          rejected: s.stepper.attempts.filter((a) => !a.accepted).length,
          order: s.stepper.order,
          nfev: s.calls.n,
        },
        tensors: { y: { role: 'state', value: numbersOf(s.stepper.y), shape: s.stepper.y.shape } },
        messages: [
          ...s.stepper.attempts
            .filter((a) => !a.accepted)
            .map((a) => ({ kind: 'rejected', payload: { h: a.h, error: Number.isFinite(a.error) ? a.error : null } })),
          ...s.lastEvents.map((e) => ({ kind: 'event', payload: { index: e.index, t: e.t } })),
        ],
      }
    },
  }
  const finish = (s: IvpState): OdeSolution => {
    const status: OdeStatus = s.status ?? 'failed'
    const message =
      s.status === undefined ? 'solveIvp: the step budget was used up before the end of the interval' : s.message
    const y = s.ys.length === 0 ? zeros([0, ...y0v.shape]) : stack(s.ys)
    let solution: DenseSolution | undefined
    if (dense && s.segments.length > 0) solution = denseSolution(s.segments)
    return {
      t: s.ts,
      y,
      ys: s.ys,
      sol: solution,
      tEvents: s.tEvents,
      yEvents: s.yEvents,
      status,
      success: status !== 'failed',
      message,
      nfev: s.calls.n,
      njev: s.stepper.counts.njev,
      nlu: s.stepper.counts.nlu,
      nSteps: s.nSteps,
      nRejected: s.nRejected,
    }
  }
  return { algorithm, finish }
}

/**
 * Solve an initial value problem with scipy's `solve_ivp` semantics: a method by name (RK23, RK45, DOP853, Radau,
 * BDF, LSODA and the others of this folder) or any stepper, output at the steps or at `tEval` (interpolated by each
 * step's dense output), an optional continuous solution, and events located by Brent's method on the dense output
 * (with direction filters and terminal counts). States may have any shape and may be traced, so the solution is
 * differentiable through the steps. The run is a traced algorithm with a step budget; it never throws on a hard
 * problem but returns a failed status with a message.
 *
 * @param f - The vector field dy/dt = f(t, y).
 * @param tspan - [t0, t1].
 * @param y0 - The initial state.
 * @param options - Method, tolerances (default rtol 1e-3, atol 1e-6), output times, dense output, events, step
 * budget and tracing.
 * @returns The solution (states time-major, `[nt, ...stateShape]`).
 * @throws WhiteboxError with code `value` for a bad span, method or output times.
 * @concept initial-value-problem
 * @example
 * ```ts
 * import { solveIvp } from 'whitebox-engine/dynamics/ode'
 * import { neg } from 'whitebox-engine/tensor'
 * const sol = solveIvp((_t, y) => neg(y), [0, 1], [1], { rtol: 1e-10, atol: 1e-12, tEval: [1] })
 * Math.abs((sol.ys[0]?.data[0] ?? 0) - Math.exp(-1)) < 1e-9 // => true
 * ```
 */
export function solveIvp(
  f: VectorField,
  tspan: readonly [number, number],
  y0: TensorLike,
  options: SolveIvpOptions = {},
): OdeSolution {
  const { algorithm, finish } = solveIvpAlgorithm(f, tspan, y0, options)
  const result = runAlgorithm(algorithm, {
    maxSteps: options.maxSteps ?? 100000,
    ...pick(options, ['seed', 'key', 'tracer', 'policy', 'runId', 'parent']),
  })
  return finish(result.state)
}

/** Options of {@link odeint}. */
export interface OdeintOptions extends Omit<SolveIvpOptions, 'tEval' | 'events' | 'denseOutput'> {}

/**
 * The solution at the given times (scipy's and torchdiffeq's `odeint`, with the field taking (t, y)): times[0] is the
 * initial time, and the result stacks the states, `[nt, ...stateShape]`. The default method is LSODA with scipy's
 * odeint tolerances (rtol = atol = 1.49012e-8).
 *
 * @param f - The vector field.
 * @param y0 - The initial state.
 * @param times - Increasing (or decreasing) output times starting at the initial time.
 * @param options - As {@link solveIvp}.
 * @returns The states at the times.
 * @throws WhiteboxError with code `convergence` when the solver fails.
 * @example
 * ```ts
 * import { odeint } from 'whitebox-engine/dynamics/ode'
 * import { neg } from 'whitebox-engine/tensor'
 * odeint((_t, y) => neg(y), [1], [0, 1, 2]).shape // => [3, 1]
 * ```
 */
export function odeint(f: VectorField, y0: TensorLike, times: readonly number[], options: OdeintOptions = {}): Value {
  return stack(odeintStates(f, y0, times, options))
}

function odeintStates(f: VectorField, y0: TensorLike, times: readonly number[], options: OdeintOptions): Value[] {
  const t0 = times[0]
  const t1 = times[times.length - 1]
  const y0v = floatState(y0)
  if (t0 === undefined || t1 === undefined) throw new WhiteboxError('value', 'odeint: give at least one time')
  if (times.length === 1 || t0 === t1) return times.map(() => y0v)
  const sol = solveIvp(f, [t0, t1], y0v, {
    method: 'LSODA',
    rtol: 1.49012e-8,
    atol: 1.49012e-8,
    ...options,
    tEval: times,
  })
  if (!sol.success) throw new WhiteboxError('convergence', `odeint: ${sol.message}`)
  // The first output is the initial state itself (exactly, and with its derivatives).
  return [y0v, ...sol.ys.slice(1)]
}

/**
 * A solver in the shape the neural ODE blocks take (`nn` `OdeSolver`): the states at each time, differentiable through
 * the steps.
 *
 * @param method - The method (default `Tsit5`).
 * @param options - Tolerances and other method options.
 * @returns `(f, y0, times) => states`.
 * @example
 * ```ts
 * import { odeSolver } from 'whitebox-engine/dynamics/ode'
 * import { neg, tensor } from 'whitebox-engine/tensor'
 * odeSolver('RK4', { dt: 0.1 })((_t, y) => neg(y), tensor([1]), [0, 0.5, 1]).length // => 3
 * ```
 */
export function odeSolver(
  method: OdeMethodName | Stepper = 'Tsit5',
  options: Omit<OdeintOptions, 'method'> = {},
): (f: (t: number, y: Value) => Value, y0: Value, times: readonly number[]) => Value[] {
  return (f, y0, times) => odeintStates(f, y0, times, { rtol: 1e-6, atol: 1e-8, ...options, method })
}

/** diffrax's step-size controllers: a constant step, or a PID controller on the embedded error estimate. */
export type StepSizeController =
  | { readonly kind: 'constant' }
  | {
      readonly kind: 'pid'
      readonly rtol?: number
      readonly atol?: number
      readonly pcoeff?: number
      readonly icoeff?: number
      readonly dcoeff?: number
      readonly dtmax?: number
      readonly safety?: number
      readonly factormin?: number
      readonly factormax?: number
    }

/** diffrax's saving policy: output times, the endpoints, every step, or the continuous solution. */
export interface SaveAt {
  readonly ts?: readonly number[]
  readonly t0?: boolean
  readonly t1?: boolean
  readonly steps?: boolean
  readonly dense?: boolean
}

/** Options of {@link diffeqsolve}. */
export interface DiffeqsolveOptions extends IvpTraceOptions {
  /** An embedded pair or fixed-step method by name (default `Tsit5`), or a stepper. */
  readonly solver?: OdeMethodName | Stepper
  readonly t0: number
  readonly t1: number
  /** The initial (or constant) step. */
  readonly dt0?: number
  readonly saveat?: SaveAt
  readonly stepsizeController?: StepSizeController
  /** The step budget (default 4096, as diffrax). */
  readonly maxSteps?: number
  readonly events?: EventSpec | readonly EventSpec[]
}

/** The result of {@link diffeqsolve} (diffrax's `Solution`). */
export interface DiffeqSolution {
  readonly ts: readonly number[]
  readonly ys: Value
  readonly sol: DenseSolution | undefined
  readonly result: 'successful' | 'max_steps_reached' | 'event_occurred' | 'failed'
  readonly stats: {
    readonly numSteps: number
    readonly numAcceptedSteps: number
    readonly numRejectedSteps: number
    readonly nfev: number
  }
  readonly message: string
}

/**
 * Solve an ODE in the style of diffrax's `diffeqsolve`: a solver, a step-size controller (constant, or a PID
 * controller on the embedded error with diffrax's coefficients) and a saving policy.
 *
 * @param f - The vector field (diffrax's ODETerm).
 * @param y0 - The initial state.
 * @param options - Solver, interval, initial step, saving policy, controller, step budget and events.
 * @returns The solution with statistics and a result code.
 * @throws WhiteboxError with code `value` when a constant step is requested without dt0.
 * @example
 * ```ts
 * import { diffeqsolve } from 'whitebox-engine/dynamics/ode'
 * import { neg } from 'whitebox-engine/tensor'
 * const s = diffeqsolve((_t, y) => neg(y), [1], { t0: 0, t1: 1, dt0: 0.1, stepsizeController: { kind: 'pid', rtol: 1e-8, atol: 1e-8 } })
 * [s.result, s.ts] // => ['successful', [1]]
 * ```
 */
export function diffeqsolve(f: VectorField, y0: TensorLike, options: DiffeqsolveOptions): DiffeqSolution {
  const controller = options.stepsizeController ?? { kind: 'constant' }
  const name = options.solver ?? 'Tsit5'
  let stepper: Stepper
  if (typeof name !== 'string') stepper = name
  else if (controller.kind === 'constant') {
    const dt = options.dt0
    if (dt === undefined) throw new WhiteboxError('value', 'diffeqsolve: a constant step needs dt0')
    const tableau = EMBEDDED[name] ?? FIXED[name]
    stepper =
      tableau === undefined
        ? stepperFor(name, { dt })
        : (explicitRk(tableau, { dt: Math.abs(dt) }) as unknown as Stepper)
  } else {
    const tableau = EMBEDDED[name]
    if (tableau === undefined) {
      stepper = stepperFor(name, {
        ...pick(controller, ['rtol', 'atol']),
        ...(options.dt0 === undefined ? {} : { firstStep: Math.abs(options.dt0) }),
      })
    } else {
      const k = tableau.errorOrder + 1
      const p = controller.pcoeff ?? 0
      const i = controller.icoeff ?? 1
      const d = controller.dcoeff ?? 0
      stepper = adaptiveRk(tableau, {
        rtol: controller.rtol ?? 1e-3,
        atol: controller.atol ?? 1e-6,
        controller: { beta1: (p + i + d) / k, beta2: (p + 2 * d) / k, beta3: d / k },
        safety: controller.safety ?? 0.9,
        minFactor: controller.factormin ?? 0.2,
        maxFactor: controller.factormax ?? 10,
        ...(controller.dtmax === undefined ? {} : { maxStep: controller.dtmax }),
        ...(options.dt0 === undefined
          ? {}
          : { firstStep: Math.min(Math.abs(options.dt0), Math.abs(options.t1 - options.t0)) }),
      }) as unknown as Stepper
    }
  }
  const save = options.saveat ?? { t1: true }
  const ts = [
    ...(save.t0 === true ? [options.t0] : []),
    ...(save.ts ?? []),
    ...(save.t1 === true && save.steps !== true ? [options.t1] : []),
  ]
  const { algorithm, finish } = solveIvpAlgorithm(f, [options.t0, options.t1], y0, {
    method: stepper,
    ...(save.steps === true ? {} : { tEval: dedupe(ts.filter((t) => t !== options.t0)) }),
    denseOutput: save.dense === true,
    ...(options.events === undefined ? {} : { events: options.events }),
  })
  const run = runAlgorithm(algorithm, {
    maxSteps: options.maxSteps ?? 4096,
    ...pick(options, ['seed', 'key', 'tracer', 'policy', 'runId', 'parent']),
  })
  const sol = finish(run.state)
  const y0v = floatState(y0)
  const withStart = save.t0 === true && save.steps !== true
  const outTs = save.steps === true ? sol.t : withStart ? [options.t0, ...sol.t] : sol.t
  const outYs = save.steps === true ? sol.ys : withStart ? [y0v, ...sol.ys] : sol.ys
  const result =
    run.state.status === undefined
      ? 'max_steps_reached'
      : sol.status === 'failed'
        ? 'failed'
        : sol.status === 'event'
          ? 'event_occurred'
          : 'successful'
  return {
    ts: outTs,
    ys: outYs.length === 0 ? zeros([0, ...y0v.shape]) : stack(outYs),
    sol: sol.sol,
    result,
    stats: {
      numSteps: sol.nSteps + sol.nRejected,
      numAcceptedSteps: sol.nSteps,
      numRejectedSteps: sol.nRejected,
      nfev: sol.nfev,
    },
    message: sol.message,
  }
}

function dedupe(ts: readonly number[]): number[] {
  const out: number[] = []
  for (const t of ts) if (out[out.length - 1] !== t) out.push(t)
  return out
}
