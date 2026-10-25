/**
 * Block-diagram algebra for linear systems, python-control style: systems in series, in parallel and in a feedback
 * loop, stacking systems side by side with `append`, and wiring their channels with `connect`, so that a closed-loop
 * system can be built from its parts.
 */

import { WhiteboxError } from '../base/index.ts'
import { convolve } from '../numerics/convolution/index.ts'
import { blockDiag, solve } from '../numerics/linalg/index.ts'
import {
  add,
  concat,
  eye,
  isValue,
  matmul,
  mul,
  pad,
  reshape,
  sub,
  type TensorLike,
  tensor,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { commonDt, float, numbers } from './_shared.ts'
import {
  type Lti,
  type StateSpace,
  type TransferFunction,
  tf,
  toSs,
  toTf,
  toZpk,
  type ZerosPolesGain,
} from './representations.ts'

// Interconnections follow python-control: series(G1, G2) is G2 G1 (the signal passes G1 first), feedback is negative
// by default, connect takes a 1-based interconnection matrix. Transfer functions combine as polynomials (differentiable
// in the coefficients), zero-pole-gain systems through them; anything multivariable or mixed goes to state space.

/** A system, or a static gain (a number or a gain matrix). */
export type SystemLike = Lti | TensorLike

const KINDS: readonly unknown[] = ['ss', 'tf', 'zpk', 'sos']
const isLti = (x: SystemLike): x is Lti =>
  typeof x === 'object' && x !== null && !isValue(x) && 'kind' in x && KINDS.includes((x as { kind: unknown }).kind)

/** A static gain as a state-space system without states. */
function gainSs(k: TensorLike, dt: number): StateSpace {
  let d = float(k)
  if (d.shape.length === 0) d = reshape(d, [1, 1])
  else if (d.shape.length === 1) d = reshape(d, [1, d.shape[0] ?? 0])
  const p = d.shape[0] ?? 0
  const m = d.shape[1] ?? 0
  return { kind: 'ss', a: zeros([0, 0]), b: zeros([0, m]), c: zeros([p, 0]), d, dt }
}

function dtOf(xs: readonly SystemLike[], name: string): number {
  return commonDt(name, ...xs.filter(isLti).map((x) => x.dt))
}

const siso = (x: SystemLike): boolean => {
  if (!isLti(x)) return numbers(float(x)).length === 1
  if (x.kind === 'ss') return x.b.shape[1] === 1 && x.c.shape[0] === 1
  return x.kind !== 'tf' || x.num.shape.length === 1
}

/** The kind of the result: transfer function or zero-pole-gain when every system (gains aside) is one, else state space. */
function formOf(xs: readonly SystemLike[]): 'tf' | 'zpk' | 'ss' {
  const systems = xs.filter(isLti)
  if (systems.every((x) => x.kind === 'tf')) return 'tf'
  if (systems.every((x) => x.kind === 'zpk')) return 'zpk'
  return 'ss'
}

/** A state-space result in the requested form. */
function finish(form: 'tf' | 'zpk' | 'ss', s: StateSpace): Lti {
  return form === 'tf' ? toTf(s) : form === 'zpk' ? toZpk(s) : s
}

function asTf(x: SystemLike, dt: number): TransferFunction {
  return isLti(x) ? toTf(x) : tf(reshape(float(x), [1]), [1], { dt })
}

function asSs(x: SystemLike, dt: number): StateSpace {
  return isLti(x) ? toSs(x) : gainSs(x, dt)
}

/** Highest-first polynomial sum of values (right-aligned). */
function polyAdd(a: Value, b: Value): Value {
  const n = Math.max(a.shape[0] ?? 0, b.shape[0] ?? 0)
  const left = (v: Value): Value => pad(v, [[n - (v.shape[0] ?? 0), 0]])
  return add(left(a), left(b))
}

function back(form: 'tf' | 'zpk', g: TransferFunction): Lti {
  return form === 'zpk' ? toZpk(g) : g
}

function ssSeries(s1: StateSpace, s2: StateSpace): StateSpace {
  const n1 = s1.a.shape[0] ?? 0
  const n2 = s2.a.shape[0] ?? 0
  if ((s1.c.shape[0] ?? 0) !== (s2.b.shape[1] ?? 0)) {
    throw new WhiteboxError('shape', 'series: the outputs of the first system must feed the inputs of the second')
  }
  const a = concat([concat([s1.a, zeros([n1, n2])], 1), concat([matmul(s2.b, s1.c), s2.a], 1)], 0)
  const b = concat([s1.b, matmul(s2.b, s1.d)], 0)
  const c = concat([matmul(s2.d, s1.c), s2.c], 1)
  return { kind: 'ss', a, b, c, d: matmul(s2.d, s1.d), dt: s1.dt }
}

function ssParallel(s1: StateSpace, s2: StateSpace): StateSpace {
  if (s1.d.shape[0] !== s2.d.shape[0] || s1.d.shape[1] !== s2.d.shape[1]) {
    throw new WhiteboxError('shape', 'parallel: the systems must have the same inputs and outputs')
  }
  return {
    kind: 'ss',
    a: blockDiag(s1.a, s2.a),
    b: concat([s1.b, s2.b], 0),
    c: concat([s1.c, s2.c], 1),
    d: add(s1.d, s2.d),
    dt: s1.dt,
  }
}

/** u1 = r + sign y2, y1 = G1 u1, y2 = G2 y1: the closed loop from r to y1. */
function ssFeedback(s1: StateSpace, s2: StateSpace, sign: number): StateSpace {
  const p = s1.c.shape[0] ?? 0
  const m = s1.b.shape[1] ?? 0
  if ((s2.b.shape[1] ?? 0) !== p || (s2.c.shape[0] ?? 0) !== m) {
    throw new WhiteboxError('shape', 'feedback: the loop dimensions do not match')
  }
  // y1 = M (C1 x1 + sign D1 C2 x2 + D1 r) with M = (I - sign D1 D2)^{-1}.
  const loop = sub(eye(p), mul(matmul(s1.d, s2.d), sign))
  const n1 = s1.a.shape[0] ?? 0
  const n2 = s2.a.shape[0] ?? 0
  const yx = solve(loop, concat([s1.c, mul(matmul(s1.d, s2.c), sign)], 1))
  const yr = solve(loop, s1.d)
  // u1 = r + sign (C2 x2 + D2 y1).
  const ux = add(concat([zeros([m, n1]), mul(s2.c, sign)], 1), mul(matmul(s2.d, yx), sign))
  const ur = add(eye(m), mul(matmul(s2.d, yr), sign))
  const a = add(
    concat([concat([s1.a, zeros([n1, n2])], 1), concat([zeros([n2, n1]), s2.a], 1)], 0),
    concat([matmul(s1.b, ux), matmul(s2.b, yx)], 0),
  )
  const b = concat([matmul(s1.b, ur), matmul(s2.b, yr)], 0)
  return { kind: 'ss', a, b, c: yx, d: yr, dt: s1.dt }
}

/**
 * Systems in series (python-control series): the output of each feeds the next, so series(G1, G2) = G2 G1.
 *
 * @param systems - Two or more systems or static gains, first to last.
 * @returns The series connection.
 * @throws WhiteboxError with code `value` for different sample times, `shape` when the channels do not chain.
 * @concept block-diagrams
 * @example
 * ```ts
 * import { series, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(series(tf([1], [1, 1]), tf([2], [1, 2])).den) // => [1, 3, 2]
 * ```
 */
export function series(...systems: readonly (TransferFunction | number)[]): TransferFunction
export function series(...systems: readonly (ZerosPolesGain | number)[]): ZerosPolesGain
export function series(...systems: readonly SystemLike[]): Lti
export function series(...systems: readonly SystemLike[]): Lti {
  if (systems.length < 2) throw new WhiteboxError('value', 'series: give at least two systems')
  const dt = dtOf(systems, 'series')
  const form = formOf(systems)
  if (form !== 'ss' && systems.every(siso)) {
    const gs = systems.map((x) => asTf(x, dt))
    const num = gs.slice(1).reduce((acc, g) => convolve(acc, g.num), (gs[0] as TransferFunction).num)
    const den = gs.slice(1).reduce((acc, g) => convolve(acc, g.den), (gs[0] as TransferFunction).den)
    return back(form, tf(num, den, { dt }))
  }
  const ss = systems.map((x) => asSs(x, dt))
  return finish(
    form,
    ss.slice(1).reduce((acc, s) => ssSeries(acc, s), ss[0] as StateSpace),
  )
}

/**
 * Systems in parallel (python-control parallel): the same input, outputs added.
 *
 * @param systems - Two or more systems or static gains with equal channels.
 * @returns The parallel connection.
 * @throws WhiteboxError with code `value` for different sample times, `shape` for different channels.
 * @concept block-diagrams
 * @example
 * ```ts
 * import { parallel, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(parallel(tf([1], [1, 1]), 1).num) // => [1, 2]
 * ```
 */
export function parallel(...systems: readonly (TransferFunction | number)[]): TransferFunction
export function parallel(...systems: readonly (ZerosPolesGain | number)[]): ZerosPolesGain
export function parallel(...systems: readonly SystemLike[]): Lti
export function parallel(...systems: readonly SystemLike[]): Lti {
  if (systems.length < 2) throw new WhiteboxError('value', 'parallel: give at least two systems')
  const dt = dtOf(systems, 'parallel')
  const form = formOf(systems)
  if (form !== 'ss' && systems.every(siso)) {
    const gs = systems.map((x) => asTf(x, dt))
    const sum = gs.slice(1).reduce(
      (acc, g) => ({
        num: polyAdd(convolve(acc.num, g.den), convolve(g.num, acc.den)),
        den: convolve(acc.den, g.den),
      }),
      { num: (gs[0] as TransferFunction).num, den: (gs[0] as TransferFunction).den },
    )
    return back(form, tf(sum.num, sum.den, { dt }))
  }
  const ss = systems.map((x) => asSs(x, dt))
  return finish(
    form,
    ss.slice(1).reduce((acc, s) => ssParallel(acc, s), ss[0] as StateSpace),
  )
}

/**
 * The feedback loop (python-control feedback): u = r + sign * G2 y, y = G1 u, so with the default negative sign the
 * closed loop is G1 / (1 + G2 G1). Transfer functions give n1 d2 / (d1 d2 - sign n1 n2) without cancellation.
 *
 * @param sys1 - The forward system.
 * @param sys2 - The feedback system or gain (default 1, unity feedback).
 * @param sign - -1 for negative feedback (default), 1 for positive.
 * @returns The closed loop from r to y.
 * @throws WhiteboxError with code `singular` for an ill-posed algebraic loop (I - sign D1 D2 singular).
 * @concept feedback
 * @example
 * ```ts
 * import { feedback, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(feedback(tf([1], [1, 0])).den) // => [1, 1]
 * ```
 */
export function feedback(sys1: TransferFunction, sys2?: TransferFunction | number, sign?: number): TransferFunction
export function feedback(sys1: ZerosPolesGain, sys2?: ZerosPolesGain | number, sign?: number): ZerosPolesGain
export function feedback(sys1: StateSpace, sys2?: SystemLike, sign?: number): StateSpace
export function feedback(sys1: SystemLike, sys2?: SystemLike, sign?: number): Lti
export function feedback(sys1: SystemLike, sys2: SystemLike = 1, sign = -1): Lti {
  const dt = dtOf([sys1, sys2], 'feedback')
  const form = formOf([sys1, sys2])
  if (form !== 'ss' && siso(sys1) && siso(sys2)) {
    const g1 = asTf(sys1, dt)
    const g2 = asTf(sys2, dt)
    const num = convolve(g1.num, g2.den)
    const den = polyAdd(convolve(g2.den, g1.den), mul(convolve(g2.num, g1.num), -sign))
    return back(form, tf(num, den, { dt }))
  }
  try {
    return finish(form, ssFeedback(asSs(sys1, dt), asSs(sys2, dt), sign))
  } catch (e) {
    if (e instanceof WhiteboxError && e.code === 'value') {
      throw new WhiteboxError('singular', 'feedback: the algebraic loop I - sign D1 D2 is singular')
    }
    throw e
  }
}

/**
 * Systems side by side (python-control append): a block-diagonal state-space system whose inputs and outputs are
 * those of each system in turn.
 *
 * @param systems - The systems or gains.
 * @returns The appended state-space system.
 * @concept block-diagrams
 * @example
 * ```ts
 * import { append, tf } from 'whitebox-engine/systems'
 * append(tf([1], [1, 1]), tf([2], [1, 3])).d.shape // => [2, 2]
 * ```
 */
export function append(...systems: readonly SystemLike[]): StateSpace {
  if (systems.length === 0) throw new WhiteboxError('value', 'append: give at least one system')
  const dt = dtOf(systems, 'append')
  const ss = systems.map((x) => asSs(x, dt))
  return {
    kind: 'ss',
    a: blockDiag(...ss.map((s) => s.a)),
    b: blockDiag(...ss.map((s) => s.b)),
    c: blockDiag(...ss.map((s) => s.c)),
    d: blockDiag(...ss.map((s) => s.d)),
    dt,
  }
}

/**
 * Interconnect the channels of a system (python-control's legacy connect, usually after {@link append}): each row
 * [i, j, k, ...] of Q feeds outputs j, k, ... (negative for a negated output, 0 for none) into input i, all 1-based;
 * the result keeps the listed external inputs and outputs.
 *
 * @param sys - The system holding every block (inputs and outputs numbered from 1).
 * @param q - The interconnection rows.
 * @param inputs - External inputs to keep (1-based).
 * @param outputs - External outputs to keep (1-based, negative to negate).
 * @returns The interconnected state-space system.
 * @throws WhiteboxError with code `value` for an index out of range.
 * @concept block-diagrams
 * @example
 * ```ts
 * import { append, connect, tf, toTf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * // Unity negative feedback around 1/s: input 1 also receives -output 1.
 * toArray(toTf(connect(append(tf([1], [1, 0])), [[1, -1]], [1], [1])).den) // => [1, 1]
 * ```
 */
export function connect(
  sys: Lti,
  q: readonly (readonly number[])[],
  inputs: readonly number[],
  outputs: readonly number[],
): StateSpace {
  const s = toSs(sys)
  const m = s.b.shape[1] ?? 0
  const p = s.c.shape[0] ?? 0
  const k = Array.from({ length: m }, () => new Array<number>(p).fill(0))
  const check = (index: number, n: number, what: string): number => {
    if (!Number.isInteger(index) || index < 1 || index > n)
      throw new WhiteboxError('value', `connect: no ${what} ${index}`)
    return index - 1
  }
  for (const row of q) {
    const i = check(row[0] ?? 0, m, 'input')
    for (const o of row.slice(1)) {
      if (o === 0) continue
      const kRow = k[i] as number[]
      kRow[check(Math.abs(o), p, 'output')] = Math.sign(o)
    }
  }
  const closed = feedback(s, tensor(k), 1)
  const uTrim = tensor(Array.from({ length: m }, (_, r) => inputs.map((u) => (check(u, m, 'input') === r ? 1 : 0))))
  const yTrim = tensor(
    outputs.map((y) => Array.from({ length: p }, (_, c) => (check(Math.abs(y), p, 'output') === c ? Math.sign(y) : 0))),
  )
  return {
    kind: 'ss',
    a: closed.a,
    b: matmul(closed.b, uTrim),
    c: matmul(yTrim, closed.c),
    d: matmul(yTrim, matmul(closed.d, uTrim)),
    dt: closed.dt,
  }
}
