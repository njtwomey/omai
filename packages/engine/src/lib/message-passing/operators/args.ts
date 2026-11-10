/**
 * What every operator reads its arguments through: the argument of a message or of an observed value, its Gaussian
 * moments (a constant is a Gaussian of variance zero), the typed reads of a factor's JSON parameters, and the error an
 * operator throws when a message it needs has no density. The operators of the other files are built from these.
 */

import { OmaiError } from '../../base/index.ts'
import type { Arg, Eta, Family, Observed, Params } from '../types.ts'

/** Thrown by an operator that cannot form a message because a quantity it needs is improper; the engine counts it and repairs. */
export class ImproperMessageError extends OmaiError {
  constructor(message: string) {
    super('value', message)
    this.name = 'ImproperMessageError'
  }
}

/**
 * An argument that carries a message.
 *
 * @param family - The family of the variable.
 * @param eta - The incoming message.
 * @returns The argument.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg } from 'omai-engine/message-passing'
 * messageArg(gaussian, gaussianEta(0, 1)).value // => undefined
 * ```
 */
export function messageArg(family: Family, eta: Eta): Arg {
  return { family, eta, value: undefined }
}

/**
 * An argument that is observed (or a constant).
 *
 * @param family - The family of the variable.
 * @param value - Its value: a number, or an array for a vector.
 * @returns The argument.
 * @example
 * ```ts
 * import { gaussian, valueArg } from 'omai-engine/message-passing'
 * valueArg(gaussian, 3).value // => 3
 * ```
 */
export function valueArg(family: Family, value: Observed): Arg {
  return { family, eta: undefined, value }
}

/** Whether an argument is observed. */
export const isObserved = (a: Arg): boolean => a.value !== undefined

/** The message of an argument that has one. */
export function etaOf(a: Arg): Eta {
  if (a.eta === undefined) throw new OmaiError('value', 'operator: expected a message, the argument is observed')
  return a.eta
}

/** The scalar value of an observed argument. */
export function scalarOf(a: Arg): number {
  const v = a.value
  if (typeof v !== 'number') throw new OmaiError('value', 'operator: expected an observed number')
  return v
}

/** The vector value of an observed argument. */
export function vectorOf(a: Arg): readonly number[] {
  const v = a.value
  if (v === undefined || typeof v === 'number') throw new OmaiError('value', 'operator: expected an observed vector')
  return v
}

/**
 * The mean and variance of a Gaussian argument: a message's moments, or an observed value with variance zero.
 *
 * @param a - The argument.
 * @returns Mean and variance.
 * @throws ImproperMessageError when the message has no density.
 */
export function gaussianOf(a: Arg): { readonly mean: number; readonly variance: number } {
  if (a.value !== undefined) return { mean: scalarOf(a), variance: 0 }
  const eta = etaOf(a)
  const j = eta[1] ?? 0
  if (!(j > 0) || !Number.isFinite(j))
    throw new ImproperMessageError('a Gaussian message has no density (its precision is not positive)')
  return { mean: (eta[0] ?? 0) / j, variance: 1 / j }
}

/** Whether every message among the arguments has a density (observed arguments always do). */
export const allProper = (args: readonly Arg[]): boolean =>
  args.every((a) => a.eta === undefined || a.family.isProper(a.eta))

/** log of the mass of an argument's unnormalised message exp(eta . T): its log-partition, or 0 for an observed value. */
export const massOf = (a: Arg): number => (a.eta === undefined ? 0 : a.family.logPartition(a.eta))

/** A number parameter. */
export function numberParam(params: Params, name: string, fallback?: number): number {
  const v = params[name]
  if (typeof v === 'number') return v
  if (v === undefined && fallback !== undefined) return fallback
  throw new OmaiError('value', `operator: the parameter ${name} must be a number`)
}

/** A vector parameter. */
export function vectorParam(params: Params, name: string): readonly number[] {
  const v = params[name]
  if (Array.isArray(v) && v.every((x) => typeof x === 'number')) return v as readonly number[]
  throw new OmaiError('value', `operator: the parameter ${name} must be an array of numbers`)
}

/** The Gaussian message of a given mean and variance; the variance must be positive. */
export function gaussianMessage(mean: number, variance: number): Eta {
  if (!(variance > 0) || !Number.isFinite(variance) || !Number.isFinite(mean)) {
    throw new ImproperMessageError(`a Gaussian message of mean ${mean} and variance ${variance} has no density`)
  }
  return [mean / variance, 1 / variance]
}

/** log N(x; mean, variance). */
export const logNormal = (x: number, mean: number, variance: number): number =>
  -0.5 * (Math.log(2 * Math.PI * variance) + (x - mean) ** 2 / variance)

/** The Gaussian message that turns a cavity into a tilted distribution of the given mean and variance: their quotient in natural parameters. */
export function gaussianSiteMessage(cavity: Eta, mean: number, variance: number): Eta {
  if (!(variance > 0) || !Number.isFinite(variance) || !Number.isFinite(mean)) {
    throw new ImproperMessageError(`the tilted Gaussian of mean ${mean} and variance ${variance} has no density`)
  }
  return [mean / variance - (cavity[0] ?? 0), 1 / variance - (cavity[1] ?? 0)]
}

/** The message that turns a cavity into the member of the family with the given moments: the quotient of the projection by the cavity. */
export function siteMessage(family: Family, cavity: Eta, moments: Eta): Eta {
  const projected = family.fromMoments(moments)
  return projected.map((v, i) => v - (cavity[i] ?? 0))
}

/** The sum of two messages without any renormalisation: the product of the unnormalised messages, for the masses the evidence needs. */
export const rawSum = (a: Eta, b: Eta): Eta => a.map((v, i) => v + (b[i] ?? 0))
