/**
 * The pieces the kernel families are built from: the window-adapted plan of the kernels whose tuned parameters are a step
 * size and an inverse mass matrix (Hamiltonian, Langevin and Barker kernels, the random walk when tempered), and the
 * plan of a kernel with parameters fixed by its knobs. Internal to the folder.
 */

import type { Json } from '../../base/index.ts'
import {
  type HamiltonianParameters,
  type MCMCKernel,
  type MCMCState,
  massMatrixInit,
  type WarmupState,
  windowAdaptationAlgorithm,
  windowAdaptationResult,
} from '../../sampling/index.ts'
import { bool, type DictSpace, dict, real } from '../../space/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'
import type { Config } from '../../trace/index.ts'
import { stateFromJson, stateToJson } from '../../trace/index.ts'
import type { McmcBuild, McmcPlan, Warmer } from './_mcmc-driver.ts'
import { flag, num } from './_support.ts'

/** The inverse mass matrix of tuned parameters in the form a kernel option takes: a vector, or a `[dim, dim]` matrix. */
export function massOf(p: HamiltonianParameters, dim: number): Float64Array | Tensor {
  return dim > 1 && p.inverseMassMatrix.length === dim * dim
    ? (fromStorage(Float64Array.from(p.inverseMassMatrix), [dim, dim], 'float64') as Tensor)
    : p.inverseMassMatrix
}

/** The diagonal of an inverse mass matrix (the diagonal of a dense one). */
export function massDiagonal(p: HamiltonianParameters, dim: number): Float64Array {
  if (p.inverseMassMatrix.length !== dim * dim || dim === 1) return p.inverseMassMatrix
  return Float64Array.from({ length: dim }, (_, i) => p.inverseMassMatrix[i * dim + i] ?? 1)
}

/** The knobs of window adaptation a family exposes: the acceptance rate it aims for and, optionally, the dense mass matrix. */
export function windowKnobs(targetAcceptance: number, dense: boolean): DictSpace {
  return dict({
    targetAcceptance: real(0.05, 0.99, { default: targetAcceptance, label: 'target acceptance' }),
    ...(dense ? { dense: bool({ default: false, label: 'dense mass matrix' }) } : {}),
  })
}

/** Stan's windowed warm-up of a kernel family as a `Warmer`. */
export function windowWarmer(
  dim: number,
  kernelFor: (p: HamiltonianParameters) => MCMCKernel,
  options: { readonly numWarmup: number; readonly dense: boolean; readonly targetAcceptance: number },
): Warmer<HamiltonianParameters> {
  // Stepping and serialising a warm-up state do not depend on the starting position, so one algorithm serves every chain.
  const shared = windowAdaptationAlgorithm(dim, kernelFor, new Float64Array(dim), options)
  return {
    init: (start, key) => windowAdaptationAlgorithm(dim, kernelFor, start, options).init(key),
    step: (w, ctx) => shared.step(w as WarmupState<MCMCState>, ctx),
    chain: (w) => (w as WarmupState).chain,
    info: (w) => (w as WarmupState).info,
    result: (w) => windowAdaptationResult(w as WarmupState),
    toJson: (w) => shared.serialise?.(w as WarmupState) ?? null,
    fromJson: (j) => shared.deserialise?.(j) as WarmupState,
    windowEnded: (w) => {
      const s = w as WarmupState
      return s.windowEnded ? { stepSize: s.stepSize, inverseMassMatrix: Array.from(s.mm.inverseMassMatrix) } : undefined
    },
  }
}

/**
 * The plan of a kernel whose parameters are a step size and an inverse mass matrix: window-adapted for `numWarmup` iterations
 * (dual averaging of the step size to `targetAcceptance`, Stan's windows for the mass matrix, diagonal or dense), then sampled
 * with the tuned parameters.
 *
 * @param b - What `plan` is given.
 * @param kernelFor - The kernel at tuned parameters.
 * @param defaults - The acceptance rate to aim for when the knob is absent.
 * @returns The plan.
 */
export function windowPlan(
  b: McmcBuild,
  kernelFor: (p: HamiltonianParameters) => MCMCKernel,
  defaults: { readonly targetAcceptance: number },
): McmcPlan<HamiltonianParameters> {
  const dim = b.um.dim
  const c: Config = b.config
  const dense = flag(c, 'dense')
  const targetAcceptance = c['targetAcceptance'] === undefined ? defaults.targetAcceptance : num(c, 'targetAcceptance')
  return {
    kernelDim: dim,
    walkers: 1,
    kernel: kernelFor,
    tuned0: { stepSize: 1, inverseMassMatrix: massMatrixInit(dim, { dense }).inverseMassMatrix },
    warm: windowWarmer(dim, kernelFor, { numWarmup: num(c, 'numWarmup'), dense, targetAcceptance }),
    tunedToJson: (t) => stateToJson(t),
    tunedFromJson: (j) => stateFromJson(j) as HamiltonianParameters,
    tunedSummary: (t) => ({ stepSize: t.stepSize, inverseMassMatrix: Array.from(t.inverseMassMatrix) }) as Json,
  }
}

/** The plan of a kernel with no adaptation: its parameters are what the knobs say, warm-up is burn-in. */
export function fixedPlan<T>(
  b: McmcBuild,
  parameters: T,
  kernel: (t: T) => MCMCKernel,
  summary: (t: T) => Json,
): McmcPlan<T> {
  return {
    kernelDim: b.um.dim,
    walkers: 1,
    kernel,
    tuned0: parameters,
    tunedToJson: (t) => stateToJson(t),
    tunedFromJson: (j) => stateFromJson(j) as T,
    tunedSummary: summary,
  }
}
