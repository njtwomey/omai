/**
 * Preference objectives for aligning a model to human (or model) judgements: the Bradley-Terry pairwise loss of a
 * reward model, and the direct objectives that skip the reward model, working on sequence log-probabilities of the
 * policy and of a frozen reference policy: DPO (and its conservative variant), IPO, KTO for unpaired good and bad
 * answers, ORPO and SimPO, which need no reference. All take log-probabilities of whole responses (sum the token
 * log-probabilities with `sequenceLogProb`) and are differentiable in the policy's.
 */

import { MdmlError, type Reduction } from '../base/index.ts'
import { log1mexp, logSigmoid } from '../numerics/special/index.ts'
import {
  add,
  cast,
  greater,
  maximum,
  mean,
  mul,
  neg,
  promoteFloat,
  sigmoid,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { reduce } from './_reduce.ts'

/**
 * The log-probability of a response from its per-token log-probabilities: their sum over the tokens that are not
 * padding.
 *
 * @param tokenLogProbs - Log-probabilities of the response tokens `[B, T]`.
 * @param mask - 1 for real tokens, 0 for padding `[B, T]` (default all real).
 * @returns The response log-probabilities `[B]`; divide by the token count for the length-normalised form.
 * @example
 * ```ts
 * import { sequenceLogProb } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(sequenceLogProb([[-1, -2, -3], [-1, -1, -1]], [[1, 1, 0], [1, 1, 1]])) // => [-3, -3]
 * ```
 */
export function sequenceLogProb(tokenLogProbs: TensorLike, mask?: TensorLike): Value {
  const [lp] = promoteFloat(tokenLogProbs) as [Value]
  return sum(mask === undefined ? lp : mul(lp, promoteFloat(mask)[0] as Value), -1)
}

/**
 * The Bradley-Terry pairwise loss of a reward model (Christiano et al., 2017; Ouyang et al., 2022): the chosen answer
 * beats the rejected one with probability sigmoid(r_chosen - r_rejected), and the loss is its negative log, optionally
 * with a required margin (Llama 2's margin loss).
 *
 * @param rewardChosen - Rewards of the preferred responses `[B]`.
 * @param rewardRejected - Rewards of the rejected responses `[B]`.
 * @param options - margin (default 0) and reduction.
 * @returns The loss, differentiable in the rewards.
 * @concept preference-optimisation
 * @example
 * ```ts
 * import { bradleyTerry } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(bradleyTerry([1], [1])) === Math.LN2 // => true
 * ```
 */
export function bradleyTerry(
  rewardChosen: TensorLike,
  rewardRejected: TensorLike,
  options: { readonly margin?: number; readonly reduction?: Reduction } = {},
): Value {
  const [c, r] = promoteFloat(rewardChosen, rewardRejected) as [Value, Value]
  return reduce(neg(logSigmoid(sub(sub(c, r), options.margin ?? 0))), { reduction: options.reduction ?? 'mean' })
}

/** The log-probabilities of a batch of preference pairs under the policy and (where needed) the reference. */
export interface PreferenceBatch {
  readonly policyChosen: TensorLike
  readonly policyRejected: TensorLike
  readonly referenceChosen?: TensorLike
  readonly referenceRejected?: TensorLike
}

/** The log-ratio of chosen to rejected under the policy, minus the same under the reference (zero if there is none). */
function logitsOf(batch: PreferenceBatch): Value {
  const [pc, pr] = promoteFloat(batch.policyChosen, batch.policyRejected) as [Value, Value]
  const policy = sub(pc, pr)
  if (batch.referenceChosen === undefined || batch.referenceRejected === undefined) return policy
  const [rc, rr] = promoteFloat(batch.referenceChosen, batch.referenceRejected) as [Value, Value]
  return sub(policy, sub(rc, rr))
}

/**
 * Direct preference optimisation (Rafailov et al., 2023): -log sigmoid(beta [(log pi(y_w) - log pi_ref(y_w)) -
 * (log pi(y_l) - log pi_ref(y_l))]), the Bradley-Terry loss with the reward taken to be beta times the policy's
 * log-ratio to the reference. With a label-smoothing epsilon it is the conservative DPO (cDPO), which assumes the label
 * is wrong with probability epsilon. Without reference log-probabilities it is the reference-free form.
 *
 * @param batch - Policy and reference log-probabilities of the chosen and rejected responses `[B]`.
 * @param options - beta (default 0.1), labelSmoothing (default 0) and reduction.
 * @returns The loss, differentiable in the policy's log-probabilities.
 * @example
 * ```ts
 * import { dpo } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * // A policy equal to the reference cannot tell the answers apart: log 2.
 * item(dpo({ policyChosen: [-5], policyRejected: [-7], referenceChosen: [-5], referenceRejected: [-7] })) === Math.LN2 // => true
 * ```
 */
export function dpo(
  batch: PreferenceBatch,
  options: { readonly beta?: number; readonly labelSmoothing?: number; readonly reduction?: Reduction } = {},
): Value {
  const z = mul(logitsOf(batch), options.beta ?? 0.1)
  const eps = options.labelSmoothing ?? 0
  const terms = add(mul(neg(logSigmoid(z)), 1 - eps), mul(neg(logSigmoid(neg(z))), eps))
  return reduce(terms, { reduction: options.reduction ?? 'mean' })
}

/** The implicit rewards a DPO-trained policy assigns, beta times its log-ratio to the reference. */
export interface ImplicitRewards {
  readonly chosen: Value
  readonly rejected: Value
  /** chosen minus rejected. */
  readonly margin: Value
  /** The share of pairs with a positive margin. */
  readonly accuracy: Value
}

/**
 * The implicit rewards of DPO, r(y) = beta log(pi(y) / pi_ref(y)), their margin and the share of pairs the policy
 * ranks correctly: the quantities one logs while training.
 *
 * @param batch - Policy and reference log-probabilities (both references required) `[B]`.
 * @param options - beta (default 0.1).
 * @returns The chosen and rejected rewards, the margin and the accuracy.
 * @throws MdmlError with code `value` when the reference log-probabilities are missing.
 * @example
 * ```ts
 * import { implicitRewards } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * const r = implicitRewards({ policyChosen: [-4], policyRejected: [-8], referenceChosen: [-5], referenceRejected: [-7] }, { beta: 1 })
 * toArray(r.margin) // => [2]
 * ```
 */
export function implicitRewards(batch: PreferenceBatch, options: { readonly beta?: number } = {}): ImplicitRewards {
  if (batch.referenceChosen === undefined || batch.referenceRejected === undefined)
    throw new MdmlError('value', 'implicitRewards: the reference log-probabilities are required')
  const [pc, pr, rc, rr] = promoteFloat(
    batch.policyChosen,
    batch.policyRejected,
    batch.referenceChosen,
    batch.referenceRejected,
  ) as [Value, Value, Value, Value]
  const beta = options.beta ?? 0.1
  const chosen = mul(sub(pc, rc), beta)
  const rejected = mul(sub(pr, rr), beta)
  const margin = sub(chosen, rejected)
  return { chosen, rejected, margin, accuracy: mean(cast(greater(margin, 0), 'float64')) }
}

/**
 * Identity preference optimisation (Azar et al., 2023): the squared distance of the policy-versus-reference log-ratio
 * margin to 1 / (2 tau), a regression target instead of DPO's sigmoid, so the margin cannot grow without bound when
 * preferences are deterministic. The regularisation strength tau plays the role of beta (smaller tau, weaker pull to the
 * reference).
 *
 * @param batch - Policy and reference log-probabilities `[B]`.
 * @param options - tau (default 0.1) and reduction.
 * @returns The loss, differentiable in the policy's log-probabilities.
 * @example
 * ```ts
 * import { ipo } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(ipo({ policyChosen: [0], policyRejected: [0] }, { tau: 0.5 })) // => 1
 * ```
 */
export function ipo(
  batch: PreferenceBatch,
  options: { readonly tau?: number; readonly reduction?: Reduction } = {},
): Value {
  const target = 1 / (2 * (options.tau ?? 0.1))
  const h = sub(logitsOf(batch), target)
  return reduce(mul(h, h), { reduction: options.reduction ?? 'mean' })
}

/**
 * Kahneman-Tversky optimisation (Ethayarajh et al., 2024): a prospect-theoretic loss on single answers labelled good or
 * bad, with no need for pairs. With the implicit reward r = log(pi / pi_ref) and a reference point z0 (an estimate of
 * the policy's KL from the reference), a desirable answer costs lambda_D (1 - sigmoid(beta (r - z0))) and an
 * undesirable one lambda_U (1 - sigmoid(beta (z0 - r))).
 *
 * @param policy - Policy log-probabilities of the responses `[B]`.
 * @param reference - Reference log-probabilities `[B]`.
 * @param desirable - 1 for good answers, 0 for bad `[B]`.
 * @param options - beta (default 0.1), the weights lambdaD and lambdaU (default 1), the reference point z0 (default
 *   the batch mean of r, clamped at zero and held constant, a biased but standard estimate) and reduction.
 * @returns The loss, differentiable in the policy's log-probabilities.
 * @example
 * ```ts
 * import { kto } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * // r = 0 and z0 = 0: sigmoid(0) = 1/2 for both kinds of answer.
 * item(kto([-3, -4], [-3, -4], [1, 0], { referencePoint: 0 })) // => 0.5
 * ```
 */
export function kto(
  policy: TensorLike,
  reference: TensorLike,
  desirable: TensorLike,
  options: {
    readonly beta?: number
    readonly lambdaD?: number
    readonly lambdaU?: number
    readonly referencePoint?: number
    readonly reduction?: Reduction
  } = {},
): Value {
  const [p, r, d] = promoteFloat(policy, reference, desirable) as [Value, Value, Value]
  const beta = options.beta ?? 0.1
  const ratio = sub(p, r)
  const z0 = options.referencePoint ?? maximum(stopGradient(mean(ratio)), 0)
  const good = sub(1, sigmoid(mul(sub(ratio, z0), beta)))
  const bad = sub(1, sigmoid(mul(sub(z0, ratio), beta)))
  const terms = where(greater(d, 0.5), mul(good, options.lambdaD ?? 1), mul(bad, options.lambdaU ?? 1))
  return reduce(terms, { reduction: options.reduction ?? 'mean' })
}

/**
 * Odds-ratio preference optimisation (Hong et al., 2024): the supervised loss on the chosen answer plus lambda times
 * -log sigmoid(log odds(chosen) - log odds(rejected)), with odds(y) = P(y) / (1 - P(y)) of the length-normalised
 * likelihood. One stage, no reference model.
 *
 * @param logProbChosen - Length-normalised (mean per token) log-probability of the chosen responses `[B]`, below zero.
 * @param logProbRejected - The same for the rejected responses `[B]`.
 * @param options - lambda (default 0.1) and reduction.
 * @returns The loss: negative log-likelihood of the chosen answer plus lambda times the odds-ratio term.
 * @example
 * ```ts
 * import { orpo } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * // odds 1 against 1/3: the odds-ratio term is log(1 + 1/3); the likelihood term is log 2.
 * Math.abs(item(orpo([Math.log(0.5)], [Math.log(0.25)], { lambda: 1 })) - (Math.LN2 + Math.log(4 / 3))) < 1e-14 // => true
 * ```
 */
export function orpo(
  logProbChosen: TensorLike,
  logProbRejected: TensorLike,
  options: { readonly lambda?: number; readonly reduction?: Reduction } = {},
): Value {
  const [c, r] = promoteFloat(logProbChosen, logProbRejected) as [Value, Value]
  const logOdds = (lp: Value): Value => sub(lp, log1mexp(lp))
  const ratioTerm = neg(logSigmoid(sub(logOdds(c), logOdds(r))))
  return reduce(add(neg(c), mul(ratioTerm, options.lambda ?? 0.1)), { reduction: options.reduction ?? 'mean' })
}

/**
 * Simple preference optimisation (Meng et al., 2024): reference-free DPO on the length-normalised log-likelihood with a
 * target margin gamma, -log sigmoid(beta (log pi(y_w) / |y_w| - log pi(y_l) / |y_l|) - gamma), which matches how the
 * model generates (by average, not total, likelihood).
 *
 * @param logProbChosen - Length-normalised log-probability of the chosen responses `[B]`.
 * @param logProbRejected - The same for the rejected responses `[B]`.
 * @param options - beta (default 2), gamma (default 0.5) and reduction.
 * @returns The loss, differentiable in the log-probabilities.
 * @example
 * ```ts
 * import { simpo } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(simpo([-1], [-1], { beta: 1, gamma: 0 })) === Math.LN2 // => true
 * ```
 */
export function simpo(
  logProbChosen: TensorLike,
  logProbRejected: TensorLike,
  options: { readonly beta?: number; readonly gamma?: number; readonly reduction?: Reduction } = {},
): Value {
  const [c, r] = promoteFloat(logProbChosen, logProbRejected) as [Value, Value]
  const z = sub(mul(sub(c, r), options.beta ?? 2), options.gamma ?? 0.5)
  return reduce(neg(logSigmoid(z)), { reduction: options.reduction ?? 'mean' })
}
