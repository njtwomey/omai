import { WhiteboxError } from '../../base/index.ts'
import { inverse, matVec, solveVector, symmetrise } from '../_dense.ts'
import { allAssignments, type Factor, factor, factorValues } from '../factor.ts'
import { variableElimination } from '../inference/elimination.ts'
import { type BayesNet, type BayesNode, bayesNet } from '../models/bayes-net.ts'
import { doIntervention, type Interventions } from './interventions.ts'
import { type DiscreteAssignment, evaluateAssignment, type Scm } from './scm.ts'

function discreteVariables(m: Scm): DiscreteAssignment[] {
  return m.variables.map((v) => {
    if (v.kind !== 'discrete') throw new WhiteboxError('value', 'counterfactuals: needs a discrete model')
    return v
  })
}

/** Deterministic CPT of X given [...parents, U] from an assignment table. */
function deterministicCpt(v: DiscreteAssignment): number[] {
  const r = v.states.length
  return v.assign.flatMap((x) => Array.from({ length: r }, (_, s) => (s === x ? 1 : 0)))
}

const noiseName = (v: string): string => `U_${v}`
const twinName = (v: string): string => `${v}*`

/**
 * The twin network of a discrete structural causal model under an intervention (Balke and Pearl 1994): the exogenous
 * noises `U_<v>` as root nodes, the factual world `<v>`, and the counterfactual world `<v>*` that shares the noises but
 * applies the intervention. Conditioning on factual evidence and querying starred variables gives counterfactuals.
 *
 * @param m - A discrete model.
 * @param interventions - The counterfactual interventions.
 * @returns The twin Bayes network.
 * @throws WhiteboxError with code `value` for a linear model or unknown states.
 * @concept counterfactuals
 * @example
 * ```ts
 * import { scm, twinNetwork } from 'whitebox-engine/pgm/causal'
 * const m = scm([{ kind: 'discrete', name: 'x', states: ['0', '1'], parents: [], noise: [0.5, 0.5], assign: [0, 1] }])
 * twinNetwork(m, { x: '1' }).nodes.map((n) => n.name) // => ['U_x', 'x', 'x*']
 * ```
 */
export function twinNetwork(m: Scm, interventions: Interventions): BayesNet {
  const vars = discreteVariables(m)
  const cut = discreteVariables(doIntervention(m, interventions) as Scm)
  const nodes: BayesNode[] = []
  for (const [i, v] of vars.entries()) {
    const k = v.noise.length
    nodes.push({
      name: noiseName(v.name),
      states: Array.from({ length: k }, (_, s) => `u${s}`),
      parents: [],
      cpt: v.noise,
    })
    nodes.push({ name: v.name, states: v.states, parents: [...v.parents, noiseName(v.name)], cpt: deterministicCpt(v) })
    const w = cut[i] as DiscreteAssignment
    if (w.name in interventions) {
      nodes.push({ name: twinName(v.name), states: v.states, parents: [], cpt: deterministicCpt(w) })
    } else {
      nodes.push({
        name: twinName(v.name),
        states: v.states,
        parents: [...v.parents.map(twinName), noiseName(v.name)],
        cpt: deterministicCpt(v),
      })
    }
  }
  return bayesNet(nodes)
}

/**
 * A counterfactual distribution P(query_{do(x)} | evidence) of a discrete SCM: what the query would have been had the
 * intervention been made, given what was actually observed, by inference in the twin network.
 *
 * @param m - A discrete model.
 * @param evidence - The factual observations (state names or indices).
 * @param interventions - The counterfactual interventions.
 * @param query - The variables whose counterfactual values are asked about.
 * @returns The distribution over the query's counterfactual values (variables named as in the model).
 * @concept counterfactuals
 * @example
 * ```ts
 * import { scm, counterfactual } from 'whitebox-engine/pgm/causal'
 * import { factorValues } from 'whitebox-engine/pgm'
 * // y copies x unless the noise flips it; seeing x = 1, y = 1 rules the flip out, so had x been 0, y would be 0.
 * const m = scm([
 *   { kind: 'discrete', name: 'x', states: ['0', '1'], parents: [], noise: [0.5, 0.5], assign: [0, 1] },
 *   { kind: 'discrete', name: 'y', states: ['0', '1'], parents: ['x'], noise: [0.9, 0.1], assign: [0, 1, 1, 0] },
 * ])
 * factorValues(counterfactual(m, { x: '1', y: '1' }, { x: '0' }, ['y'])) // => [1, 0]
 * ```
 */
export function counterfactual(
  m: Scm,
  evidence: Readonly<Record<string, string | number>>,
  interventions: Interventions,
  query: readonly string[],
): Factor {
  const twin = twinNetwork(m, interventions)
  const post = variableElimination(twin, query.map(twinName), { evidence }).posterior
  return { ...post, variables: [...query] }
}

/**
 * Abduction: the posterior over the exogenous noises given factual evidence, as one joint factor over the `U_<v>`
 * variables (in model order).
 *
 * @param m - A discrete model.
 * @param evidence - The factual observations.
 * @returns The joint posterior of the noises.
 * @concept counterfactuals
 * @example
 * ```ts
 * import { scm, abduction } from 'whitebox-engine/pgm/causal'
 * import { factorValues } from 'whitebox-engine/pgm'
 * const m = scm([{ kind: 'discrete', name: 'x', states: ['0', '1'], parents: [], noise: [0.5, 0.5], assign: [0, 1] }])
 * factorValues(abduction(m, { x: '1' })) // => [0, 1]
 * ```
 */
export function abduction(m: Scm, evidence: Readonly<Record<string, string | number>>): Factor {
  const twin = twinNetwork(m, {})
  return variableElimination(
    twin,
    m.variables.map((v) => noiseName(v.name)),
    { evidence },
  ).posterior
}

/**
 * A counterfactual by the three steps of Pearl's definition: abduction (the noise posterior given the evidence),
 * action (the intervention), and prediction (evaluating the mutilated model under every noise configuration, weighted
 * by its posterior). Equal to {@link counterfactual} on the twin network.
 *
 * @param m - A discrete model.
 * @param evidence - The factual observations.
 * @param interventions - The counterfactual interventions.
 * @param query - The query variables.
 * @returns The counterfactual distribution over the query.
 * @concept counterfactuals
 * @example
 * ```ts
 * import { scm, abductActPredict } from 'whitebox-engine/pgm/causal'
 * import { factorValues } from 'whitebox-engine/pgm'
 * const m = scm([
 *   { kind: 'discrete', name: 'x', states: ['0', '1'], parents: [], noise: [0.5, 0.5], assign: [0, 1] },
 *   { kind: 'discrete', name: 'y', states: ['0', '1'], parents: ['x'], noise: [0.9, 0.1], assign: [0, 1, 1, 0] },
 * ])
 * factorValues(abductActPredict(m, { x: '1', y: '1' }, { x: '0' }, ['y'])) // => [1, 0]
 * ```
 */
export function abductActPredict(
  m: Scm,
  evidence: Readonly<Record<string, string | number>>,
  interventions: Interventions,
  query: readonly string[],
): Factor {
  const vars = discreteVariables(m)
  const posterior = abduction(m, evidence)
  const acted = doIntervention(m, interventions) as Scm
  const card = query.map((q) => {
    const v = vars.find((x) => x.name === q)
    if (v === undefined) throw new WhiteboxError('value', `unknown variable ${q}`)
    return v.states.length
  })
  const out = new Array<number>(card.reduce((a, b) => a * b, 1)).fill(0)
  const noises = allAssignments(
    vars.map((v) => noiseName(v.name)),
    vars.map((v) => v.noise.length),
  )
  for (const [i, u] of noises.entries()) {
    const w = Math.exp(posterior.logValues[i] ?? Number.NEGATIVE_INFINITY)
    if (w === 0) continue
    const values: Record<string, number> = {}
    for (const v of acted.variables) {
      const a = v as DiscreteAssignment
      values[a.name] =
        a.name in interventions
          ? (a.assign[0] ?? 0)
          : evaluateAssignment(acted, a.name, values, u[noiseName(a.name)] ?? 0)
    }
    let idx = 0
    for (const [k, q] of query.entries()) idx = idx * (card[k] ?? 1) + (values[q] ?? 0)
    out[idx] = (out[idx] ?? 0) + w
  }
  return factor(query, card, out)
}

/**
 * The probabilities of necessity and sufficiency of a binary cause for a binary effect (Pearl 2009, chapter 9):
 * PN = P(Y_{x'} = y' | X = x, Y = y) and PS = P(Y_x = y | X = x', Y = y'), computed exactly from a discrete SCM.
 *
 * @param m - A discrete model.
 * @param cause - The cause and its factual and alternative states.
 * @param effect - The effect and its factual and alternative states.
 * @returns PN and PS.
 * @concept counterfactuals
 * @example
 * ```ts
 * import { scm, causationProbabilities } from 'whitebox-engine/pgm/causal'
 * const m = scm([
 *   { kind: 'discrete', name: 'x', states: ['0', '1'], parents: [], noise: [0.5, 0.5], assign: [0, 1] },
 *   { kind: 'discrete', name: 'y', states: ['0', '1'], parents: ['x'], noise: [0.9, 0.1], assign: [0, 1, 1, 0] },
 * ])
 * causationProbabilities(m, { name: 'x', state: '1', alternative: '0' }, { name: 'y', state: '1', alternative: '0' }) // => { necessity: 1, sufficiency: 1 }
 * ```
 */
export function causationProbabilities(
  m: Scm,
  cause: { readonly name: string; readonly state: string; readonly alternative: string },
  effect: { readonly name: string; readonly state: string; readonly alternative: string },
): { readonly necessity: number; readonly sufficiency: number } {
  const states = discreteVariables(m).find((v) => v.name === effect.name)?.states ?? []
  const pn = factorValues(
    counterfactual(m, { [cause.name]: cause.state, [effect.name]: effect.state }, { [cause.name]: cause.alternative }, [
      effect.name,
    ]),
  )
  const ps = factorValues(
    counterfactual(
      m,
      { [cause.name]: cause.alternative, [effect.name]: effect.alternative },
      { [cause.name]: cause.state },
      [effect.name],
    ),
  )
  return { necessity: pn[states.indexOf(effect.alternative)] ?? 0, sufficiency: ps[states.indexOf(effect.state)] ?? 0 }
}

/**
 * Counterfactuals in a linear SCM with Gaussian noise: abduction conditions the noise vector U on the observed
 * variables (X = (I - W)^{-1} (b + U)), action replaces the intervened assignments, and prediction pushes the noise
 * posterior through the modified model, giving the counterfactual mean and covariance of the query.
 *
 * @param m - A linear model with normal noise.
 * @param evidence - Observed values.
 * @param interventions - Numeric interventions.
 * @param query - The query variables.
 * @returns The counterfactual mean and covariance of the query.
 * @throws WhiteboxError with code `value` for discrete models or non-Gaussian noise.
 * @concept counterfactuals
 * @example
 * ```ts
 * import { scm, linearCounterfactual } from 'whitebox-engine/pgm/causal'
 * const m = scm([
 *   { kind: 'linear', name: 'x', parents: [], weights: [], intercept: 0, noise: { distribution: 'normal', scale: 1 } },
 *   { kind: 'linear', name: 'y', parents: ['x'], weights: [2], intercept: 0, noise: { distribution: 'normal', scale: 1 } },
 * ])
 * // Observed x = 1, y = 3 (so U_y = 1); had x been 2, y would have been 2 * 2 + 1.
 * linearCounterfactual(m, { x: 1, y: 3 }, { x: 2 }, ['y']).mean.map((v) => Math.round(v * 1e6) / 1e6) // => [5]
 * ```
 */
export function linearCounterfactual(
  m: Scm,
  evidence: Readonly<Record<string, number>>,
  interventions: Readonly<Record<string, number>>,
  query: readonly string[],
): { readonly mean: number[]; readonly covariance: number[][] } {
  const names = m.variables.map((v) => v.name)
  const n = names.length
  const lin = m.variables.map((v) => {
    if (v.kind !== 'linear' || v.noise.distribution !== 'normal')
      throw new WhiteboxError('value', 'linearCounterfactual: needs linear assignments with normal noise')
    return v
  })
  const reduced = (cut: boolean): { L: number[][]; b: number[] } => {
    // X = b + W X + U  =>  X = (I - W)^{-1} (b + U); an intervened row becomes X_i = value (no parents, no noise).
    const IW = names.map((_, i) => names.map((_, j) => (i === j ? 1 : 0)))
    const b = lin.map((v) => v.intercept)
    for (const [i, v] of lin.entries()) {
      if (cut && v.name in interventions) {
        b[i] = interventions[v.name] ?? 0
        continue
      }
      for (const [k, p] of v.parents.entries()) (IW[i] as number[])[names.indexOf(p)] = -(v.weights[k] ?? 0)
    }
    return { L: inverse(IW), b }
  }
  const sigma = lin.map((v) => Math.max(v.noise.scale ** 2, 1e-300))
  // Abduction: U | X_o, with X_o = L_o (b + U).
  const { L, b } = reduced(false)
  const obs = names.map((_, i) => i).filter((i) => (names[i] ?? '') in evidence)
  const Lo = obs.map((i) => L[i] ?? [])
  const mu0 = matVec(L, b)
  const Sxx = Lo.map((ri) => Lo.map((rj) => ri.reduce((s, v, k) => s + v * (rj[k] ?? 0) * (sigma[k] ?? 0), 0) + 1e-12))
  const resid = obs.map((i) => (evidence[names[i] ?? ''] ?? 0) - (mu0[i] ?? 0))
  const alpha = solveVector(Sxx, resid)
  const SxxInv = inverse(Sxx)
  // Cov(U, X_o) = Sigma L_o^T.
  const SuX = Array.from({ length: n }, (_, k) => Lo.map((r) => (sigma[k] ?? 0) * (r[k] ?? 0)))
  const uMean = SuX.map((row) => row.reduce((s, v, j) => s + v * (alpha[j] ?? 0), 0))
  const uCov = Array.from({ length: n }, (_, a) =>
    Array.from({ length: n }, (_, c) => {
      let s = a === c ? (sigma[a] ?? 0) : 0
      for (let i = 0; i < obs.length; i++)
        for (let j = 0; j < obs.length; j++) s -= (SuX[a]?.[i] ?? 0) * (SxxInv[i]?.[j] ?? 0) * (SuX[c]?.[j] ?? 0)
      return s
    }),
  )
  // Action and prediction: X* = L* (b* + D U), D zeroing the noise of intervened variables.
  const act = reduced(true)
  const keep = names.map((v) => (v in interventions ? 0 : 1))
  const A = act.L.map((row) => row.map((v, k) => v * (keep[k] ?? 1)))
  const meanAll = act.L.map(
    (row, i) =>
      row.reduce((s, v, k) => s + v * (act.b[k] ?? 0), 0) +
      (A[i] ?? []).reduce((s, v, k) => s + v * (uMean[k] ?? 0), 0),
  )
  const qi = query.map((q) => {
    const i = names.indexOf(q)
    if (i < 0) throw new WhiteboxError('value', `unknown variable ${q}`)
    return i
  })
  const covariance = qi.map((i) =>
    qi.map((j) => {
      let s = 0
      for (let a = 0; a < n; a++)
        for (let c = 0; c < n; c++) s += (A[i]?.[a] ?? 0) * (uCov[a]?.[c] ?? 0) * (A[j]?.[c] ?? 0)
      return s
    }),
  )
  return { mean: qi.map((i) => meanAll[i] ?? 0), covariance: symmetrise(covariance) }
}
