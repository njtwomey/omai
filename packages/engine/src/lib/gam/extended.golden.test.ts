import { learnGolden, num, str } from '../../testing/learn.ts'
import type { ExtendedFamily } from '../dists/families/index.ts'
import { families } from '../dists/index.ts'
import { dsyev } from '../numerics/linalg/index.ts'
import { coxData, coxProblem } from '../survival/index.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import {
  differencedLikelihood,
  EXTENDED_DEFAULTS,
  type ExtendedModel,
  etaLikelihood,
  evaluateExtended,
  fitExtended,
} from './extended.ts'

const FAMILIES: Record<string, () => ExtendedFamily> = {
  beta: () => families.betaRegression(),
  negativeBinomial: () => families.negativeBinomialTheta(),
  gaussianLocationScale: () => families.gaussianLocationScale(),
  zeroInflatedPoisson: () => families.zeroInflatedPoissonLogit(),
  orderedCategorical: () => families.orderedCategorical(4),
  multinomial: () => families.multinomialLogit(3),
  gamlssGamma: () => families.gamlssExtended('gamma'),
  gamlssStudentT: () => families.gamlssExtended('studentT'),
}

const flat = (t: Tensor): Float64Array => Float64Array.from(toFlat(t))

/** The structure of one smooth block spanning all the coefficients, the way the oracle stacks them. */
function structureOf(S: Float64Array, P: number): ExtendedModel['structure'] {
  const { w } = dsyev(S, P)
  const top = Math.max(...Array.from(w))
  const rank = Array.from(w).filter((v) => v > 1e-9 * top).length
  return {
    p: P,
    blocks: [{ offset: 0, size: P, term: 0, matrix: S }],
    terms: [{ offset: 0, size: P, blocks: [0], rank }],
  }
}

function modelOf(inputs: readonly Tensor[], params: Readonly<Record<string, unknown>>): ExtendedModel {
  const family = (FAMILIES[str(params, 'family')] as () => ExtendedFamily)()
  const y = flat(inputs[0] as Tensor)
  const n = y.length
  const designs = inputs.slice(1, -1).map((d) => ({ X: flat(d), p: (d.shape[1] as number) ?? 1 }))
  const S = flat(inputs[inputs.length - 1] as Tensor)
  const P = designs.reduce((a, d) => a + d.p, 0)
  const constants = Array.from(toFlat(params['constants'] as Tensor))
  const start = new Float64Array(P)
  let at = 0
  designs.forEach((d, k) => {
    start[at] = constants[k] ?? 0
    at += d.p
  })
  const w = new Float64Array(n).fill(1)
  const aux0 = Array.from(toFlat(params['aux0'] as Tensor))
  return {
    n,
    likelihood: etaLikelihood(family, y, w, designs),
    structure: structureOf(S, P),
    predictors: [],
    terms: [],
    start,
    aux: family.aux(y, w).map((a, i) => ({ ...a, start: aux0[i] ?? a.start })),
    family,
  }
}

function coxModelOf(inputs: readonly Tensor[]): ExtendedModel {
  const [time, event, design, penalty] = inputs as [Tensor, Tensor, Tensor, Tensor]
  const p = design.shape[1] as number
  const X = flat(design)
  const problem = coxProblem(coxData(flat(time), flat(event), fromStorage(X, [flat(time).length, p], 'float64')), {
    ties: 'breslow',
  })
  return {
    n: flat(time).length,
    likelihood: differencedLikelihood(p, (beta) => ({
      logLik: problem.logLikelihood(beta),
      gradient: Float64Array.from(problem.score(beta)),
      information: problem.information(beta),
    })),
    structure: structureOf(flat(penalty), p),
    predictors: [],
    terms: [],
    start: new Float64Array(p),
    aux: [],
  }
}

const fixed = (model: ExtendedModel, logLambda: number, aux: readonly number[]) => {
  const ev = evaluateExtended(model, [logLambda], aux, { gradient: false })
  return { coef: Array.from(ev.fit.beta), logLik: [ev.fit.logLik], edf: [ev.edf], criterion: [ev.value] }
}

const selected = (model: ExtendedModel) => {
  const fit = fitExtended(model, { ...EXTENDED_DEFAULTS, maxIter: 100 })
  if (!fit.converged) throw new Error('the extended outer iteration did not converge')
  const ev = fit.evaluation
  return {
    logLambda: fit.state.logLambda,
    aux: fit.state.aux,
    criterion: [ev.value],
    edf: [ev.edf],
    logLik: [ev.fit.logLik],
  }
}

// The reference (gam/extended) is numpy and scipy: scipy's densities, five-point finite differences for the derivatives
// of the log-likelihood in the predictors, Newton for the penalised coefficients, and a grid then Nelder-Mead for the
// smoothing parameter and the family parameters. The engine uses autodiff derivatives (or hand-written ones) and Newton
// steps on the analytic gradient of the Laplace score.
learnGolden('extended-family additive models against a numpy and scipy reference', 'gam/extended', {
  extendedFixed: (inputs, p) => fixed(modelOf(inputs, p), num(p, 'logLambda'), Array.from(toFlat(p['aux'] as Tensor))),
  extendedSelected: (inputs, p) => selected(modelOf(inputs, p)),
  extendedCoxFixed: (inputs, p) => fixed(coxModelOf(inputs), num(p, 'logLambda'), []),
  extendedCoxSelected: (inputs) => selected(coxModelOf(inputs)),
})
