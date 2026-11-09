import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { type LinearPenalty, type LinearSvcLoss, type LinearSvrLoss, linearSvc, linearSvr } from './linear.ts'

type Params = Readonly<Record<string, unknown>>

const svc = (inputs: readonly (Tensor | undefined)[], o: Params, solver?: string) => {
  const [x, y, w] = inputs as Tensor[]
  const balanced = o['class_weight'] === 'balanced'
  return linearSvc
    .with({
      C: o['C'] as number,
      loss: o['loss'] === 'hinge' ? 'hinge' : ('squaredHinge' as LinearSvcLoss),
      penalty: (o['penalty'] ?? 'l2') as LinearPenalty,
      fitIntercept: (o['fit_intercept'] ?? true) as boolean,
      interceptScaling: (o['intercept_scaling'] ?? 1) as number,
      classWeight: balanced ? 'balanced' : null,
      tol: 1e-10,
      maxIter: 200_000,
      ...(solver === undefined ? {} : { solver: { kind: solver } as never }),
    })
    .fit({ x: x as Tensor, y: y as Tensor, ...(w === undefined ? {} : { weights: w }) })
}

const svr = (inputs: readonly (Tensor | undefined)[], o: Params) => {
  const [x, y] = inputs as Tensor[]
  return linearSvr
    .with({
      C: o['C'] as number,
      epsilon: o['epsilon'] as number,
      loss: (o['loss'] === 'epsilon_insensitive' ? 'epsilonInsensitive' : 'squaredEpsilonInsensitive') as LinearSvrLoss,
      tol: 1e-10,
      maxIter: 200_000,
    })
    .fit({ x: x as Tensor, y: y as Tensor })
}

goldenSuite('LinearSVC and LinearSVR against scikit-learn (liblinear)', 'svm/linear', {
  coef: (i, o) => tensor(svc(i, o).params.coef),
  // scikit-learn's intercept_ is the scalar 0.0, not an array, when no intercept is fitted.
  intercept: (i, o) => (o['fit_intercept'] === false ? 0 : tensor(svc(i, o).params.intercept)),
  decision: (i, o) => svc(i, o).decisionFunction(i[0] as Tensor),
  coefWeighted: (i, o) => tensor(svc(i, o).params.coef),
  interceptWeighted: (i, o) => tensor(svc(i, o).params.intercept),
  svrCoef: (i, o) => tensor(svr(i, o).params.coef),
  svrIntercept: (i, o) => tensor([svr(i, o).params.intercept]),
})
