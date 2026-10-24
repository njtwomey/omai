import { gaussianNB, learnGolden, logisticRegression, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import {
  betaCalibration,
  type CalibrationMethod,
  calibratedClassifierCV,
  histogramBinning,
  isotonicCalibration,
  plattScaling,
  temperatureScaling,
  vectorScaling,
} from './calibration.ts'
import { stratifiedKFold } from './model-selection/splitters.ts'

learnGolden('calibration against sklearn.calibration', 'learn/calibration', {
  platt: ([s, y, t]) => {
    const c = plattScaling(s as Tensor, y as Tensor)
    return { a: c.params.a, b: c.params.b, predict: c.predict(t as Tensor) }
  },
  isotonic: ([s, y, t]) => ({ predict: isotonicCalibration(s as Tensor, y as Tensor).predict(t as Tensor) }),
  temperature: ([s, y, t]) => {
    const c = temperatureScaling(s as Tensor, y as Tensor)
    return { beta: c.params.beta, predict: c.predict(t as Tensor) }
  },
  vector: ([z, y, t]) => ({ predict: vectorScaling(z as Tensor, y as Tensor).predict(t as Tensor) }),
  beta: ([p, y, t]) => {
    const c = betaCalibration(p as Tensor, y as Tensor)
    return { a: c.params.a, b: c.params.b, c: c.params.c, predict: c.predict(t as Tensor) }
  },
  histogram: ([p, y, t], params) => ({
    predict: histogramBinning(p as Tensor, y as Tensor, { nBins: num(params, 'nBins') }).predict(t as Tensor),
  }),
  calibratedCv: ([x, y, t], p) => {
    const f = calibratedClassifierCV({
      estimator: gaussianNB(),
      method: str(p, 'method') as CalibrationMethod,
      cv: stratifiedKFold(3),
      ensemble: p['ensemble'] === true,
    }).fit({ x: x as Tensor, y: y as Tensor })
    return { proba: f.predictProba(t as Tensor) }
  },
  calibratedLogistic: ([x, y, t]) => {
    const f = calibratedClassifierCV({ estimator: logisticRegression(), cv: stratifiedKFold(3) }).fit({
      x: x as Tensor,
      y: y as Tensor,
    })
    return { proba: f.predictProba(t as Tensor) }
  },
})
