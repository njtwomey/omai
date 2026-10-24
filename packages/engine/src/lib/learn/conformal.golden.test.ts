import { gaussianNB, learnGolden, leastSquares, num, str } from '../../testing/learn.ts'
import { add, type Tensor, toFlat } from '../tensor/index.ts'
import {
  type ClassificationScore,
  conformalizedQuantileRegressor,
  crossConformalRegressor,
  mondrianConformalRegressor,
  splitConformalClassifier,
  splitConformalRegressor,
} from './conformal.ts'
import { kFold, leaveOneOut } from './model-selection/splitters.ts'

const interval = (r: { readonly lower: Tensor; readonly upper: Tensor }) => ({ lower: r.lower, upper: r.upper })

learnGolden('conformal prediction against MAPIE', 'learn/conformal', {
  splitAbsolute: ([xt, yt, xc, yc, z], p) => {
    const f = leastSquares().fit({ x: xt as Tensor, y: yt as Tensor })
    return interval(
      splitConformalRegressor(f, { x: xc as Tensor, y: yc as Tensor }, { alpha: num(p, 'alpha') }).predictInterval(
        z as Tensor,
      ),
    )
  },
  splitGamma: ([xt, yt, xc, yc, z], p) => {
    const f = leastSquares().fit({ x: xt as Tensor, y: yt as Tensor })
    return interval(
      splitConformalRegressor(
        f,
        { x: xc as Tensor, y: yc as Tensor },
        { alpha: num(p, 'alpha'), score: 'gamma' },
      ).predictInterval(z as Tensor),
    )
  },
  cqr: ([xt, yt, xc, yc, z], p) => {
    const lower = leastSquares().fit({ x: xt as Tensor, y: add(yt as Tensor, -1) as Tensor })
    const upper = leastSquares().fit({ x: xt as Tensor, y: add(yt as Tensor, 1) as Tensor })
    const c = conformalizedQuantileRegressor(
      { lower, upper },
      { x: xc as Tensor, y: yc as Tensor },
      {
        alpha: num(p, 'alpha'),
        symmetric: p['symmetric'] === true,
      },
    )
    return interval(c.predictInterval(z as Tensor))
  },
  cross: ([x, y, z], p) => {
    const c = crossConformalRegressor(
      leastSquares(),
      { x: x as Tensor, y: y as Tensor },
      {
        alpha: num(p, 'alpha'),
        method: str(p, 'method') as 'plus' | 'minmax',
        cv: p['loo'] === true ? leaveOneOut() : kFold(5),
      },
    )
    return interval(c.predictInterval(z as Tensor))
  },
  mondrian: ([xt, yt, xc, yc, z, gc, gz], p) => {
    const f = leastSquares().fit({ x: xt as Tensor, y: yt as Tensor })
    const c = mondrianConformalRegressor(f, { x: xc as Tensor, y: yc as Tensor }, toFlat(gc as Tensor), {
      alpha: num(p, 'alpha'),
    })
    return interval(c.predictInterval(z as Tensor, toFlat(gz as Tensor)))
  },
  classifier: ([xt, yt, xc, yc, z, u], p) => {
    const f = gaussianNB().fit({ x: xt as Tensor, y: yt as Tensor })
    const score = str(p, 'score') as ClassificationScore
    const c = splitConformalClassifier(
      f,
      { x: xc as Tensor, y: yc as Tensor },
      {
        alpha: num(p, 'alpha'),
        score,
        ...(u === undefined ? {} : { randomized: true, uniforms: toFlat(u) }),
      },
    )
    return { sets: c.predictSet(z as Tensor) }
  },
})
