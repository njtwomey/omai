import { describe, expect, it } from 'vitest'
import { flatObjective, smoothSolverConfig } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  fitMlnProblem,
  groundMln,
  learnMlnDiscriminative,
  learnMlnWeights,
  learnMlnWeightsExact,
  type MarkovLogicNetwork,
  mlnConditionalLikelihoodProblem,
  mlnExact,
  mlnLikelihoodProblem,
  mlnLogWeight,
  mlnPerceptronAlgorithm,
  mlnPseudoLikelihoodAlgorithm,
  mlnPseudoLikelihoodProblem,
  mlnSampleWorlds,
  votedWeights,
  withWeights,
  worldVector,
} from './index.ts'

/** Every entry of `actual` is within `tolerance` of `expected`. */
function within(actual: readonly number[], expected: readonly number[], tolerance: number): void {
  expect(actual.length).toBe(expected.length)
  for (let j = 0; j < actual.length; j++)
    expect(Math.abs((actual[j] as number) - (expected[j] as number))).toBeLessThan(tolerance)
}

const PLANTED = [1.1, 1.5, -0.4, -0.2]
const smokers = (domain: string[], weights: number[] = PLANTED): MarkovLogicNetwork => ({
  domain,
  formulas: [
    { weight: weights[0] as number, formula: 'friends(X,Y) & smokes(X) -> smokes(Y)' },
    { weight: weights[1] as number, formula: 'smokes(X) -> cancer(X)' },
    { weight: weights[2] as number, formula: 'friends(X,Y)' },
    { weight: weights[3] as number, formula: 'smokes(X)' },
  ],
})

const planted = groundMln(smokers(['anna', 'bob', 'carl']))
const data = mlnSampleWorlds(planted, 600, key(11))
const large = mlnSampleWorlds(planted, 4000, key(12))
const blank = withWeights(planted, [0, 0, 0, 0])

describe('the objectives', () => {
  it('pseudo-likelihood equals its definition: the mean over worlds and atoms of -log P(x_l | rest)', () => {
    const w = [0.7, -0.3, 0.2, 0.5]
    const g = withWeights(planted, w)
    const few = data.slice(0, 5)
    let direct = 0
    for (const record of few) {
      const x = worldVector(g, record)
      for (let a = 0; a < g.atoms.length; a++) {
        const here = mlnLogWeight(g, x)
        x[a] = 1 - (x[a] as number)
        const flipped = mlnLogWeight(g, x)
        x[a] = 1 - (x[a] as number)
        direct += Math.log(1 + Math.exp(flipped - here))
      }
    }
    direct /= few.length * g.atoms.length
    const flat = flatObjective(mlnPseudoLikelihoodProblem(blank, few).objective, w)
    expect(flat.value(w)).toBeCloseTo(direct, 10)
  })

  it.each([
    ['pseudo-likelihood', mlnPseudoLikelihoodProblem(blank, data.slice(0, 40), { l2: 0.1 })],
    ['exact likelihood', mlnLikelihoodProblem(blank, data.slice(0, 40), { l2: 0.1 })],
    [
      'conditional likelihood',
      mlnConditionalLikelihoodProblem(blank, data.slice(0, 20), ['smokes', 'cancer'], { l2: 0.1 }),
    ],
  ])('the %s gradient and Hessian agree with finite differences', (_name, problem) => {
    const flat = flatObjective(problem.objective, problem.x0)
    const x = [0.4, -0.2, 0.3, 0.1]
    const g = flat.valueAndGrad(x).grad
    const h = flat.hessian(x)
    const eps = 1e-6
    for (let j = 0; j < 4; j++) {
      const up = [...x]
      const down = [...x]
      up[j] = (up[j] as number) + eps
      down[j] = (down[j] as number) - eps
      expect(g[j]).toBeCloseTo((flat.value(up) - flat.value(down)) / (2 * eps), 6)
      const gu = flat.valueAndGrad(up).grad
      const gd = flat.valueAndGrad(down).grad
      for (let k = 0; k < 4; k++)
        expect(h[j * 4 + k]).toBeCloseTo(((gu[k] as number) - (gd[k] as number)) / (2 * eps), 5)
    }
  })

  it('exact maximum likelihood matches the moments: the expected counts equal the data counts', () => {
    const fit = learnMlnWeightsExact(blank, data)
    expect(fit.converged).toBe(true)
    const e = mlnExact(withWeights(planted, fit.weights))
    const mean = [0, 0, 0, 0]
    for (const w of data) {
      const x = worldVector(planted, w)
      planted.formulas.forEach((f) => {
        if (f.holds(x)) mean[f.source] = (mean[f.source] as number) + 1 / data.length
      })
    }
    within(e.expectedCounts, mean, 1e-5)
  })
})

describe('generative weight learning', () => {
  it('a single formula: the pseudo-likelihood estimate is the log odds of the true fraction', () => {
    const g = groundMln({ domain: ['a', 'b', 'c', 'd'], formulas: [{ weight: 0, formula: 'p(X)' }] })
    const fit = learnMlnWeights(g, [{ 'p(a)': true, 'p(b)': true, 'p(c)': true }])
    expect(fit.weights[0]).toBeCloseTo(Math.log(3), 6)
    expect(fit.converged).toBe(true)
  })

  it('recovers planted weights from sampled worlds, close to the exact maximum-likelihood estimate', () => {
    const pll = learnMlnWeights(blank, large)
    const ml = learnMlnWeightsExact(blank, large)
    expect(pll.converged).toBe(true)
    within(pll.weights, PLANTED, 0.12)
    within(ml.weights, PLANTED, 0.1)
    within(pll.weights, ml.weights, 0.1)
  })

  it('every smooth solver reaches the same optimum', () => {
    const reference = learnMlnWeights(blank, data).weights
    for (const kind of ['newton', 'bfgs', 'conjugateGradient', 'gradientDescent']) {
      const fit = learnMlnWeights(blank, data, { solver: smoothSolverConfig({ kind, gradTol: 1e-8 }), maxSteps: 2000 })
      within(fit.weights, reference, 2e-3)
    }
  })

  it('the ridge penalty shrinks the weights, and the tape holds the weights and the solver’s channels', () => {
    const plain = learnMlnWeights(blank, data)
    const ridge = learnMlnWeights(blank, data, { l2: 0.5 })
    expect(Math.hypot(...ridge.weights)).toBeLessThan(Math.hypot(...plain.weights))
    const rec = recorder()
    const run = runAlgorithm(mlnPseudoLikelihoodAlgorithm(blank, data.slice(0, 50)), {
      seed: 0,
      maxSteps: 5,
      tracer: rec,
    })
    expect(run.steps).toBeGreaterThan(0)
    const text = JSON.stringify(rec.run())
    expect(text).toContain('mln/weights')
    expect(text).toContain('solver/objective')
  })

  it('rejects hard formulas and unknown starting sizes', () => {
    const hard = groundMln({ domain: ['a'], formulas: [{ weight: Number.POSITIVE_INFINITY, formula: 'p(a)' }] })
    expect(() => learnMlnWeights(hard, [{ 'p(a)': true }])).toThrow()
    expect(() => mlnPseudoLikelihoodProblem(blank, data, { initial: [0] })).toThrow()
  })
})

describe('discriminative learning', () => {
  // The prior on friendships involves only evidence atoms, so its weight is not identified by the conditional
  // likelihood; the discriminative network leaves it out.
  const planted3 = groundMln({
    domain: ['anna', 'bob', 'carl'],
    formulas: [
      { weight: 1.1, formula: 'friends(X,Y) & smokes(X) -> smokes(Y)' },
      { weight: 1.5, formula: 'smokes(X) -> cancer(X)' },
      { weight: -0.2, formula: 'smokes(X)' },
    ],
  })
  const blank3 = withWeights(planted3, [0, 0, 0])
  const query = ['smokes', 'cancer']
  // Friendships are the evidence: sample them at random, then the query atoms given them from the network.
  const train = mlnSampleWorlds(planted3, 150, key(21))

  it('exact expectations climb the conditional likelihood to its optimum', () => {
    const problem = mlnConditionalLikelihoodProblem(blank3, train, query)
    const flat = flatObjective(problem.objective, problem.x0)
    const best = fitMlnProblem(problem, { solver: smoothSolverConfig({ kind: 'newton', gradTol: 1e-10 }) })
    expect(best.converged).toBe(true)
    const run = learnMlnDiscriminative(blank3, train, { query, expectation: 'exact', learningRate: 1, passes: 400 })
    expect(flat.value(run.weights)).toBeLessThan(flat.value(Array.from(problem.x0 as ArrayLike<number>)))
    expect(flat.value(run.weights) - best.objective).toBeLessThan(2e-3)
    within(run.weights, best.weights, 0.1)
  })

  it('the MAP perceptron lowers the training error of the MAP world', () => {
    const alg = mlnPerceptronAlgorithm(blank3, train, { query, learningRate: 0.2 })
    const rec = recorder()
    const run = runAlgorithm(alg, { seed: 0, maxSteps: 25, tracer: rec })
    const first = runAlgorithm(alg, { seed: 0, maxSteps: 1 }).state.error
    expect(run.state.error).toBeLessThan(first)
    // The voted weights are the average of the iterates, and smoking implies cancer gets a positive weight.
    const voted = votedWeights(run.state)
    expect(voted[1]).toBeGreaterThan(0)
    const text = JSON.stringify(rec.run())
    for (const p of ['mln/weights', 'mln/voted', 'mln/error', 'mln/update']) expect(text).toContain(p)
    const back = alg.deserialise?.(alg.serialise?.(run.state) as never)
    expect(back?.weights).toEqual(run.state.weights)
  })
})
