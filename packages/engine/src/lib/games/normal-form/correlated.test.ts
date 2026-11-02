import { describe, expect, it } from 'vitest'
import { bimatrixGame, correlatedEquilibrium, isCorrelatedEquilibrium } from '../index.ts'

describe('correlatedEquilibrium on nearly tied payoffs', () => {
  // Found by fast-check on payoffs that tie to within 1e-6 (seed 2075411073): the linear programs are degenerate (every
  // right-hand side is 0), and the simplex used to pivot on entries of 1e-6 and 9e-9 of rounding noise, ending on a
  // singular basis; the interior point then called the bounded program unbounded.
  const a = [
    [-0.4710669971067, -0.5, -0.4999929999993],
    [0.30822808082279995, -0.5, -0.49999199999919997],
    [0.4999690999969, -0.012329951233000007, -0.5],
  ]
  const b = [
    [-0.4999979999998, -0.4999939999994, -0.08327295832729999],
    [-0.4159559915956, -0.5, -0.3674939867494],
    [0.4999840999984, -0.5, 0.4999960999996],
  ]
  // The same game with one payoff of the last row moved: the simplex with Bland's rule used to cycle on its coarse program.
  const b2 = b.map((row, i) => (i === 2 ? [0.4999840999984, -0.4999909999991, 0.49999209999919997] : row))
  // And a 3 x 4 game whose coarse program cycled to the default iteration limit.
  const a3 = [
    [-0.49999499999950003, 0.30773008077299996, 0.49998709999869995, -0.4999939999994],
    [0.49999409999940003, 0.49998109999810003, -0.4999869999987, -0.4999979999998],
    [0.4999850999985, -0.4999909999991, -0.49999499999950003, -0.49998299999830004],
  ]
  const b3 = [
    [0.49999809999980005, 0.4999850999985, -0.19919396991939997, 0.49999009999900007],
    [-0.1509259650926, -0.21167597116759998, 0.24415707441569995, 0.3914870891487],
    [-0.0987459598746, -0.49999899999990005, 0.006970050697000032, 0.4511950951195],
  ]
  it.each([
    ['the fast-check counterexample', a, b],
    ['the same with a payoff moved', a, b2],
    ['a 3 x 4 game', a3, b3],
  ])('solves %s, plain and coarse, with an equilibrium at 1e-7', (_, ra, rb) => {
    const game = bimatrixGame(ra, rb)
    for (const coarse of [false, true]) {
      const ce = correlatedEquilibrium(game, { coarse })
      expect(isCorrelatedEquilibrium(game, ce.distribution, { coarse, tol: 1e-7 })).toBe(true)
    }
  })
})
