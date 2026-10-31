import { describe, expect, it } from 'vitest'
import { expectClose } from '../../../testing/close.ts'
import { minimise } from '../../../testing/optimise.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { odeSolver } from '../../dynamics/ode/index.ts'
import { key, normal, uniform } from '../../random/index.ts'
import { cos, mean, mul, sliceAxis, square, sub, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { Softplus } from '../module/activations.ts'
import { Lambda, mlp } from '../module/containers.ts'
import { HamiltonianNet, LagrangianNet } from './mechanics.ts'

const q = (x: Value): Value => sliceAxis(x, -1, 0, 1)
const p = (x: Value): Value => sliceAxis(x, -1, 1, 2)
const rk4 = odeSolver('RK4', { dt: 0.02 })

/** H = p^2 / 2 - cos q, the pendulum. */
const pendulum = new Lambda(
  (x) => sub(mul(square(p(x)), 0.5), cos(q(x))),
  (s) => [...s.slice(0, -1), 1],
)

describe('Hamiltonian network', () => {
  it('the field is the symplectic gradient: (dH/dp, -dH/dq)', () => {
    const net = new HamiltonianNet(1, { energy: pendulum })
    const state = tensor([
      [0.7, -0.4],
      [1.3, 0.9],
    ])
    // dH/dq = sin q, dH/dp = p.
    expectClose(
      net.forward(state),
      tensor([
        [-0.4, -Math.sin(0.7)],
        [0.9, -Math.sin(1.3)],
      ]),
      1e-12,
    )
    const learned = new HamiltonianNet(2, { hidden: [6], key: key(0) })
    const s = normal(key(1), [3, 4])
    const field = toFlat(learned.forward(s))
    const h = 1e-6
    const energy = (row: number[]): number => toFlat(learned.energy(learned.params, [row]))[0] as number
    const rows = toFlat(s)
    for (let r = 0; r < 3; r++) {
      const x = rows.slice(r * 4, r * 4 + 4)
      const dH = x.map(
        (_, i) =>
          (energy(x.map((v, j) => (j === i ? v + h : v))) - energy(x.map((v, j) => (j === i ? v - h : v)))) / (2 * h),
      )
      expect(field[r * 4]).toBeCloseTo(dH[2] as number, 6)
      expect(field[r * 4 + 1]).toBeCloseTo(dH[3] as number, 6)
      expect(field[r * 4 + 2]).toBeCloseTo(-(dH[0] as number), 6)
      expect(field[r * 4 + 3]).toBeCloseTo(-(dH[1] as number), 6)
    }
  })

  it('conserves the learned energy along the flow, where a plain neural ODE does not', () => {
    const net = new HamiltonianNet(1, { hidden: [16, 16], key: key(2) })
    const start = tensor([[0.8, 0.3]])
    const times = Array.from({ length: 51 }, (_, i) => i * 0.1)
    const states = net.flow(net.params, start, times, rk4)
    const energies = states.map((s) => toFlat(net.energy(net.params, s))[0] as number)
    const spread = Math.max(...energies) - Math.min(...energies)
    expect(spread).toBeLessThan(1e-7)
    // The same network used directly as a field is not conservative: its "energy" changes.
    const free = mlp([2, 16, 16, 2], { activation: () => new Softplus(), key: key(2) })
    const path = rk4((_t, y) => free.forward(y), start, times)
    const e2 = path.map((s) => toFlat(net.energy(net.params, s))[0] as number)
    expect(Math.max(...e2) - Math.min(...e2)).toBeGreaterThan(1e-3)
  })

  it('learns the pendulum energy from observed derivatives, up to a constant', () => {
    const states = mul(sub(uniform(key(3), [96, 2]), 0.5), 4)
    const target = tensor(
      toFlat(states).reduce<number[][]>((rows, v, i) => {
        if (i % 2 === 0) rows.push([toFlat(states)[i + 1] as number, -Math.sin(v)])
        return rows
      }, []),
    )
    const net = new HamiltonianNet(1, { hidden: [24, 24], key: key(4) })
    const fit = minimise(net.params, (params) => mean(square(sub(net.apply(params, states), target))), 250, 0.02)
    expect(fit.losses[0] as number).toBeGreaterThan(0.2)
    expect(fit.losses.at(-1) as number).toBeLessThan(0.02)
    // H_learned - H_true is nearly constant over fresh states.
    const test = mul(sub(uniform(key(5), [40, 2]), 0.5), 3)
    const truth = toFlat(pendulum.forward(test))
    const learned = toFlat(net.energy(fit.params, test))
    const meanOf = (xs: number[]): number => xs.reduce((t, v) => t + v, 0) / xs.length
    const spreadOf = (xs: number[]): number => Math.sqrt(meanOf(xs.map((v) => (v - meanOf(xs)) ** 2)))
    const sd = spreadOf(learned.map((v, i) => v - (truth[i] as number)))
    const spread = spreadOf(truth)
    expect(sd).toBeLessThan(0.15 * spread)
  })
})

describe('Lagrangian network', () => {
  it('recovers the equations of motion from analytic Lagrangians', () => {
    // L = (qdot^2 - q^2) / 2: qddot = -q.
    const sho = new LagrangianNet(1, {
      lagrangian: new Lambda(
        (x) => mul(sub(square(p(x)), square(q(x))), 0.5),
        (s) => [...s.slice(0, -1), 1],
      ),
    })
    expectClose(
      sho.accelerations({}, [
        [0.6, 1.1],
        [-2, 0.3],
      ]),
      tensor([[-0.6], [2]]),
      1e-10,
    )
    // The pendulum L = qdot^2 / 2 + cos q: qddot = -sin q.
    const pend = new LagrangianNet(1, {
      lagrangian: new Lambda(
        (x) => sub(mul(square(p(x)), 0.5), mul(cos(q(x)), -1)),
        (s) => [...s.slice(0, -1), 1],
      ),
    })
    expectClose(pend.accelerations({}, [[0.9, 0.2]]), tensor([[-Math.sin(0.9)]]), 1e-10)
    // A position-dependent mass m(q) = 1 + q^2 gives the coupling term -m'(q) qdot^2 / (2 m) - V'(q) / m.
    const heavy = new LagrangianNet(1, {
      lagrangian: new Lambda(
        (x) => sub(mul(mul(add1(square(q(x))), square(p(x))), 0.5), mul(square(q(x)), 0.5)),
        (s) => [...s.slice(0, -1), 1],
      ),
    })
    const [qq, vv] = [0.5, 0.8]
    const m = 1 + qq * qq
    expect(toFlat(heavy.accelerations({}, [[qq, vv]]))[0]).toBeCloseTo((-qq - qq * vv * vv) / m, 10)
  })

  it('the learned Lagrangian flow follows the harmonic oscillator when given its Lagrangian', () => {
    const sho = new LagrangianNet(1, {
      lagrangian: new Lambda(
        (x) => mul(sub(square(p(x)), square(q(x))), 0.5),
        (s) => [...s.slice(0, -1), 1],
      ),
    })
    const times = [0, 1, 2, 3]
    const path = sho.flow({}, [[1, 0]], times, odeSolver('RK4', { dt: 0.01 }))
    for (const [i, t] of times.entries()) expect(toFlat(path[i] as Value)[0]).toBeCloseTo(Math.cos(t), 6)
  })

  it('has gradients through the Euler-Lagrange solve', () => {
    const net = new LagrangianNet(1, { hidden: [4], key: key(6) })
    const state = tensor([[0.3, 0.4]])
    expect(checkPrimitive((s: Value) => net.accelerations(net.params, s), [state], { rtol: 1e-4 }).failures).toEqual([])
    const layer = (net.params['lagrangian'] as Record<string, Record<string, Value>>)['2'] as Record<string, Value>
    const f = (w: Value): Value =>
      net.accelerations(
        { lagrangian: { ...(net.params['lagrangian'] as object), '2': { ...layer, weight: w } } },
        state,
      )
    expect(checkPrimitive(f, [layer['weight'] as Value], { rtol: 1e-4 }).failures).toEqual([])
  })
})

const add1 = (x: Value): Value => sub(x, -1)
