import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { key } from '../random/index.ts'
import type { Env } from '../rl/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { acrobot, cartPole, mountainCar, mountainCarContinuous, pendulum } from './classic-control.ts'

const envs: Readonly<Record<string, Env<readonly number[], readonly number[], unknown>>> = {
  cartPole: cartPole.create() as unknown as Env<readonly number[], readonly number[], unknown>,
  mountainCar: mountainCar.create() as unknown as Env<readonly number[], readonly number[], unknown>,
  mountainCarContinuous: mountainCarContinuous.create() as unknown as Env<
    readonly number[],
    readonly number[],
    unknown
  >,
  pendulum: pendulum.create() as unknown as Env<readonly number[], readonly number[], unknown>,
  acrobot: acrobot.create() as unknown as Env<readonly number[], readonly number[], unknown>,
}

describe('classic control against gymnasium, step for step', () => {
  for (const c of loadFixture('methods/rl/environments/classic_control').cases) {
    it(c.name, () => {
      const env = envs[c.fn]
      if (env === undefined) throw new Error(`no environment ${c.fn}`)
      let state = toFlat(decode(c.inputs[0]) as Tensor) as readonly number[]
      const states: number[][] = []
      const rewards: number[] = []
      const terminated: number[] = []
      for (const a of c.params['actions'] as unknown[]) {
        const r = env.step(state, a, key(0))
        state = r.state
        states.push([...state])
        rewards.push(r.reward)
        terminated.push(r.terminated ? 1 : 0)
        if (r.terminated) break
      }
      const part = c.params['part']
      const got = part === 'states' ? tensor(states) : part === 'rewards' ? tensor(rewards) : tensor(terminated)
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})
