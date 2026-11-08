/**
 * Example "An agent learning": tabular Q-learning on the cliff-walking environment, trained for 300 episodes as
 * a traced run. The learning curve is drawn from the tape.
 */
import { qLearning } from 'omai-engine/agents'
import { cliffWalking } from 'omai-engine/environments'
import { make, trainAgent } from 'omai-engine/rl'
import { type Run, recorder, tracePolicy } from 'omai-engine/trace'
import { episodeChart } from 'omai-render/trace'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'An agent learning',
  summary: 'Q-learning on the cliff-walking gridworld: the return of each episode, drawn from the recorded run.',
  readme: false,
  alt: 'Episode returns while Q-learning learns to walk along the cliff',
}

export function run(): Outcome<Run> {
  const tape = recorder()
  const agent = qLearning(cliffWalking.create(), { alpha: 0.5, gamma: 1, epsilon: 0.1 })
  const trained = trainAgent(make(cliffWalking), agent, {
    episodes: 300,
    seed: 0,
    tracer: tape,
    policy: tracePolicy({ metrics: false }),
  })
  const last = trained.returns.slice(-50)
  const notes = [
    `first returns ${trained.returns.slice(0, 3)}; mean of the last 50 ${(last.reduce((a, b) => a + b, 0) / last.length).toFixed(1)}`,
  ]
  return { result: tape.run(), tapes: { run: tape.run() }, notes }
}

export function View({ result }: { result: Run }) {
  return (
    <Figure width={640} height={340} title={meta.alt}>
      <Panel width={640} height={340} still={episodeChart(result, { y: { limits: [-200, 0] } })} />
    </Figure>
  )
}
