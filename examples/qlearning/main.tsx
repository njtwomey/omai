/**
 * README example "An agent learning": tabular Q-learning on the cliff-walking environment, trained for 300 episodes as
 * a traced run. The learning curve is drawn from the tape.
 */
import { qLearning } from 'mdml-engine/agents'
import { cliffWalking } from 'mdml-engine/environments'
import { make, trainAgent } from 'mdml-engine/rl'
import { episodes, type Run, recorder, tracePolicy } from 'mdml-engine/trace'
import { cartesian, lines, points } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { rolling } from 'mdml-render/trace'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'An agent learning',
  summary: 'Q-learning on the cliff-walking gridworld: the return of each episode, drawn from the recorded run.',
  readme: true,
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

/** The lowest return the axis shows: the first episodes return about -2000, which would flatten everything else. */
const FLOOR = -400
const WINDOW = 20

export function View({ result }: { result: Run }) {
  const eps = episodes(result)
  const x = eps.map((e) => e.episode)
  const y = eps.map((e) => e.return)
  // The trailing mean from the first full window on, so that it starts where it is an average of WINDOW episodes.
  const mean = rolling(y, WINDOW).mean.slice(WINDOW - 1)
  const off = y.filter((v) => v < FLOOR)
  const plot = cartesian(
    [
      points([{ x, y, name: 'episode return', colour: '#8a94a0', size: 4, opacity: 0.55, outline: false }]),
      lines([{ x: x.slice(WINDOW - 1), y: mean, name: `${WINDOW}-episode mean`, colour: '#1f77b4', width: 2.5 }]),
    ],
    {
      figure: figure({ width: 640, height: 330, padding: { top: 30, right: 14, bottom: 38, left: 50 } }),
      x: { label: 'episode', limits: [0, x.length - 1] },
      y: { label: 'return', limits: [FLOOR, 0] },
      legend: true,
    },
  )
  const note = off.length === 0 ? '' : `: ${off.length} below ${FLOOR} not shown (${off.join(', ')})`
  return (
    <Figure width={640} height={356} title={meta.alt}>
      <Panel width={640} height={330} title={`return of each episode and its ${WINDOW}-episode mean${note}`} still={plot} />
    </Figure>
  )
}
