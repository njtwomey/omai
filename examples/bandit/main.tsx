/**
 * Example "Which arm to pull": three ways of playing a five-armed bandit whose arms pay 1 with probabilities the player
 * does not know (epsilon-greedy, UCB1 and Thompson sampling), each run as a traced algorithm for 400 pulls. The tape
 * holds the arm pulled and the regret so far at every pull, and the figure is drawn from them: the regret curves, and
 * how often each strategy pulled each arm.
 */
import { banditAlgorithm, epsilonGreedyBandit, runBandit, thompsonBernoulli, ucb1 } from 'omai-engine/agents'
import { bernoulliBandit } from 'omai-engine/environments'
import { key } from 'omai-engine/random'
import { metricSeries, type Run, recorder, runAlgorithm, tracePolicy } from 'omai-engine/trace'
import { barChart } from 'omai-render/charts'
import { runsChart } from 'omai-render/trace'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Which arm to pull',
  summary:
    'Epsilon-greedy, UCB1 and Thompson sampling on a five-armed bandit: the regret curves and how often each arm was pulled.',
  readme: false,
  alt: 'Cumulative regret of three bandit strategies over 400 pulls, and the number of pulls of each of the five arms',
}

const probabilities = [0.2, 0.35, 0.5, 0.6, 0.7]
const pulls = 400
/** One run is drawn; the notes give the mean regret over ten seeds, since a single run can be lucky or not. */
const seed = 2

type Strategy = 'epsilon-greedy' | 'ucb1' | 'thompson'

export function run(): Outcome<Record<Strategy, Run>> {
  const env = bernoulliBandit.create({ probs: probabilities, horizon: pulls })
  const agents = {
    'epsilon-greedy': epsilonGreedyBandit(env, { epsilon: 0.1 }),
    ucb1: ucb1(env),
    thompson: thompsonBernoulli(env),
  }
  const tapes = {} as Record<Strategy, Run>
  const notes: string[] = []
  for (const [name, agent] of Object.entries(agents) as [Strategy, (typeof agents)[Strategy]][]) {
    const tape = recorder()
    runAlgorithm(banditAlgorithm(env, agent, { pulls, key: key(seed) }), {
      key: key(seed),
      maxSteps: pulls,
      tracer: tape,
      policy: tracePolicy({ metrics: ['bandit/arm', 'bandit/regret'] }),
    })
    tapes[name] = tape.run()
    const regret = metricSeries(tapes[name], 'bandit/regret').values
    notes.push(`${name}: regret ${regret.at(-1)?.toFixed(1)}, pulls of each arm ${countArms(tapes[name]).join(' ')}`)
  }
  const mean = Object.fromEntries(
    Object.entries(agents).map(([name, agent]) => {
      const total = Array.from(
        { length: 10 },
        (_, k) => runBandit(env, agent, { pulls, key: key(k) }).regret.at(-1) ?? 0,
      )
      return [name, total.reduce((a, b) => a + b, 0) / total.length]
    }),
  )
  notes.push(
    `mean regret over seeds 0 to 9: ${Object.entries(mean)
      .map(([n, r]) => `${n} ${r.toFixed(1)}`)
      .join(', ')}`,
  )
  return { result: tapes, tapes, notes }
}

/** How many times each arm was pulled, counted from the recorded arms. */
function countArms(tape: Run): number[] {
  const counts = probabilities.map(() => 0)
  for (const arm of metricSeries(tape, 'bandit/arm').values) counts[arm] = (counts[arm] ?? 0) + 1
  return counts
}

export function View({ result }: { result: Record<Strategy, Run> }) {
  const names = Object.keys(result) as Strategy[]
  return (
    <Row
      alt={meta.alt}
      width={470}
      height={320}
      panels={[
        {
          title: 'cumulative regret (one run of each)',
          still: runsChart(
            names.map((name) => ({ name, run: result[name] })),
            'bandit/regret',
            { y: { label: 'regret' } },
          ),
        },
        {
          title: 'pulls of each arm (the best arm is the last)',
          still: barChart(
            probabilities.map((p) => `p = ${p}`),
            names.map((name) => ({ name, values: countArms(result[name]) })),
            { legend: true, y: { label: 'pulls' } },
          ),
        },
      ]}
    />
  )
}
