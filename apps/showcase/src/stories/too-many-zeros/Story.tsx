import { type ReactNode, useMemo } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex } from '../../kit/Maths.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { CHECKPOINTS, FIRST_SEED, ROWS, series, TRUTH, X_RANGE } from './data.ts'
import { BeliefsFigure, ChancesFigure, CountsFigure } from './Figures.tsx'
import { BANDS, logistic, oddsText, readBeliefs, readZeros } from './reading.ts'

/** A number with a fixed number of decimals. */
const fixed = (v: number, digits = 2): string => v.toFixed(digits)
/** A share as a percentage. */
const pct = (v: number): string => `${Math.round(v * 100)}%`

/** 'Counts with too many zeros': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(
    () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
    FIRST_SEED,
  )
  // The counts of the first run: cheap, so the page shows them before anything is computed.
  const first = useMemo(() => series(FIRST_SEED), [])
  const zeroCount = first.y.filter((v) => v === 0).length
  return (
    <StoryPage>
      <p>
        Here are {ROWS} counts. Each was taken at a value of <Tex>x</Tex> between {X_RANGE[0]} and {X_RANGE[1]}. The
        counts above zero get larger as <Tex>x</Tex> grows. {zeroCount} of the {ROWS} counts are zero.
      </p>

      <Figure caption="The counts against x. A count above zero is a dot at its height. The zeros are the squares in rows under the axis.">
        <CountsFigure x={first.x} y={first.y} />
      </Figure>

      <p>
        A count can be zero for two reasons. The first is that the thing could happen and did not. If a count is Poisson
        with rate <Tex>\lambda</Tex>, it is zero with probability <Tex>{String.raw`e^{-\lambda}`}</Tex>. A rate of 0.5
        gives a zero about 6 times in 10, and a rate of 5 gives one about once in 150. These are ordinary zeros. The
        second reason is that the thing could not happen at all. These are structural zeros. A zero looks the same
        either way, so for each zero the model has to decide which it was.
      </p>

      <p>
        The model gives every count a switch with two states, <em>count</em> and <em>structural</em>. A structural
        switch makes the count zero. A count switch makes it Poisson, with a rate that depends on <Tex>x</Tex> through
        two coefficients, <Tex>b_0</Tex> and <Tex>b_1</Tex>. The model is not told the switches, the coefficients, or{' '}
        <Tex>\pi</Tex>, the probability that a switch is structural. It starts with <Tex>\pi</Tex> uniform between 0 and
        1, and with <Tex>b_0</Tex> and <Tex>b_1</Tex> normal with mean 0 and standard deviation 2.
      </p>

      <Equation
        label="One count, by the state of its switch"
        reading="The rate lambda i is e to the power b zero plus b one times x i. The count y i is zero when the switch is structural, and Poisson with rate lambda i when the switch is count."
      >
        {String.raw`\begin{aligned} \lambda_i &= e^{b_0 + b_1 x_i} \\ y_i &= 0 \quad \text{if structural} \\ y_i &\sim \Poisson(\lambda_i) \quad \text{if count} \end{aligned}`}
      </Equation>

      <p>
        Expectation propagation, or EP, fits this model by passing messages on a graph. The graph has a variable for{' '}
        <Tex>\pi</Tex>, one for each switch, and one each for <Tex>b_0</Tex> and <Tex>b_1</Tex>. Each count is a factor
        that connects its switch, <Tex>b_0</Tex> and <Tex>b_1</Tex> to the count. When you press play, a worker reads
        the counts above and runs EP on the first {CHECKPOINTS.slice(0, -1).join(', ')} and then all {ROWS} of them. It
        also runs EP on all of them with a model that has no switch, and it works out the exact answer for comparison.
      </p>

      <Play run={run} label="Fit the model" estimate="about ten seconds" />

      <Results run={run}>
        {(result) => {
          const z = readZeros(result)
          const b = readBeliefs(result)
          const lowest = z.zeros[0]
          const highest = z.zeros.at(-1)
          if (lowest === undefined || highest === undefined) throw new Error('the counts have no zeros')
          const last = result.beliefs.at(-1)
          const [lowBand, midBand, highBand] = z.bands
          const [, , highReal] = z.bandsStructural
          const { withSwitch, withoutSwitch, exact } = result
          return (
            <>
              <h2>Which zeros are structural</h2>

              <p>
                The model ends with a chance for each zero that it is structural. The{' '}
                <span className="num">{z.zeros.length}</span> chances add up to{' '}
                <span className="num">{fixed(z.expected, 1)}</span>, so the model expects about that many of the zeros
                to be structural. The counts were made with <span className="num">{z.structural}</span> structural
                zeros, which the model was not told.
              </p>

              <Figure
                caption="The same counts after the run. Each zero is coloured by the chance the model gives it of being structural, and a diamond marks a zero that really was structural. The curves are the Poisson rate against x: dotted for the rate the counts were made from, solid for the rate EP fits with the switch, dashed for the rate it fits without."
                wide
              >
                <ChancesFigure result={result} />
              </Figure>

              <p>
                At large <Tex>x</Tex> the model is nearly sure. <span className="num">{highBand}</span> zeros have a
                chance over {pct(BANDS[1])}, and <span className="num">{highReal}</span> of them were structural. At
                small <Tex>x</Tex> it is not sure. The rate there is low, so an ordinary zero is likely and a zero says
                little. <span className="num">{lowBand}</span> zeros have a chance under {pct(BANDS[0])} and{' '}
                <span className="num">{midBand}</span> are between {pct(BANDS[0])} and {pct(BANDS[1])}. The lowest
                chance is <span className="num">{pct(z.lowest)}</span>. The model reports these zeros as uncertain and
                does not guess.
              </p>

              <h2>What the messages do</h2>

              <p>
                EP keeps a belief for every variable and a message from every factor to each variable it touches. The
                variable multiplies its messages together to get its belief. EP stores a message as natural parameters,
                so the natural parameters of a belief are the sum of those of its messages. A sweep recomputes the
                messages of every factor once. This run took <span className="num">{result.sweeps}</span> sweeps
                {result.settled ? ', and then no belief changed.' : ', and the beliefs were still changing.'}
              </p>

              <p>
                A count above zero cannot come from the structural state. Its factor sends the switch a message that
                gives structural no weight, so the switch is certain to be count. The count then acts on <Tex>b_0</Tex>{' '}
                and <Tex>b_1</Tex> as an ordinary Poisson count. There are <span className="num">{z.positives}</span> of
                these.
              </p>

              <p>
                A zero has probability 1 in the structural state and <Tex>{String.raw`e^{-\lambda_i}`}</Tex> in the
                count state. To send its message to the switch, the factor takes the belief about <Tex>b_0</Tex> and{' '}
                <Tex>b_1</Tex> with its own earlier message removed, which is called the cavity. It averages{' '}
                <Tex>{String.raw`e^{-\lambda_i}`}</Tex> over the cavity. The message is the pair of numbers 1 for
                structural and that average for count.
              </p>

              <Equation
                label="The message from the factor of a zero to its switch, as odds"
                reading="The odds of structural against count in the message equal one divided by the average of e to the minus lambda i under the cavity."
              >
                {String.raw`\frac{m_i(\text{structural})}{m_i(\text{count})} = \frac{1}{\expect_{\text{cavity}}\!\left[\,e^{-\lambda_i}\,\right]}`}
              </Equation>

              <p>
                In the last sweep the zero at <Tex>{`x = ${fixed(lowest.x)}`}</Tex>, where the rate is low, got a
                message with log-odds <span className="num">{fixed(lowest.gate)}</span> for structural, which is odds of{' '}
                <span className="num">{oddsText(lowest.gate)}</span>. The zero at <Tex>{`x = ${fixed(highest.x)}`}</Tex>
                , where the rate is high, got log-odds <span className="num">{fixed(highest.gate)}</span>, odds of{' '}
                <span className="num">{oddsText(highest.gate)}</span>.
              </p>

              <Aside>
                The log-odds of a chance <Tex>p</Tex> is <Tex>{String.raw`\log\frac{p}{1-p}`}</Tex>. A switch multiplies
                the odds of its messages, so their log-odds add.
              </Aside>

              <p>
                The switch also receives a message from the factor that ties it to <Tex>\pi</Tex>. For the zero at{' '}
                <Tex>{`x = ${fixed(lowest.x)}`}</Tex> its log-odds are{' '}
                <span className="num">{fixed(lowest.prior)}</span>. The belief of the switch is the product of the two
                messages, so its log-odds are <span className="num">{fixed(lowest.prior)}</span> +{' '}
                <span className="num">{fixed(lowest.gate)}</span> ={' '}
                <span className="num">{fixed(lowest.prior + lowest.gate)}</span>, a chance of{' '}
                <span className="num">{pct(logistic(lowest.prior + lowest.gate))}</span>. For the zero at{' '}
                <Tex>{`x = ${fixed(highest.x)}`}</Tex> it is <span className="num">{fixed(highest.prior)}</span> +{' '}
                <span className="num">{fixed(highest.gate)}</span> ={' '}
                <span className="num">{fixed(highest.prior + highest.gate)}</span>, a chance of{' '}
                <span className="num">{pct(logistic(highest.prior + highest.gate))}</span>. These are the colours in the
                figure.
              </p>

              <p>
                The messages also go the other way. For a zero, the factor's belief about the rate is a mixture. In the
                count state the zero says the rate is low. In the structural state it says nothing about the rate. EP
                replaces the mixture with one bell curve that has the same mean and variance, which is moment matching,
                divides it by the cavity and sends the result to <Tex>b_0</Tex> and <Tex>b_1</Tex>. A zero that is
                almost certainly structural has nearly all of its weight in the state that says nothing, so its message
                to the rate is nearly flat.
              </p>

              <p>
                The figure shows what that does. With no switch, the same EP must treat every zero as an ordinary
                Poisson zero. It fits an intercept of{' '}
                <Tex>{`b_0 = ${fixed(withoutSwitch.b0.mean)} \\pm ${fixed(withoutSwitch.b0.sd)}`}</Tex>. The counts were
                made with <Tex>{`b_0 = ${TRUTH.intercept}`}</Tex>, which is{' '}
                <span className="num">{fixed(b.withoutSwitch, 1)}</span> spreads from the fit.{' '}
                {b.withoutSwitch > b.withSwitch
                  ? 'The rate comes out too low, because a low rate is the only way the model can make so many zeros likely. '
                  : 'In this run the fit without the switch is no further from the truth. '}
                With the switch, <Tex>{`b_0 = ${fixed(withSwitch.b0.mean)} \\pm ${fixed(withSwitch.b0.sd)}`}</Tex>,
                which is <span className="num">{fixed(b.withSwitch, 1)}</span> spreads from the truth. The structural
                zeros no longer pull the rate down.
              </p>

              <h2>The belief about the switch probability</h2>

              <p>
                Each switch also sends a message to <Tex>\pi</Tex>. EP holds the belief about <Tex>\pi</Tex> as a Beta
                curve, a Dirichlet distribution over the two states, and its natural parameters again add up the
                messages. The figure shows the belief after the first {CHECKPOINTS.slice(0, -1).join(', ')} counts and
                after all {ROWS}. They are separate runs of EP.
              </p>

              <Figure
                caption={`The belief about the chance that a switch is structural, after ${CHECKPOINTS.slice(0, -1).join(', ')} and ${ROWS} counts. The dashed line is the chance the counts were made with, ${pct(TRUTH.inflation)}.`}
              >
                <BeliefsFigure result={result} />
              </Figure>

              <p>
                After {result.beliefs[0]?.rows} counts the belief has a spread of{' '}
                <span className="num">{fixed(b.firstSd)}</span> and covers much of the range. After {ROWS} counts the
                spread is <span className="num">{fixed(b.lastSd)}</span>, with a mean of{' '}
                <span className="num">{fixed(last?.mean ?? Number.NaN)}</span>. The range that holds nine tenths of the
                belief contains the true chance, {pct(TRUTH.inflation)}, at{' '}
                {b.inside === b.sizes ? (
                  `all ${b.sizes} sizes.`
                ) : (
                  <>
                    <span className="num">{b.inside}</span> of the {b.sizes} sizes.
                  </>
                )}
              </p>

              <h2>What EP got right and what it did not</h2>

              <p>
                The library also works out the exact posterior of this model. It sums each switch out, which makes the
                probability of a zero <Tex>{String.raw`\pi + (1-\pi)e^{-\lambda}`}</Tex>, and integrates <Tex>\pi</Tex>,{' '}
                <Tex>b_0</Tex> and <Tex>b_1</Tex> on a grid. EP's means for these three are within{' '}
                <span className="num">{fixed(b.meanGap, 3)}</span> exact spreads of the exact means. The chance EP gives
                each zero is within <span className="num">{fixed(exact.chanceGap, 3)}</span> of the exact chance.
              </p>

              <p>
                {b.sdRatio[1] < 1
                  ? `EP's spreads are smaller than the exact ones, by ${pct(1 - b.sdRatio[1])} to ${pct(1 - b.sdRatio[0])}. Its 90% ranges are that much too narrow.`
                  : `EP's spreads are between ${fixed(b.sdRatio[0])} and ${fixed(b.sdRatio[1])} times the exact ones.`}{' '}
                The spread of <Tex>\pi</Tex> is <span className="num">{fixed(last?.sd ?? Number.NaN, 3)}</span> from EP
                and <span className="num">{fixed(exact.pi.sd, 3)}</span> exactly.
              </p>

              <p>
                Two limits are in the counts and not in EP. At small <Tex>x</Tex> a zero is nearly as likely to be
                ordinary as structural, so no method can tell most of them apart. Calling a zero structural when its
                chance is over {pct(0.5)}, the model calls <span className="num">{z.called}</span> zeros structural.{' '}
                <span className="num">{z.calledRight}</span> of them were, and it missed{' '}
                <span className="num">{z.missed}</span> structural zeros. The estimates also differ from the truth: the
                exact posterior is up to <span className="num">{fixed(b.exactFromTruth, 1)}</span> of its own spreads
                from the true values, so a gap between EP and the truth is mostly in the {ROWS} counts, not in the
                approximation.
              </p>

              <p>
                The switch here is the same kind of structure as in{' '}
                <Link to={storyHref('signal-in-the-clutter')} className="underline underline-offset-2">
                  A signal in the clutter
                </Link>
                , where each reading is either a measurement or garbage, and the messages give each one a chance of
                being garbage while they find the value.
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
