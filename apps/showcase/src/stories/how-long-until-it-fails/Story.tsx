import type { ReactNode } from 'react'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Equation } from '../../kit/Maths.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { lifetimes, TRUE_SCALE, TRUE_SHAPE, UNITS } from './data.ts'
import { CurvesFigure, UpdateFigure } from './Figures.tsx'
import { readData, readResult } from './reading.ts'

// The forty units are the same every time, so the page can count them before anything runs.
const UNIT_DATA = lifetimes()
const DATA = readData(UNIT_DATA.time, UNIT_DATA.event)

const fixed = (v: number, digits = 1): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(100 * v)}%`

/** 'How long until it fails': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        A test puts <span className="num">{UNITS}</span> units to work and records how long each one lasts. The test
        cannot run for ever. Each unit is followed for a limited time, and the follow-up times differ from unit to unit.
        The lifetimes here are made up, so that we know the true answer and can check against it at the end. Time is in
        any fixed unit.
      </p>

      <p>
        <span className="num">{DATA.failed}</span> of the {DATA.count} units failed while they were followed. The other{' '}
        <span className="num">{DATA.censored}</span> were still working when their follow-up ended. A unit like that is
        called <em>censored</em>. We know that it lasted at least as long as it was followed. We do not know when it
        would have failed.
      </p>

      <p>
        The censored units still carry information. A unit that was followed for a long time and was still working tells
        us that units can last that long. If we drop the censored units, we keep only the units that failed early enough
        to be seen, and the estimate of how long units last comes out too short. The model should use each unit as what
        it is: a failure time for one, a lower bound for the other.
      </p>

      <h2>The model</h2>

      <p>
        We take the lifetime of a unit to follow a Weibull distribution. It has two numbers. The <em>scale</em> s is the
        time by which 63 in 100 units have failed. The <em>shape</em> k says how the risk of failing changes with age.
        Above 1 the risk rises with age, below 1 it falls. The chance that a unit is still working at time t is the{' '}
        <em>survival function</em>.
      </p>

      <Equation
        label="The survival function of a Weibull lifetime"
        reading="S of t equals e to the minus, t over s, to the power k."
      >
        {String.raw`S(t) = \exp\!\Big(-\big(t/s\big)^{k}\Big)`}
      </Equation>

      <p>
        We do not know s and k. We work with their logs, so that a bell curve (a normal distribution) can describe each.
        Each gets a wide bell curve as its prior: log s has a mean of 1 and a variance of 4, and log k has a mean of 0
        and a variance of 1.
      </p>

      <p>
        Each unit adds one factor to the model. A unit that failed at time y says that its lifetime was y. Its factor is
        the survival function at y times the <em>hazard</em> at y, the rate of failing at y among the units that have
        lasted until then. A unit that was censored at y says only that its lifetime was longer than y. Its factor is
        the survival function at y.
      </p>

      <Equation
        label="The factor of one unit"
        reading="For a unit that failed at y: k over s, times y over s to the power k minus 1, times e to the minus y over s to the power k. For a unit censored at y: e to the minus y over s to the power k."
      >
        {String.raw`p(\text{unit} \mid s, k) = \begin{cases}
          \dfrac{k}{s}\Big(\dfrac{y}{s}\Big)^{k-1} \exp\!\Big(-\big(y/s\big)^{k}\Big) & \text{failed at } y \\[6pt]
          \exp\!\Big(-\big(y/s\big)^{k}\Big) & \text{censored at } y \end{cases}`}
      </Equation>

      <h2>Passing messages</h2>

      <p>
        The model is a factor graph with two unknowns, log s and log k, and one factor for each unit. Every factor is
        connected to both unknowns. Expectation propagation (EP) keeps a bell curve for each unknown, and a message from
        each factor to each unknown. A message is also a bell curve. It is stored as two numbers: the precision (one
        over the variance) and the precision times the mean. The belief about an unknown is its prior times all its
        messages, which in these two numbers is a sum.
      </p>

      <ol className="mb-5 list-decimal">
        <li>
          Take one unit. Remove its own message from the belief about each unknown. What is left is the <em>cavity</em>:
          what we believe about the unknowns without this unit.
        </li>
        <li>
          Multiply the cavity by the unit&rsquo;s factor. The product is the <em>tilted</em> curve. It is not a bell
          curve.
        </li>
        <li>
          For each unknown, find the bell curve that has the same mean and variance as the tilted curve. This is moment
          matching.
        </li>
        <li>Divide that bell curve by the cavity. The quotient is the unit&rsquo;s new message to that unknown.</li>
        <li>
          Do this for every unit in turn. One pass over all the units is a sweep. Repeat the sweeps until no message
          changes.
        </li>
      </ol>

      <p>
        A failed unit and a censored unit differ only in step 2. The factor of a failed unit is the survival function
        times the hazard. The factor of a censored unit is the survival function alone. The library works out the tilted
        moments by numerical integration over both unknowns.
      </p>

      <Play run={run} label="Pass the messages" estimate="up to half a minute, in the background" />

      <Results run={run}>
        {(result) => {
          const read = readResult(result)
          const { update, all, dropped, exact, curves, sharpening } = result
          const last = sharpening.units.length - 1
          return (
            <>
              <h2>One unit, as a failure and as a censoring</h2>

              <p>
                In the first sweep the messages visit the units in order. Before unit{' '}
                <span className="num">{update.unit}</span> is visited, the units before it have sent their messages.
                Unit {update.unit} has the time <span className="num">{fixed(update.time, 2)}</span>, and the data say
                it was {update.actual === 'failed' ? 'a failure' : 'censored'}. The figure works out its update twice
                from the same cavity, once as a failure at that time and once as a censoring at that time.
              </p>

              <Figure
                caption="The belief about the scale for one unit's update: before the unit (grey, dashed), with the unit's factor (blue) and the bell curve that replaces it (orange, dotted). The vertical line is the unit's time."
                wide
              >
                <UpdateFigure update={update} />
              </Figure>

              <p>
                Take the failure first. A failure at time {fixed(update.time, 2)} says that the scale is not far from
                that time. The mean of log s moves {read.moved.failed}, from{' '}
                <span className="num">{fixed(update.failed.before.mean, 2)}</span> to{' '}
                <span className="num">{fixed(update.failed.after.mean, 2)}</span>, and its spread falls from{' '}
                <span className="num">{fixed(update.failed.before.sd, 2)}</span> to{' '}
                <span className="num">{fixed(update.failed.after.sd, 2)}</span>. The new message adds a precision of{' '}
                <span className="num">{fixed(update.failed.precisionScale, 2)}</span> to the scale.
              </p>

              <p>
                The censoring says that the unit lasted past time {fixed(update.time, 2)}. That rules out scales much
                smaller than the time, because they make a lifetime this long unlikely. It says nothing against larger
                scales. The mean of log s moves {read.moved.censored}, from{' '}
                <span className="num">{fixed(update.censored.before.mean, 2)}</span> to{' '}
                <span className="num">{fixed(update.censored.after.mean, 2)}</span>, and its spread falls from{' '}
                <span className="num">{fixed(update.censored.before.sd, 2)}</span> to{' '}
                <span className="num">{fixed(update.censored.after.sd, 2)}</span>. The message adds a precision of{' '}
                <span className="num">{fixed(update.censored.precisionScale, 2)}</span>, less than the failure did. A
                censored unit is worth less than a failed one, and it still counts.
              </p>

              <p>
                {update.actual === 'failed'
                  ? 'This unit failed, so the first case is the real update.'
                  : 'This unit was censored, so the second case is the real update.'}{' '}
                The message that the library&rsquo;s engine sent for it differs from the one worked out here by at most{' '}
                <span className="num">
                  {(
                    (update.actual === 'failed' ? update.failed.engineGap : update.censored.engineGap) ?? Number.NaN
                  ).toFixed(5)}
                </span>
                .
              </p>

              <h2>After all the units</h2>

              <p>
                The belief about log s started with a spread of{' '}
                <span className="num">{fixed(sharpening.scale[0] as number, 1)}</span>, the spread of its prior. In the
                first sweep it fell to <span className="num">{fixed(sharpening.scale[1] as number, 2)}</span> after{' '}
                {sharpening.units[1]} units, <span className="num">{fixed(sharpening.scale[2] as number, 2)}</span>{' '}
                after {sharpening.units[2]}, <span className="num">{fixed(sharpening.scale[3] as number, 2)}</span>{' '}
                after {sharpening.units[3]} and{' '}
                <span className="num">{fixed(sharpening.scale[last] as number, 2)}</span> after all{' '}
                {sharpening.units[last]}. The messages settled after <span className="num">{all.sweeps}</span> sweeps,
                at a spread of <span className="num">{fixed(all.scale.sd, 2)}</span>. The first sweep ended with a
                spread {read.firstSweepShare < 1 ? 'smaller' : 'larger'} than the settled one, so its belief was{' '}
                {read.firstSweepShare < 1 ? 'too confident' : 'too cautious'}. The later sweeps correct it.
              </p>

              <p>
                The settled belief gives a scale of <span className="num">{fixed(read.scale.value, 1)}</span> (a 90%
                interval from <span className="num">{fixed(read.scale.low, 1)}</span> to{' '}
                <span className="num">{fixed(read.scale.high, 1)}</span>) and a shape of{' '}
                <span className="num">{fixed(read.shape.value, 2)}</span> (from{' '}
                <span className="num">{fixed(read.shape.low, 2)}</span> to{' '}
                <span className="num">{fixed(read.shape.high, 2)}</span>). The units were made with a scale of{' '}
                <span className="num">{fixed(TRUE_SCALE, 1)}</span> and a shape of{' '}
                <span className="num">{fixed(TRUE_SHAPE, 1)}</span>.
              </p>

              <h2>The survival curve</h2>

              <p>
                Each pair of s and k is one survival curve. We draw <span className="num">2000</span> pairs from the two
                bell curves, which gives 2000 curves. At each time, the 5th and the 95th percentile of the curves are
                the edges of a 90% band. The draws come from the run&rsquo;s seed, so <em>Run again</em> draws them
                again and the edges of the band move a little.
              </p>

              <Figure
                caption="The fitted survival curve with a 90% band, the true curve (dashed), and the curve fitted without the censored units (dotted). Under the axis, a dot is a unit that failed and a cross is a unit that was still working."
                wide
              >
                <CurvesFigure result={result} />
              </Figure>

              <p>
                The true curve lies inside the band at <span className="num">{percent(curves.covered)}</span> of the
                times drawn. The units were made with a median life of{' '}
                <span className="num">{fixed(read.median.truth, 1)}</span>, the time by which half of them have failed.
                The fit that uses every unit gives <span className="num">{fixed(read.median.fitted, 1)}</span>.
              </p>

              <p>
                The dotted curve is the same model fitted with the <span className="num">{DATA.censored}</span> censored
                units dropped. It gives a median life of <span className="num">{fixed(read.median.dropped, 1)}</span>.
                The mean of log s moves from <span className="num">{fixed(all.scale.mean, 2)}</span> to{' '}
                <span className="num">{fixed(dropped.scale.mean, 2)}</span>, which is{' '}
                <span className="num">{fixed(Math.abs(all.scale.mean - dropped.scale.mean) / all.scale.sd, 1)}</span>{' '}
                spreads of the full fit. The curve falls too fast, because the units that were dropped are the ones that
                were still working.
              </p>

              <h2>What EP got right and what it did not</h2>

              <p>
                The model is small enough to work out exactly: a grid over log s and log k, and the prior and every
                unit&rsquo;s factor added up at each cell. The exact belief about log s has a mean of{' '}
                <span className="num">{fixed(exact.scale.mean, 3)}</span> and a spread of{' '}
                <span className="num">{fixed(exact.scale.sd, 3)}</span>. EP has{' '}
                <span className="num">{fixed(all.scale.mean, 3)}</span> and{' '}
                <span className="num">{fixed(all.scale.sd, 3)}</span>. For log k the exact values are{' '}
                <span className="num">{fixed(exact.shape.mean, 3)}</span> and{' '}
                <span className="num">{fixed(exact.shape.sd, 3)}</span>, and EP has{' '}
                <span className="num">{fixed(all.shape.mean, 3)}</span> and{' '}
                <span className="num">{fixed(all.shape.sd, 3)}</span>.
              </p>

              <p>
                The means agree to within <span className="num">{fixed(read.meanOff, 2)}</span> of an exact spread. The
                spreads are <span className="num">{percent(read.spreadShare.scale)}</span> and{' '}
                <span className="num">{percent(read.spreadShare.shape)}</span> of the exact ones. EP keeps one bell
                curve for each unknown, and the two curves are independent. The exact beliefs have a correlation of{' '}
                <span className="num">{fixed(exact.correlation, 2)}</span>, so little is lost here. With a stronger
                correlation, the draws for the band would not follow the exact belief as closely.
              </p>

              <Aside>
                The band is for the survival curve, the chance that a unit lasts to a time. It is not a band for the
                failure time of one new unit.
              </Aside>

              <p>
                The model assumes a Weibull shape for the lifetimes, and that a unit&rsquo;s follow-up time says nothing
                about how long it would have lasted. Both hold for these made-up units. The Cox model makes no
                assumption about the shape. The library has it as a fitted estimator, but its partial likelihood is not
                yet a message-passing model.
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
