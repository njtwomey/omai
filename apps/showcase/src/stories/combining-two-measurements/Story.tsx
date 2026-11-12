import { type ModelRun, modelRunAt } from 'omai-engine/exposition'
import { messageTex } from 'omai-render/views'
import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex, tex } from '../../kit/Maths.tsx'
import { MessageReplay } from '../../kit/MessageReplay.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { DIAL_SD, DIGITAL_SD, PRIOR } from './data.ts'
import { MessagesFigure, PriorFigure } from './Figures.tsx'
import { readCombination } from './reading.ts'

/** A number with a fixed number of decimals. */
const fixed = (v: number, digits = 2): string => v.toFixed(digits)
/** A number with three significant digits, without trailing zeros. */
const sig3 = (v: number): string => String(Number(v.toPrecision(3)))
/** A share as a percentage. */
const pct = (v: number): string => `${Math.round(v * 100)}%`

const SENDER: Readonly<Record<string, string>> = {
  'factor:temperature': 'The prior',
  'factor:digital': 'The digital reading',
  'factor:dial': 'The dial reading',
}

/** What happens at one step of the replay, in words: the messages sent, in order, and the belief that follows. */
function describeStep(step: number, record: ModelRun): ReactNode {
  const at = modelRunAt(record, step)
  const index = record.steps.findIndex((s) => s.step === step)
  const before = record.steps[index - 1]?.messages?.map((m) => m.text).join('|')
  const same = at !== undefined && at.messages.length > 0 && at.messages.map((m) => m.text).join('|') === before
  const belief = at?.nodes['temperature']
  if (at === undefined || at.messages.length === 0)
    return (
      <>
        <p className="m-0 font-medium">Before the first sweep</p>
        <p className="m-0">No message has been sent. The temperature has no belief yet.</p>
      </>
    )
  return (
    <>
      <p className="m-0 font-medium">Sweep {step}</p>
      <ul className="m-0 list-none p-0">
        {at.messages.map((m) => (
          <li key={m.from}>
            {SENDER[m.from] ?? m.from} sends{' '}
            {messageTex(m.text) === undefined ? m.text : <Tex>{messageTex(m.text) as string}</Tex>}.
          </li>
        ))}
      </ul>
      <p className="m-0">
        {belief?.mean === undefined || belief.sd === undefined
          ? 'The temperature has no belief yet.'
          : `The belief about the temperature is now mean ${sig3(belief.mean)}, standard deviation ${sig3(belief.sd)}.`}
        {same ? ' The messages are the same as in the sweep before, so the belief did not change. EP stops.' : ''}
      </p>
    </>
  )
}

/** 'Combining two measurements': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        A room has a temperature, and we want to know it. Before we measure, we believe it is about {PRIOR.mean} degrees
        Celsius, with a spread of {PRIOR.sd} degrees. Then we measure it twice. A digital thermometer has an error with
        a spread of {DIGITAL_SD} degrees. An old dial thermometer has an error with a spread of {DIAL_SD} degrees.
        Neither reading is exactly right.
      </p>

      <Figure caption="The belief before any reading: a bell curve centred on 20 degrees, mostly between 12 and 28.">
        <PriorFigure />
      </Figure>

      <p>
        We now have three pieces of information about one number: the belief before measuring, the digital reading and
        the dial reading. Each is a bell curve over the temperature. The belief after both readings is the product of
        the three curves, rescaled to have area 1.
      </p>

      <Aside>
        A bell curve (a normal distribution) has a <em>mean</em>, where its peak is, and a <em>variance</em>, which says
        how wide it is. Its <em>precision</em> is 1 divided by the variance, so a narrow curve has a high precision.
      </Aside>

      <h2>Doing the product with messages</h2>

      <p>
        Expectation propagation, or EP, computes beliefs by passing messages on a graph. Here the graph has one node,
        the temperature, and three factors: the prior, the digital reading and the dial reading. A message is a bell
        curve that a factor sends to the temperature. It says what that factor alone knows.
      </p>

      <p>
        EP works in sweeps. In a sweep, each factor sends its message to the temperature, and the temperature multiplies
        them to get its belief. A product of bell curves is a bell curve. Its precision is the sum of the three
        precisions, and its mean is the average of the three means, weighted by precision. EP stops when a sweep changes
        no belief.
      </p>

      <Equation
        label={
          <>
            The product of the three bells: precision <Tex>J_k</Tex> and mean <Tex>\mu_k</Tex>, for the prior (0), the
            digital reading (1) and the dial reading (2)
          </>
        }
        reading="The precision J of the product is J zero plus J one plus J two. Its mean mu is the sum of each precision times its mean, divided by J."
      >
        {String.raw`J = J_0 + J_1 + J_2, \qquad \mu = \frac{J_0\,\mu_0 + J_1\,\mu_1 + J_2\,\mu_2}{J}`}
      </Equation>

      <p>
        When you press play, a worker makes a room and two readings from a seed, and runs EP on them. Then the page
        shows what was sent.
      </p>

      <Play run={run} label="Measure the room twice" estimate="about a second" />

      <Results run={run}>
        {(result) => {
          const r = readCombination(result)
          return (
            <>
              <h2>The run</h2>

              <p>
                The room&apos;s real temperature was <span className="num">{fixed(result.truth)}</span> degrees. The
                model was not told this number. The digital thermometer read{' '}
                <span className="num">{fixed(result.digital)}</span>, which is{' '}
                <span className="num">{fixed(Math.abs(result.digital - result.truth))}</span> degrees away. The dial
                read <span className="num">{fixed(result.dial)}</span>, which is{' '}
                <span className="num">{fixed(Math.abs(result.dial - result.truth))}</span> degrees away. EP made{' '}
                <span className="num">{result.sweeps}</span> sweeps.
              </p>

              <MessageReplay
                record={result.record}
                label="The messages of the run, sweep by sweep"
                describe={describeStep}
                direction="right"
                rankSep={150}
                caption="The graph of the model: the circle is the temperature and the squares are the factors. Step through the sweeps with the button or the scrubber."
              />

              <p>
                In the replay, <Tex>{String.raw`\Gauss(\mu = ${PRIOR.mean}, \sigma^2 = ${PRIOR.sd ** 2})`}</Tex> is a
                bell curve with mean {PRIOR.mean} and variance {PRIOR.sd ** 2}. In the first sweep, the prior sends mean{' '}
                <span className="num">{sig3(r.prior.mean)}</span> and variance{' '}
                <span className="num">{sig3(r.prior.variance)}</span>, the belief we started with. The digital factor
                sends mean <span className="num">{sig3(r.digital.mean)}</span> and variance{' '}
                <span className="num">{sig3(r.digital.variance)}</span>, the reading with the variance of its error. The
                dial factor sends mean <span className="num">{sig3(r.dial.mean)}</span> and variance{' '}
                <span className="num">{sig3(r.dial.variance)}</span>.
              </p>

              <h2>The product of the messages</h2>

              <p>
                The temperature multiplies the three messages. Their precisions are{' '}
                <span className="num">{sig3(r.prior.precision)}</span>,{' '}
                <span className="num">{sig3(r.digital.precision)}</span> and{' '}
                <span className="num">{sig3(r.dial.precision)}</span>, and they add up to{' '}
                <span className="num">{sig3(r.total)}</span>. So the belief has a variance of{' '}
                <span className="num">{sig3(1 / r.total)}</span> and a standard deviation of{' '}
                <span className="num">{sig3(r.product.sd)}</span>. Its mean, weighted by precision, is{' '}
                <span className="num">{sig3(r.product.mean)}</span>.
              </p>

              <Equation label="The same sums with the numbers of this run, rounded to three digits">
                {tex`\begin{aligned}
                  J &\approx ${sig3(r.prior.precision)} + ${sig3(r.digital.precision)} + ${sig3(r.dial.precision)} \approx ${sig3(r.total)} \\
                  \mu &\approx \frac{${sig3(r.prior.precision)} \cdot ${sig3(r.prior.mean)} + ${sig3(r.digital.precision)} \cdot ${sig3(r.digital.mean)} + ${sig3(r.dial.precision)} \cdot ${sig3(r.dial.mean)}}{${sig3(r.total)}} \approx ${sig3(r.product.mean)}
                \end{aligned}`}
              </Equation>

              <p>
                EP reports a belief with mean <span className="num">{sig3(result.belief.mean)}</span> and standard
                deviation <span className="num">{sig3(result.belief.sd)}</span>.{' '}
                {r.agrees
                  ? 'This matches the product worked out by hand. The graph has no loops, so one sweep gives the exact answer.'
                  : 'This differs from the product worked out by hand, which the exact answer should not.'}
              </p>

              <Figure caption="The three messages of the first sweep and their product, the belief. The narrowest curve is the tallest.">
                <MessagesFigure result={result} />
              </Figure>

              <p>
                The digital reading supplies <span className="num">{pct(r.share.digital)}</span> of the total precision.
                The dial supplies <span className="num">{pct(r.share.dial)}</span> and the prior{' '}
                <span className="num">{pct(r.share.prior)}</span>. The belief&apos;s mean is therefore{' '}
                <span className="num">{fixed(r.gapToDigital)}</span> degrees from the digital reading and{' '}
                <span className="num">{fixed(r.gapToDial)}</span> degrees from the dial reading. Its standard deviation,{' '}
                <span className="num">{sig3(result.belief.sd)}</span>, is smaller than the {DIGITAL_SD} of the digital
                thermometer alone. Every message adds precision, so combining readings never leaves us less sure than
                the best one.
              </p>

              <p>
                {result.settled
                  ? `In sweep ${result.sweeps} the largest change of any belief was ${r.lastChange === undefined ? 'not recorded' : sig3(r.lastChange)}, and EP stopped.`
                  : 'EP did not settle within its sweep limit.'}{' '}
                Here each message depends only on its own factor, so nothing changes after the first sweep. In a graph
                with loops the messages depend on each other, and EP needs several sweeps.
              </p>

              <p>
                Press <em>Run again</em> for another room. The readings and the messages change, and the precisions
                still add. <Link to={storyHref('two-humps')}>Two kinds of wait at Old Faithful</Link> uses the same kind
                of messages on a model with many more factors.
              </p>
              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
