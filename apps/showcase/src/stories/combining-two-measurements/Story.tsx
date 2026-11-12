import { type ModelRun, modelRunAt } from 'omai-engine/exposition'
import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { MessageReplay } from '../../kit/MessageReplay.tsx'
import { Play } from '../../kit/Play.tsx'
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
            {SENDER[m.from] ?? m.from} sends {m.text}.
          </li>
        ))}
      </ul>
      <p className="m-0">
        {belief?.mean === undefined || belief.sd === undefined
          ? 'The temperature has no belief yet.'
          : `The belief about the temperature is now mean ${sig3(belief.mean)}, standard deviation ${sig3(belief.sd)}.`}
        {same ? ' These are the same messages as in the sweep before, so the belief did not change and EP stops.' : ''}
      </p>
    </>
  )
}

/** 'Combining two measurements': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const r = result === undefined ? undefined : readCombination(result)
  return (
    <StoryPage>
      <p>
        A room has a temperature, and we want to know it. Before we measure anything, we believe it is about{' '}
        {PRIOR.mean} degrees Celsius, with a spread of {PRIOR.sd} degrees. Then we measure it twice. The first reading
        comes from a digital thermometer whose error has a spread of {DIGITAL_SD} degrees. The second comes from an old
        dial thermometer whose error has a spread of {DIAL_SD} degrees. Each reading is near the real temperature, and
        neither is exactly on it.
      </p>

      <Figure caption="The belief before any reading: a bell curve centred on 20 degrees. Most of it lies between 12 and 28 degrees.">
        <PriorFigure />
      </Figure>

      <p>
        We now have three pieces of information about one number: the belief before measuring, the digital reading and
        the dial reading. Each of them can be written as a bell curve over the temperature. The belief after both
        readings is the product of the three curves, rescaled so that its area is 1.
      </p>

      <Aside>
        A bell curve (a normal distribution) is described by its <em>mean</em>, where its peak is, and its{' '}
        <em>variance</em>, which measures how wide it is. Its <em>precision</em> is 1 divided by the variance, so a
        narrow curve has a high precision.
      </Aside>

      <h2>Doing the product with messages</h2>

      <p>
        Expectation propagation, or EP, computes beliefs by passing messages on a graph. For this model the graph has
        one node, the temperature, and three factors attached to it: the prior, the digital reading and the dial
        reading. A message is a bell curve that a factor sends to the temperature. It states what that factor alone says
        about the temperature.
      </p>

      <p>
        EP works in sweeps. In one sweep, each factor sends its message to the temperature. The temperature then
        multiplies the messages it received to get its belief. When the messages are bell curves, the product is also a
        bell curve. Its precision is the sum of the three precisions, and its mean is the average of the three means,
        each weighted by its precision.
      </p>

      <p>
        EP stops when a sweep changes no belief. When you press play, a worker makes the room and the two readings from
        a seed, writes the model, and runs EP on it. After it finishes, the page shows what was sent.
      </p>

      <Play run={run} label="Measure the room twice" estimate="about a second" />

      {r === undefined || result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the readings are made and EP is run in a background
          worker. The messages appear below this button.
        </p>
      ) : (
        <Reveal>
          <h2>The run</h2>

          <p>
            The room&apos;s real temperature was <span className="num">{fixed(result.truth)}</span> degrees. The model
            was not told this number. The digital thermometer read <span className="num">{fixed(result.digital)}</span>,
            which is <span className="num">{fixed(Math.abs(result.digital - result.truth))}</span> degrees away. The
            dial read <span className="num">{fixed(result.dial)}</span>, which is{' '}
            <span className="num">{fixed(Math.abs(result.dial - result.truth))}</span> degrees away. EP made{' '}
            <span className="num">{result.sweeps}</span> sweeps.
          </p>

          <MessageReplay
            record={result.record}
            label="The messages of the run, sweep by sweep"
            describe={describeStep}
            direction="right"
            rankSep={150}
            caption="The graph of the model. The circle in the middle is the temperature. The small squares are the factors: one for the prior, and one for each reading, joined to the circle of that reading. A message that is sent appears on the line it travels along. Press the button to step through the sweeps, or move the scrubber to any of them. The text under the graph says what was sent."
          />

          <p>
            In the replay, <code>{`N(μ = ${PRIOR.mean}, σ² = ${PRIOR.sd ** 2})`}</code> means a bell curve with mean{' '}
            {PRIOR.mean} and variance {PRIOR.sd ** 2}. The replay starts before any message. In the first sweep the
            three factors send these messages. The prior sends a bell curve with mean{' '}
            <span className="num">{sig3(r.prior.mean)}</span> and variance{' '}
            <span className="num">{sig3(r.prior.variance)}</span>, which is the belief we started with. The digital
            factor sends mean <span className="num">{sig3(r.digital.mean)}</span> and variance{' '}
            <span className="num">{sig3(r.digital.variance)}</span>: the reading, with the variance of that
            instrument&apos;s error. The dial factor sends mean <span className="num">{sig3(r.dial.mean)}</span> and
            variance <span className="num">{sig3(r.dial.variance)}</span>.
          </p>

          <h2>The product of the messages</h2>

          <p>
            The temperature multiplies the three messages. The precisions are{' '}
            <span className="num">{sig3(r.prior.precision)}</span>,{' '}
            <span className="num">{sig3(r.digital.precision)}</span> and{' '}
            <span className="num">{sig3(r.dial.precision)}</span>. They add up to{' '}
            <span className="num">{sig3(r.total)}</span>, so the belief has a variance of{' '}
            <span className="num">{sig3(1 / r.total)}</span> and a standard deviation of{' '}
            <span className="num">{sig3(r.product.sd)}</span>. Its mean is the average of the three means weighted by
            their precisions, which is <span className="num">{sig3(r.product.mean)}</span>.
          </p>

          <p>
            EP reports a belief with mean <span className="num">{sig3(result.belief.mean)}</span> and standard deviation{' '}
            <span className="num">{sig3(result.belief.sd)}</span>.{' '}
            {r.agrees
              ? 'This is the same as the product worked out by hand. The graph has no loops, so one sweep of EP gives the exact answer.'
              : 'This differs from the product worked out by hand, which the exact answer should not.'}
          </p>

          <Figure caption="The three messages of the first sweep, each as a bell curve, and the belief, which is their product. The curves are drawn at their true heights, so the narrowest curve is the tallest.">
            <MessagesFigure result={result} />
          </Figure>

          <p>
            The digital reading supplies <span className="num">{pct(r.share.digital)}</span> of the total precision. The
            dial supplies <span className="num">{pct(r.share.dial)}</span> and the prior{' '}
            <span className="num">{pct(r.share.prior)}</span>. So the belief&apos;s mean is{' '}
            <span className="num">{fixed(r.gapToDigital)}</span> degrees from the digital reading and{' '}
            <span className="num">{fixed(r.gapToDial)}</span> degrees from the dial reading. Its standard deviation,{' '}
            <span className="num">{sig3(result.belief.sd)}</span>, is smaller than the {DIGITAL_SD} of the digital
            thermometer alone. Every message adds precision, so combining the readings never leaves us less certain than
            the best one on its own.
          </p>

          <p>
            In the first sweep the temperature went from having no belief to the belief above.{' '}
            {result.settled
              ? `In sweep ${result.sweeps} the largest change of any belief was ${r.lastChange === undefined ? 'not recorded' : sig3(r.lastChange)}, and EP stopped.`
              : 'EP did not settle within its sweep limit.'}{' '}
            Here, the message of a factor depends only on that factor, so the messages do not change after the first
            sweep. In a graph with loops the messages depend on each other, and EP needs several sweeps before they stop
            changing.
          </p>

          <p>
            Press <em>Run again</em> for another room. The readings change, the messages change with them, and the
            precisions still add. The story <Link to={storyHref('two-humps')}>Two humps</Link> uses the same kind of
            messages on a model with many more factors.
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
