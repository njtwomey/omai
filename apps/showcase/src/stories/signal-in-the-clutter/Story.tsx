import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { CLUTTER_READINGS, clearReadings, DECOY, DECOY_READINGS, SIGNAL, SIGNAL_READINGS } from './data.ts'
import { ReadingsFigure, SolutionFigure } from './Figures.tsx'
import { CLUTTER_ABOVE, readCase, readData } from './reading.ts'

// The first twelve readings are the same every time, so the page shows them before anything runs.
const CLEAR = clearReadings()
const DATA = readData(CLEAR.values)
const AVERAGE = CLEAR.values.reduce((a, b) => a + b, 0) / CLEAR.values.length

const fixed = (v: number, digits = 1): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(100 * v)}%`

/** 'A signal in the clutter': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const clear = result === undefined ? undefined : readCase(result.clear)
  const mixed = result === undefined ? undefined : readCase(result.ambiguous)
  // how the model's choice compares with the way the readings were made
  const garbageCalled =
    result === undefined
      ? 0
      : CLEAR.kind.filter((k, i) => k === 1 && (result.clear.clutter[i] as number) > CLUTTER_ABOVE).length
  const goodCalled =
    result === undefined
      ? 0
      : CLEAR.kind.filter((k, i) => k === 0 && (result.clear.clutter[i] as number) > CLUTTER_ABOVE).length
  const decoyChance =
    result === undefined
      ? 0
      : result.ambiguous.clutter.slice(CLEAR.values.length).reduce((a, b) => a + b, 0) / DECOY_READINGS
  const firstChance =
    result === undefined
      ? 0
      : result.ambiguous.clutter.slice(0, CLEAR.values.length).reduce((a, b) => a + b, 0) / CLEAR.values.length
  return (
    <StoryPage>
      <p>
        A sensor measures one value that we cannot see. We take <span className="num">{DATA.count}</span> readings. Most
        of them are good: each is the value plus a small error. Some are garbage. Now and then the sensor returns a
        number that has nothing to do with the value, and we cannot tell from the number alone which readings those are.
        Here are the readings, on a number line.
      </p>

      <Figure caption="The twelve readings. Each dot is one reading.">
        <ReadingsFigure values={CLEAR.values} />
      </Figure>

      <p>
        The readings run from <span className="num">{fixed(DATA.lowest)}</span> to{' '}
        <span className="num">{fixed(DATA.highest)}</span>. Their average is{' '}
        <span className="num">{fixed(AVERAGE)}</span>. That average includes the garbage, so it is probably not the
        value. To find the value we have to decide which readings to trust, and the only evidence is the readings
        themselves.
      </p>

      <h2>A model with a switch for every reading</h2>

      <p>
        We write down how the readings could have been made. A good reading is the value plus an error with a spread of
        1. A garbage reading comes from a much wider bell curve centred on 0, with a spread of about 3.2. Before we look
        at a reading, we give it a one in two chance of being garbage. The value itself starts as a wide bell curve: a
        mean of 0 and a spread of 10, so almost any value is allowed. This is the clutter problem of Thomas Minka, who
        invented expectation propagation.
      </p>

      <Aside>
        A bell curve is a normal distribution. It has a mean, which is its centre, and a spread, which is how wide it
        is. A message in this story is a bell curve, so it is two numbers.
      </Aside>

      <p>The model has a hidden switch for each reading, good or garbage. The method works in these steps.</p>

      <ol className="mb-5 list-decimal">
        <li>The value starts with its wide bell curve: mean 0, spread 10.</li>
        <li>
          One reading looks at the value&rsquo;s current bell curve, with its own earlier message taken out. It asks
          what the value would be if the reading were good, and what it would be if the reading were garbage. If good,
          the value is near the reading. If garbage, the reading says nothing and the value stays as it was. The answer
          is a mixture of those two cases, weighted by how likely each is.
        </li>
        <li>
          That mixture is not a bell curve. The reading replaces it with the single bell curve that has the same mean
          and the same spread, and sends that to the value as its message.
        </li>
        <li>The reading also keeps the weight of the garbage case. That weight is its chance of being garbage.</li>
        <li>
          Every reading does this in turn. The value multiplies the messages together to get a new bell curve. One pass
          over all the readings is a sweep, and the sweeps repeat until no message changes.
        </li>
      </ol>

      <p>
        The result is one bell curve for the value, and a chance of being garbage for every reading. Because the
        messages are single bell curves, EP always ends with a single bell curve, even when the real answer has more
        than one hump. To see what that leaves out, the library also works out the exact answer, by adding up the
        possibilities over a fine grid of values. The figure will draw both.
      </p>

      <Play run={run} label="Pass the messages" estimate="about a second, in the background" />

      {result === undefined || clear === undefined || mixed === undefined ? null : (
        <Reveal>
          <h2>The twelve readings</h2>

          <p>
            The messages settled after <span className="num">{result.clear.sweeps}</span> sweeps. The bell curve for the
            value has a mean of <span className="num">{fixed(result.clear.bell.mean, 2)}</span> and a spread of{' '}
            <span className="num">{fixed(result.clear.bell.sd, 2)}</span>. The thin line is the exact curve, with a mean
            of <span className="num">{fixed(result.clear.exact.mean, 2)}</span> and a spread of{' '}
            <span className="num">{fixed(result.clear.exact.sd, 2)}</span>. The dotted line is the value that the good
            readings were made around, <span className="num">{SIGNAL}</span>.
          </p>

          <Figure
            caption="The value after twelve readings. Thick line: the bell curve from the messages. Thin line: the exact curve. Dotted line: the true value. Grey dots are readings that the model calls garbage."
            wide
          >
            <SolutionFigure result={result.clear} truth={SIGNAL} />
          </Figure>

          <p>
            The grey dots are the readings whose chance of being garbage is above {percent(CLUTTER_ABOVE)}:{' '}
            <span className="num">{clear.clutter}</span> of {DATA.count}. The other{' '}
            <span className="num">{clear.kept}</span> have an average of{' '}
            <span className="num">{fixed(clear.keptMean, 2)}</span>. The readings were made up, so we can check the
            choice. Of the <span className="num">{CLUTTER_READINGS}</span> that really were garbage,{' '}
            <span className="num">{garbageCalled}</span> are grey. Of the <span className="num">{SIGNAL_READINGS}</span>{' '}
            good ones, <span className="num">{goodCalled}</span> are grey.
          </p>

          <h2>A second group of readings</h2>

          <p>
            Now we add <span className="num">{DECOY_READINGS}</span> more readings that all lie near{' '}
            <span className="num">{DECOY}</span>. Together they form a second group, and it could be the signal. The
            model has two explanations. Either the first twelve readings are good and the new ones are garbage, or the
            new ones are good and the first twelve are mostly garbage. The same messages are run on all{' '}
            <span className="num">{result.ambiguous.readings.length}</span> readings.
          </p>

          <Figure
            caption="The same figure with the second group added. Thick line: the bell curve from the messages. Thin line: the exact curve. Dotted line: the value the first twelve readings were made around. Grey dots are readings that the model calls garbage."
            wide
          >
            <SolutionFigure result={result.ambiguous} truth={SIGNAL} />
          </Figure>

          <p>
            The bell curve for the value now has a mean of{' '}
            <span className="num">{fixed(result.ambiguous.bell.mean, 2)}</span> and a spread of{' '}
            <span className="num">{fixed(result.ambiguous.bell.sd, 2)}</span>. The exact curve has a mean of{' '}
            <span className="num">{fixed(result.ambiguous.exact.mean, 2)}</span> and a spread of{' '}
            <span className="num">{fixed(result.ambiguous.exact.sd, 2)}</span>. The first twelve readings have an
            average garbage chance of <span className="num">{percent(firstChance)}</span>, and the new ones{' '}
            <span className="num">{percent(decoyChance)}</span>.
          </p>

          {mixed.humps.length >= 2 ? (
            <p>
              The exact curve has {mixed.humps.length} humps, at{' '}
              {mixed.humps.map((h, i) => (
                <span key={h.at}>
                  {i === 0 ? '' : i === mixed.humps.length - 1 ? ' and ' : ', '}
                  <span className="num">{fixed(h.at)}</span> (holding <span className="num">{percent(h.share)}</span> of
                  the probability)
                </span>
              ))}
              . Both explanations are possible. A single bell curve cannot show two humps.{' '}
              {mixed.inValley
                ? 'The bell curve sits where the exact curve is low, so its mean is a value that the exact answer gives little probability.'
                : `The bell curve sits on the larger hump (at ${fixed(mixed.main?.at ?? 0)}) and does not show the smaller one.`}{' '}
              Its spread of <span className="num">{fixed(result.ambiguous.bell.sd, 2)}</span> is{' '}
              {mixed.spreadRatio < 1 ? 'smaller' : 'larger'} than the exact spread of{' '}
              <span className="num">{fixed(result.ambiguous.exact.sd, 2)}</span>, so it{' '}
              {mixed.spreadRatio < 1 ? 'understates' : 'overstates'} how unsure the answer is.
            </p>
          ) : (
            <p>
              The exact curve has one main hump, at <span className="num">{fixed(mixed.humps[0]?.at ?? 0)}</span>.{' '}
              {mixed.spreadRatio < 0.8
                ? `The exact spread, ${fixed(result.ambiguous.exact.sd, 2)}, is larger than the bell curve's, ${fixed(result.ambiguous.bell.sd, 2)}. The exact curve has a tail that a single bell curve cannot draw, so the bell curve understates how unsure the answer is. `
                : 'The bell curve and the exact curve agree closely. '}
              Press <em>Run again</em> to draw the extra readings again. With some draws the two groups are closer to
              equally likely, and the exact curve then has two humps.
            </p>
          )}

          {result.ambiguous.settled ? null : (
            <Aside>
              In this run the messages had not settled when the method stopped, so the bell curve is where they were at
              that point. When two explanations are about equally likely, the messages can keep swinging between them.
            </Aside>
          )}

          <h2>Why this is useful</h2>

          <p>
            Real measurements have outliers: a sensor that glitches, a form filled in wrongly, a mislabelled example. A
            model with a switch for each reading is one way to deal with them, and the messages give every reading a
            chance of being garbage while they find the value. That is the information the grey dots show.
          </p>

          <p>
            The thing to take care over is the single bell curve. When the readings clearly point at one value, it is
            close to the exact curve, as in the first figure. When there are two likely answers, the bell curve gives
            one mean and one spread for both, and the exact curve is the better guide. Another story with messages
            between a hidden label and its observations is{' '}
            <Link to={storyHref('two-humps')} className="underline underline-offset-2">
              Two humps
            </Link>
            .
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
