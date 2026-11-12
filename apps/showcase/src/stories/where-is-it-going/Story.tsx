import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { AHEAD, DT, PUSH, READING_SD, simulate } from './data.ts'
import { ReadingsFigure, TrackFigure } from './Figures.tsx'
import { readData, readResult } from './reading.ts'

// The readings are the same every time, so the page shows them before anything runs.
const READINGS = simulate(0).readings
const DATA = readData(READINGS)

const fixed = (v: number, digits = 1): string => v.toFixed(digits)

/** 'Where is it going?': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        A sensor reads the position of an object that moves along a line. It takes a reading every{' '}
        <span className="num">{DT}</span> units of time, and each reading is off by a random error with a spread of{' '}
        <span className="num">{READING_SD}</span>. The object does not move steadily. Its velocity changes a little at
        random all the time. There are <span className="num">{DATA.count}</span> readings, and here they are.
      </p>

      <Figure caption="The sensor's readings. Each dot is one reading of the position.">
        <ReadingsFigure readings={READINGS} />
      </Figure>

      <p>
        The first reading is <span className="num">{fixed(DATA.first)}</span> and the last is{' '}
        <span className="num">{fixed(DATA.last)}</span>. The dots jump around a smooth path, and the path is what we
        want. We have three questions. Where was the object at each moment? How fast was it going? Where will it be
        next?
      </p>

      <h2>A model of the motion</h2>

      <p>
        We describe the object by two numbers, its position and its velocity. Together they are its <em>state</em>. Then
        we say how the state changes from one step to the next. The position moves by the velocity times the time step.
        The velocity changes by a small random push, of size <span className="num">{PUSH}</span>.
      </p>

      <p>
        We also say how a reading is made: it is the position plus an error with a spread of{' '}
        <span className="num">{READING_SD}</span>. Before any reading, we only know that the first position is near 0
        and the first velocity is near 1.
      </p>

      <h2>Messages along a chain</h2>

      <p>
        The states form a chain, because each state follows from the one before it. A reading is attached to its state.
        Expectation propagation sends messages along this chain. A message is a bell curve, with a mean and a spread,
        sent from one part of the model to another. The steps are these.
      </p>

      <ol className="mb-5 list-decimal">
        <li>At the start every message is empty. It says nothing.</li>
        <li>Each reading sends its state a message: a bell curve for the position, centred on the reading.</li>
        <li>
          Each state combines the messages it has and sends one forward to the next state. To make it, the state applies
          the motion rule: the position moves by the velocity, and the spread grows because of the random push. The
          forward message says what the readings so far imply about the next state.
        </li>
        <li>
          Each state also sends a message backward to the previous state. It says what the later readings imply about
          the earlier state.
        </li>
        <li>
          Every state multiplies the messages from the left, from the right and from its own reading. The product is its
          bell curve, for position and velocity together.
        </li>
        <li>One pass forward and backward is a sweep. The sweeps repeat until no message changes.</li>
      </ol>

      <p>
        Here the motion rule is linear and every error is a bell curve, so each message is exactly a bell curve. Nothing
        is approximated, and the messages settle on the exact answer. This is what the Kalman smoother computes.
      </p>

      <p>
        To predict, we add <span className="num">{AHEAD}</span> more steps at the end that have no readings. We give
        each of them a reading with an error of 10000, so large that the reading says nothing. These steps then receive
        only the forward messages from the real readings, and what comes out for them is a prediction.
      </p>

      <Play run={run} label="Follow the object" estimate="about 3 seconds, in the background" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          return (
            <>
              <h2>Where the object was</h2>

              <p>
                The messages settled after <span className="num">{result.sweeps}</span> sweeps. The figure shows the
                estimate of the position at every step: a line for the mean and a band for the 90% range. The dotted
                line is the true position, which the method never saw.
              </p>

              <Figure
                caption="The estimated position (blue) with its 90% band, the readings (grey dots), the true position (dotted) and the prediction after the last reading (orange)."
                wide
              >
                <TrackFigure result={result} />
              </Figure>

              <p>
                A reading is on average <span className="num">{fixed(r.readingError, 2)}</span> away from the true
                position. The estimated path is on average <span className="num">{fixed(r.pathError, 2)}</span> away.{' '}
                {r.pathError < r.readingError
                  ? 'The path is closer to the truth than the readings are, because each estimate also uses the readings on both sides of it.'
                  : 'Here the path is not closer to the truth than the readings are.'}{' '}
                The true position is inside the 90% band at <span className="num">{r.inside}</span> of{' '}
                <span className="num">{r.steps}</span> steps.
              </p>

              <p>
                The band is not equally wide everywhere. Its half-width is{' '}
                <span className="num">{fixed(r.firstHalf, 2)}</span> at the first step,{' '}
                <span className="num">{fixed(r.middleHalf, 2)}</span> in the middle and{' '}
                <span className="num">{fixed(r.lastHalf, 2)}</span> at the last reading. At the two ends there are
                readings on only one side, so there is less to go on.
              </p>

              <h2>Where it is going</h2>

              <p>
                After the last reading the band widens. At time <span className="num">{fixed(r.endTime)}</span>, which
                is <span className="num">{r.ahead}</span> steps after the last reading, the mean position is{' '}
                <span className="num">{fixed(r.end.mean)}</span> and the 90% range is{' '}
                <span className="num">{fixed(r.end.lo)}</span> to <span className="num">{fixed(r.end.hi)}</span>, a
                half-width of <span className="num">{fixed(r.endHalf, 2)}</span>. The band widens because each step adds
                a random push to the velocity, and an error in the velocity grows into an error in the position.
              </p>

              <p>
                The true position was <span className="num">{fixed(r.endTruth)}</span> there. It is inside the band at{' '}
                <span className="num">{r.aheadInside}</span> of the <span className="num">{r.ahead}</span> steps ahead.
              </p>

              <p>
                The velocity at the last reading is estimated at{' '}
                <span className="num">{fixed(result.velocity.last.mean, 2)}</span>, with a 90% range of{' '}
                <span className="num">{fixed(result.velocity.last.lo, 2)}</span> to{' '}
                <span className="num">{fixed(result.velocity.last.hi, 2)}</span>. The true velocity was{' '}
                <span className="num">{fixed(result.velocity.truthLast, 2)}</span>, which is{' '}
                {r.velocityInside ? 'inside' : 'outside'} that range. The velocity was never read. It is worked out from
                how the position changes.
              </p>

              <Aside>
                The steps after the last reading were made up for this page, so we can compare the prediction with what
                happened. Press <em>Run again</em> to draw them again. The readings and the estimate stay the same, and
                the true path ahead changes.
              </Aside>

              <h2>Why this is useful</h2>

              <p>
                Tracking is the standard use: a phone following its position, a robot following a wall, a rocket
                following its course. The messages make it cheap. A new reading adds one message to the chain, and we do
                not start again from the first reading.
              </p>

              <p>
                The band is only as good as the model of the motion. We told the model the size of the random pushes,
                and the data here were made with pushes of that size. An object that sometimes pushes much harder would
                leave the band more often than 1 time in 10. A similar chain of messages, with a switch for a change of
                rate instead of a motion rule, is in{' '}
                <Link to={storyHref('when-did-the-rate-change')} className="underline underline-offset-2">
                  When did the rate change?
                </Link>
                .
              </p>
              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
