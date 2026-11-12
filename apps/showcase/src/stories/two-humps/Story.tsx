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
import { waits } from './data.ts'
import { HumpsFigure, SettlingFigure } from './Figures.tsx'
import { NEAR, readData, readResult, SURE } from './reading.ts'

// The data are a table lookup, so the page shows them before anything runs.
const WAITS = waits()
const DATA = readData(WAITS)

const fixed = (v: number, digits = 1): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(100 * v)}%`

/** 'Two kinds of wait at Old Faithful': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        Old Faithful is a geyser in Yellowstone. Someone wrote down the wait before each of{' '}
        <span className="num">{DATA.rows}</span> eruptions, in minutes. The shortest wait was{' '}
        <span className="num">{DATA.shortest}</span> and the longest was <span className="num">{DATA.longest}</span>.
        The figure counts the waits in two-minute slots.
      </p>

      <Figure
        caption={`The ${DATA.rows} waits, counted in two-minute slots. Each grey dot under the bars is one eruption.`}
        wide
      >
        <HumpsFigure waits={WAITS} result={undefined} />
      </Figure>

      <p>
        The counts form two humps with a dip between them. The average wait is{' '}
        <span className="num">{fixed(DATA.average)}</span> minutes, which is in the dip. Only{' '}
        <span className="num">{DATA.nearAverage}</span> of the {DATA.rows} waits fall within {NEAR} minutes of it. So
        there are two ways to read the data. Either there is one kind of eruption with an uneven spread of waits, or
        there are two kinds, a short wait and a long wait, mixed together.
      </p>

      <h2>Two bell curves and a hidden label</h2>

      <p>
        To test the second reading, we build a model of it. There are two kinds of wait. Each kind spreads out as a bell
        curve around its own centre. The two curves share one width. Some fraction of the eruptions belong to the first
        kind, and the rest belong to the second. Nobody recorded which eruption is which. That is the hidden part: a
        label for each of the {DATA.rows} eruptions, first kind or second kind. The model has to work out the labels
        from the waits alone.
      </p>

      <Aside>
        Two bell curves with unknown centres can swap places without changing the fit. To stop the names swapping from
        one run to the next, the model keeps the centres in order. Curve one is always the one with the lower centre.
      </Aside>

      <h2>Expectation propagation</h2>

      <p>
        The method is called expectation propagation. The parts of the model pass messages to each other. A message is a
        short summary of what one part believes: a centre and a width, not a whole distribution. The method starts with
        both curves at the average of all the waits. Then it repeats three steps.
      </p>

      <ol>
        <li>Each eruption looks at the two curves as they stand and works out how well its wait fits each one.</li>
        <li>
          Each curve collects these results from all the eruptions. It moves its centre toward the waits that fit it
          best, and it updates its width and its share of the eruptions.
        </li>
        <li>The new curves are sent back to the eruptions, and the next round starts.</li>
      </ol>

      <p>
        One round of these steps is called a sweep. The first sweep gives a rough answer. Each later sweep corrects the
        one before. The method stops when a new sweep changes nothing. No random numbers are used. The one arbitrary
        choice is the order in which the eruptions are handled, and the answer should not depend on it.
      </p>

      <p>Press play to run the sweeps.</p>

      <Play run={run} label="Run the sweeps" estimate="about 5 seconds, in the background" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result, WAITS)
          return (
            <>
              <h2>What the sweeps settled on</h2>

              <p>
                The last pass took <span className="num">{result.sweeps}</span> sweeps. The figure shows the two curves
                it left, drawn over the same histogram.
              </p>

              <Figure
                caption="The same waits, with the two fitted curves (blue and orange), their sum (dark line), and each eruption coloured by the group it most likely belongs to. Grey dots could be either."
                wide
              >
                <HumpsFigure waits={WAITS} result={result} />
              </Figure>

              <p>
                The lower curve is centred at <span className="num">{fixed(result.means[0])}</span> minutes and the
                upper curve at <span className="num">{fixed(result.means[1])}</span>, a gap of{' '}
                <span className="num">{fixed(r.gap)}</span> minutes. Both have the same width, a standard deviation of
                about <span className="num">{fixed(result.spread)}</span> minutes. The lower curve takes{' '}
                <span className="num">{percent(result.shares[0])}</span> of the eruptions and the upper curve takes{' '}
                <span className="num">{percent(result.shares[1])}</span>. The dark line is their sum. It is what the
                two-kinds reading says the histogram should look like.
              </p>

              <p>
                These are fitted values and nothing more. There is no shaded margin of error. Expectation propagation is
                quick, but the uncertainty it reports tends to be too small, so bands around these curves would claim
                more than the method supports.
              </p>

              <p>
                Each dot under the bars has the colour of the group its eruption most likely belongs to.{' '}
                <span className="num">{r.lowerSure}</span> eruptions are more than {percent(SURE)} sure to be short
                waits. <span className="num">{r.upperSure}</span> are as sure to be long waits.{' '}
                <span className="num">{r.unsure}</span> stay grey
                {r.unsureFrom === undefined || r.unsureTo === undefined ? null : (
                  <>
                    . Their waits lie between <span className="num">{r.unsureFrom}</span> and{' '}
                    <span className="num">{r.unsureTo}</span> minutes, in the dip, where either curve could have
                    produced them
                  </>
                )}
                . The model does not claim to know which kind those eruptions were.
              </p>

              <p>
                Two bell curves account for the two humps. A single bell curve would have to sit at the average,{' '}
                <span className="num">{fixed(DATA.average)}</span> minutes, where only{' '}
                <span className="num">{DATA.nearAverage}</span> of the waits are.
              </p>

              <h2>How the sweeps settled</h2>

              <p>The next figure shows the two centres after each sweep.</p>

              <Figure caption="The two centres after each sweep. At sweep 0, before any message is passed, both curves sit at the average of all the waits.">
                <SettlingFigure result={result} />
              </Figure>

              <p>
                Before the first sweep both centres are at the average. After one sweep they have split, to{' '}
                <span className="num">{r.afterFirst === undefined ? '' : fixed(r.afterFirst[0])}</span> and{' '}
                <span className="num">{r.afterFirst === undefined ? '' : fixed(r.afterFirst[1])}</span> minutes. That is
                already within{' '}
                <span className="num">{r.firstSweepError === undefined ? '' : fixed(r.firstSweepError)}</span> minutes
                of where they end. Later sweeps refine them
                {r.settledBy === undefined ? null : (
                  <>
                    . By sweep <span className="num">{r.settledBy}</span> neither centre is more than a twentieth of a
                    minute from its final place
                  </>
                )}
                .
              </p>

              {result.restarted ? (
                <Aside>
                  The method restarted once, with smaller updates, because its first pass could not place some of the
                  eruptions in the dip. The figure shows the last pass.
                </Aside>
              ) : null}

              <h2>What to take from it</h2>

              <p>
                A mixture model is a common first choice when measurements may hide several kinds: customers, galaxies,
                heartbeats, geyser eruptions. The model is the small part. The messages show how it was solved. Each
                piece of the problem talks only to the pieces it touches. The same scheme works for problems that are
                too big to write down in one piece.
              </p>

              <p>
                Press <em>Run again</em> to hand the eruptions over in a new order. The first sweeps look different.
                When this was tried with other orders, the two centres stayed within a quarter of a minute of each
                other. For a story where the uncertainty is the point, read{' '}
                <Link to={storyHref('bayesian-linear-regression')} className="underline underline-offset-2">
                  A line, and how sure we are
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
