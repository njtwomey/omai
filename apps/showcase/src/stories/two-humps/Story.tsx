import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
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

/** 'Two humps': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const r = result === undefined ? undefined : readResult(result, WAITS)
  return (
    <StoryPage>
      <p>
        Old Faithful, the geyser in Yellowstone, erupts about once an hour, give or take. The &ldquo;give or take&rdquo;
        is the interesting part. Someone wrote down the wait before each of <span className="num">{DATA.rows}</span>{' '}
        eruptions, in minutes, from <span className="num">{DATA.shortest}</span> up to{' '}
        <span className="num">{DATA.longest}</span>. Counted into two-minute slots, they look like this.
      </p>

      <Figure
        caption={
          result === undefined
            ? 'The 272 waits, counted in two-minute slots. Each grey dot under the bars is one eruption.'
            : 'The same waits, with the two groups the notes settled on (blue and orange), their sum (dark line), and each eruption coloured by the group it most likely belongs to. Grey dots could be either.'
        }
        wide
      >
        <HumpsFigure waits={WAITS} result={result} />
      </Figure>

      <p>
        Two lumps, with a dip between them. The average wait is <span className="num">{fixed(DATA.average)}</span>{' '}
        minutes, and that sits in the dip between the humps: only <span className="num">{DATA.nearAverage}</span> of the{' '}
        {DATA.rows} waits fall within {NEAR} minutes of it. So the question is a fair one. Is this a single kind of
        eruption that happens to be lumpy, or two kinds, a short wait and a long one, mixed together?
      </p>

      <h2>Two bell curves, and a guess about every eruption</h2>

      <p>
        One way to find out is to build the second idea and see what it says. Suppose there are two kinds of waits. Each
        kind spreads out like a bell curve around its own centre; the two share one width; and some fraction of all the
        eruptions are of the first kind. Nobody tells us which eruption is which. That is the hidden part: a label on
        each of the {DATA.rows} eruptions, &ldquo;first kind&rdquo; or &ldquo;second kind&rdquo;, that was never written
        down and has to be worked out from the waits alone.
      </p>

      <Aside>
        Two bell curves with unknown centres can swap places without changing anything about the fit. To stop the names
        swapping from one run to the next, the model puts the centres in order: curve one is always the one with the
        lower centre.
      </Aside>

      <h2>Notes, passed back and forth</h2>

      <p>
        The method is called expectation propagation, and it works by passing notes. Every eruption looks at the two
        curves as they stand and tells each of them how well it fits: a wait of 79 minutes writes &ldquo;the upper curve
        suits me far better&rdquo;. Every curve reads the notes from its eruptions, moves its centre toward the waits
        that vouched for it, and writes back to each eruption how likely it is that the eruption is one of its own.
      </p>

      <p>
        Then everyone reads the new notes and writes again. The first round is a rough guess, the second corrects it,
        the third corrects the correction, and after enough rounds a new note says what the last one said: nobody
        changes their mind. Each round is called a sweep. No random numbers are involved. The one arbitrary choice is
        the order in which the eruptions speak, and the answer should not depend on that.
      </p>

      <Aside>
        A note here is not a whole distribution. It is a short summary of one (a centre and a width), which is what
        keeps the rounds quick enough to repeat dozens of times.
      </Aside>

      <Play run={run} label="Pass the notes" estimate="about 5 seconds, in the background" />

      {r === undefined || result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the notes are passed in a background worker, and the
          curves appear on the figure above.
        </p>
      ) : (
        <Reveal>
          <h2>What the notes settled on</h2>

          <p>
            The two curves in the figure are what was left after <span className="num">{result.sweeps}</span> sweeps.
            The lower one is centred at <span className="num">{fixed(result.means[0])}</span> minutes and the upper one
            at <span className="num">{fixed(result.means[1])}</span>, a gap of{' '}
            <span className="num">{fixed(r.gap)}</span> minutes. Both have the same width, a standard deviation of about{' '}
            <span className="num">{fixed(result.spread)}</span> minutes, and the lower one claims{' '}
            <span className="num">{percent(result.shares[0])}</span> of the eruptions, the upper one{' '}
            <span className="num">{percent(result.shares[1])}</span>. The dark line is their sum: what the two-kinds
            idea says the histogram ought to look like.
          </p>

          <p>
            These are the fitted means and spreads, and they are a best fit, nothing more. There is no shaded margin of
            error around them. Expectation propagation is quick, but the doubt it reports tends to come out too small,
            so dressing these curves in bands would claim more than the method can back up.
          </p>

          <p>
            Now look at the dots under the bars. Each eruption has taken the colour of the group it most likely belongs
            to. <span className="num">{r.lowerSure}</span> are more than {percent(SURE)} sure to be short waits,{' '}
            <span className="num">{r.upperSure}</span> are as sure to be long ones, and{' '}
            <span className="num">{r.unsure}</span> stay grey
            {r.unsureFrom === undefined || r.unsureTo === undefined ? null : (
              <>
                : waits between <span className="num">{r.unsureFrom}</span> and{' '}
                <span className="num">{r.unsureTo}</span> minutes, in the dip, where either curve could have produced
                them
              </>
            )}
            . The model does not pretend to know which kind those were.
          </p>

          <p>
            So, one group or two? Two bell curves give a tidy account of the page. A single bell curve would have had to
            sit at the average, <span className="num">{fixed(DATA.average)}</span> minutes, where only{' '}
            <span className="num">{DATA.nearAverage}</span> of the waits are.
          </p>

          <Figure caption="The two centres after each sweep. At sweep 0, before any notes, both curves sit at the average of all the waits.">
            <SettlingFigure result={result} />
          </Figure>

          <p>
            This figure is the notes settling. Before the first sweep both curves sit at the average. One sweep later
            they have split, to <span className="num">{r.afterFirst === undefined ? '' : fixed(r.afterFirst[0])}</span>{' '}
            and <span className="num">{r.afterFirst === undefined ? '' : fixed(r.afterFirst[1])}</span> minutes, already
            within <span className="num">{r.firstSweepError === undefined ? '' : fixed(r.firstSweepError)}</span>{' '}
            minutes of where they end. Everything after that is polishing
            {r.settledBy === undefined ? null : (
              <>
                : by sweep <span className="num">{r.settledBy}</span> neither centre is more than a twentieth of a
                minute from its final place
              </>
            )}
            .
          </p>

          {result.restarted ? (
            <Aside>
              On the way the method started over once, more gently, because its first pass could not place a few of the
              eruptions in the dip. The figure shows the pass that counted.
            </Aside>
          ) : null}

          <h2>Why this is worth knowing</h2>

          <p>
            Mixtures are the first thing to reach for when a pile of measurements might be hiding kinds: customers,
            galaxies, heartbeats, geyser eruptions. The model is the small part. The notes show how it was solved: each
            piece of the problem only ever talks to the pieces it touches, so the same scheme carries over to problems
            far too big to write down in one go.
          </p>

          <p>
            Press <em>Run again</em>. The eruptions are handed over in a new order, so the first sweeps look different.
            When this was tried with other orders, the two centres stayed within a quarter of a minute of each other.
            And if you want a story where the doubt is the point, read{' '}
            <Link to={storyHref('bayesian-linear-regression')} className="underline underline-offset-2">
              A line, and how sure we are
            </Link>
            .
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
