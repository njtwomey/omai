import { type ReactNode, useMemo } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { CALL, FIRST_SEED, simulate, tally } from './data.ts'
import { DifferenceFigure, RatesFigure, StripFigure, TallyBars } from './Figures.tsx'
import { pct, points, rate, readButtons } from './reading.ts'

/** 'Which button wins?': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(
    () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
    FIRST_SEED,
  )
  // The raw counts of the first run's visitors: cheap, so the page shows them before anything is computed.
  const first = useMemo(() => tally(simulate(FIRST_SEED)), [])
  const callPct = Math.round(CALL * 100)
  return (
    <StoryPage>
      <p>
        You changed a button. Version A says one thing and version B says another. You showed each version to{' '}
        {first.visitors.toLocaleString('en-GB')} visitors, and some of them clicked. A got {first.clicksA} clicks and B
        got {first.clicksB}. Did the buttons make the difference, or did the visitors who happened to arrive?
      </p>

      <Figure caption="The raw counts: how many of each version's visitors clicked.">
        <TallyBars visitors={first.visitors} clicksA={first.clicksA} clicksB={first.clicksB} />
      </Figure>

      <p>
        Each version has a true click rate: the share of all possible visitors who would click. We never see it. We see{' '}
        {first.visitors.toLocaleString('en-GB')} visitors and what they did. That gives a rough idea of the rate, and
        not enough to be sure of it.
      </p>

      <p>
        So each rate gets a <em>belief</em>: a curve over every possible rate. The curve is flat before the test. As the
        clicks come in, it narrows toward the rates the clicks support. This is the same calculation as the coin in{' '}
        <Link to={storyHref('is-this-coin-fair')}>Is this coin fair?</Link>, done for two buttons.
      </p>

      <h2>Asking the question directly</h2>

      <p>
        With a curve for each button, the question about the difference is easy to ask. Pick a rate from A&rsquo;s curve
        and a rate from B&rsquo;s curve, each in proportion to how plausible it is. Check which is higher. Repeat this
        20,000 times. The share of pairs in which B is higher is the probability that B really is the better button.
      </p>

      <p>
        Test dashboards update as visitors arrive, and it is tempting to stop as soon as the number looks good. So we
        also run the same calculation after the first 20 visitors of each version, then 40, 60 and so on up to all of
        them, and draw it as a line.
      </p>

      <p>
        The two beliefs are exact: each is a Beta curve that the library works out from the counts. The probability that
        B is better is sampled, by comparing draws from the two curves.
      </p>

      <Aside>
        A significance test asks a different question: how surprising would these counts be if the buttons were
        identical? Many people read it as the question above. Here we ask that question directly.
      </Aside>

      <p>Press play to run the test.</p>

      <Play run={run} label="Run the test" estimate="about a second" />

      <Results run={run}>
        {(result) => {
          const r = readButtons(result)
          const newCrowd = result.tally.clicksA !== first.clicksA || result.tally.clicksB !== first.clicksB
          return (
            <>
              {newCrowd ? (
                <p>
                  This run has a new pair of buttons and new visitors. A got {result.tally.clicksA} clicks out of{' '}
                  {result.tally.visitors.toLocaleString('en-GB')} and B got {result.tally.clicksB}.
                </p>
              ) : null}

              <h2>The two rates and the gap between them</h2>

              <Figure
                caption="The belief about each button's true click rate, with the middle 90% darker. The dashed lines are the rates the visitors were really drawn with."
                wide
              >
                <RatesFigure result={result} />
              </Figure>

              <p>
                A&rsquo;s rate is centred near <span className="num">{rate(result.rateA.mean)}</span>, and the middle
                90% of the belief runs from <span className="num">{rate(result.rateA.lo)}</span> to{' '}
                <span className="num">{rate(result.rateA.hi)}</span>. B&rsquo;s rate is centred near{' '}
                <span className="num">{rate(result.rateB.mean)}</span>, and runs from{' '}
                <span className="num">{rate(result.rateB.lo)}</span> to{' '}
                <span className="num">{rate(result.rateB.hi)}</span>.{' '}
                {r.overlap
                  ? 'The two ranges overlap. Some rates fit both buttons, so this is still a question and not yet an answer.'
                  : 'The two ranges do not touch, so there is little to argue about.'}
              </p>

              <Figure caption="B's rate minus A's rate, for each pair of draws. Bars right of the line are pairs where B is better, and bars left of it are pairs where A is. The dashed line is the real difference.">
                <DifferenceFigure result={result} />
              </Figure>

              <p>
                Of the <span className="num">{result.draws.toLocaleString('en-GB')}</span> pairs, B&rsquo;s rate was the
                higher in <span className="num">{pct(r.better)}</span>.{' '}
                {r.leader === 'B'
                  ? 'This share is the chance that B beats A.'
                  : `Put the other way, A is the better button with a probability of ${pct(1 - r.better)}.`}{' '}
                {r.call === 'clear'
                  ? `This is above the ${callPct}% that many teams ask for before they act.`
                  : r.call === 'leaning'
                    ? `This is a lean, not a verdict. It is below the ${callPct}% that many teams ask for before they act.`
                    : 'This is barely better than a coin toss. The test does not say yet.'}{' '}
                It is a statement about the buttons, not about how unusual this sample was.
              </p>

              <p>
                The typical difference is <span className="num">{points(result.difference.mean)}</span> percentage
                points. Nine in ten of the plausible differences lie between{' '}
                <span className="num">{points(result.difference.lo)}</span> and{' '}
                <span className="num">{points(result.difference.hi)}</span>, a range that{' '}
                {r.excludesZero ? 'does not include zero' : 'includes zero'}.
              </p>

              <p>
                The real rates were <span className="num">{rate(result.truth.rateA)}</span> for A and{' '}
                <span className="num">{rate(result.truth.rateB)}</span> for B. So B really{' '}
                {r.bReallyHigher ? 'is' : 'is not'} the better button, by{' '}
                <span className="num">{points(Math.abs(r.trueGap))}</span> points
                {r.bReallyHigher ? '' : ' the wrong way'}, and that difference {r.truthInside ? 'is' : 'is not'} inside
                the interval above.
              </p>

              <h2>When could we have called it?</h2>

              <p>
                That answer used all {result.tally.visitors.toLocaleString('en-GB')} visitors of each version. The next
                figure shows the same probability after fewer.
              </p>

              <Figure
                caption={`The chance that B is better, recomputed after more and more visitors. The shaded bands are the ${callPct}% lines, where most teams would stop.`}
                wide
              >
                <StripFigure result={result} />
              </Figure>

              <p>
                Look at the left end first. Among the first 100 visitors of each version, the chance that B is better
                ranged from <span className="num">{pct(r.earlyLow)}</span> to{' '}
                <span className="num">{pct(r.earlyHigh)}</span>. The buttons did not change between those points. A few
                dozen clicks are a small sample, and early on the line says more about the visitors than about the
                buttons.
              </p>

              {r.dip === undefined ? null : (
                <p>
                  {r.dip.peakVisitors <= 100 ? 'Some of those highs did not last. ' : 'The wobbles continue. '}At{' '}
                  <span className="num">{r.dip.peakVisitors}</span> visitors of each version the chance for {r.leader}{' '}
                  was <span className="num">{pct(r.dip.peak)}</span>. By{' '}
                  <span className="num">{r.dip.lowVisitors}</span> it had fallen to{' '}
                  <span className="num">{pct(r.dip.low)}</span>. A dashboard would have shown a promising number and
                  then a weaker one, from the same two buttons.
                </p>
              )}

              <p>
                {r.firstCall === undefined ? (
                  <>
                    On this run the line never crossed the {callPct}% lines in either direction. Nobody could honestly
                    have called it. All that {result.tally.visitors.toLocaleString('en-GB')} visitors give is the answer
                    above: <span className="num">{pct(r.leaderChance)}</span> for {r.leader}.
                  </>
                ) : (
                  <>
                    The line first crossed {r.firstCall.side === 'B' ? 'the upper' : 'the lower'} {callPct}% line at{' '}
                    <span className="num">{r.firstCall.visitors.toLocaleString('en-GB')}</span> visitors of each
                    version.{' '}
                    {r.firstCall.stayed
                      ? 'It never came back inside, so on this run a team that stopped at the first crossing would have been right to.'
                      : `It came back inside afterwards, and ends at ${pct(r.better)}. A team that stopped at the first crossing would have claimed more certainty than the rest of the data supported.`}
                  </>
                )}
              </p>

              <h2>What to take from it</h2>

              <p>
                Looking at the line is harmless. Stopping at the first good-looking number is the problem. If every test
                stops the first time its line passes {callPct}%, some of the buttons you ship will have done nothing.
                The remedy is to decide before the test how many visitors you need, and to read the line before then as
                a guide, not a result.
              </p>

              <p>
                Press <em>Run again</em> for a new pair of buttons and new visitors. Some runs end with a clear winner.
                Some stay in the middle of the figure. That is also a result: these visitors cannot tell yet.
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
