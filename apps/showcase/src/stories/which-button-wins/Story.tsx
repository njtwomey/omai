import { type ReactNode, useMemo } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
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
  const result = run.result
  const r = result === undefined ? undefined : readButtons(result)
  // The raw counts of the first run's visitors: cheap, so the page shows them before anything is computed.
  const first = useMemo(() => tally(simulate(FIRST_SEED)), [])
  const counts = result === undefined ? first : result.tally
  const callPct = Math.round(CALL * 100)
  return (
    <StoryPage>
      <p>
        You changed a button. Version A says one thing, version B says another, and you showed each of them to{' '}
        {counts.visitors.toLocaleString('en-GB')} visitors. Some clicked. Counting is easy: A got {counts.clicksA}{' '}
        clicks and B got {counts.clicksB}. What you want to know is whether that gap is the buttons or the luck of who
        happened to turn up.
      </p>

      <Figure caption="The raw tally, before any thinking. This is all the data there is: how many of each version's visitors clicked.">
        <TallyBars visitors={counts.visitors} clicksA={counts.clicksA} clicksB={counts.clicksB} />
      </Figure>

      <p>
        Each version has a true click rate, the share of all possible visitors who would click, and we never see it. We
        see a thousand visitors and what they did. A thousand is enough to have a rough idea of the rate and not enough
        to be sure of it, so each rate gets a <em>belief</em>: a curve over every possible rate, flat before the test,
        and narrowing to where the clicks point once they come in. It is the same arithmetic as the coin in{' '}
        <Link to={storyHref('is-this-coin-fair')}>Is this coin fair?</Link>, done for two buttons.
      </p>

      <h2>Asking the question directly</h2>

      <p>
        With a curve for each button, the question about the gap becomes easy to put. Pick a rate from A&apos;s curve
        and a rate from B&apos;s curve, each in proportion to how plausible it is. Look at which is higher. Do it twenty
        thousand times. The share of pairs in which B is higher is the probability that B really is the better button.
        That is a number you can use, and it is the question people mean when they ask who won.
      </p>

      <p>
        There is one more thing to look at. Test dashboards update as visitors arrive, and it is very tempting to watch
        the number and stop the moment it looks good. So the same calculation is run after the first 20 visitors of each
        version, then 40, 60 and on up to all of them, and drawn as a line. The line is where the story is.
      </p>

      <p>
        The two posteriors are exact: each is a Beta curve that the library&apos;s inference writes down from the
        counts. The probability that B is better is sampled, by comparing draws from those two curves. The code at the
        bottom of the page says which is which.
      </p>

      <Aside>
        A significance test answers a neighbouring question: how surprising would these counts be if the two buttons
        were identical? Most people hear it as the one above. Here it is asked head on.
      </Aside>

      <Play run={run} label="Run the test" estimate="about a second" />

      {r === undefined || result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the model is solved in a background worker, one version of
          the visitors at a time, and the curves and the line appear here.
        </p>
      ) : (
        <Reveal>
          <h2>The two rates, and the gap between them</h2>

          <Figure
            caption="What the visitors make plausible for each button's true click rate. The darker part holds the middle 90% of the belief, and the dashed lines are the rates the visitors were really drawn with, which the model never saw."
            wide
          >
            <RatesFigure result={result} />
          </Figure>

          <p>
            A&apos;s rate is centred near <span className="num">{rate(result.rateA.mean)}</span>, with the middle 90% of
            the belief running from <span className="num">{rate(result.rateA.lo)}</span> to{' '}
            <span className="num">{rate(result.rateA.hi)}</span>. B&apos;s is centred near{' '}
            <span className="num">{rate(result.rateB.mean)}</span>, from{' '}
            <span className="num">{rate(result.rateB.lo)}</span> to <span className="num">{rate(result.rateB.hi)}</span>
            .{' '}
            {r.overlap
              ? 'The two ranges overlap, which is why this is a question and not yet an answer: some rates fit both buttons.'
              : 'The two ranges do not even touch, so there is little to argue about.'}
          </p>

          <Figure caption="B's rate minus A's rate, for each pair of draws. Bars to the right of the line are the worlds in which B is the better button; bars to the left are the worlds in which A is. The dashed line is the real gap.">
            <DifferenceFigure result={result} />
          </Figure>

          <p>
            Of the <span className="num">{result.draws.toLocaleString('en-GB')}</span> pairs, B&apos;s rate was the
            higher in <span className="num">{pct(r.better)}</span>.{' '}
            {r.leader === 'B'
              ? 'That share is the chance that B beats A.'
              : `So B is the better button only about that often; put the other way, A is the better one with a probability of ${pct(1 - r.better)}.`}{' '}
            The typical gap is <span className="num">{points(result.difference.mean)}</span> percentage points, and nine
            in ten of the plausible gaps lie between <span className="num">{points(result.difference.lo)}</span> and{' '}
            <span className="num">{points(result.difference.hi)}</span>, a range that{' '}
            {r.excludesZero ? 'keeps clear of zero' : 'runs across zero'}.
          </p>

          <p>
            In plain words: take every way the buttons could really be, weigh each by how well it explains the clicks,
            and in <span className="num">{pct(r.leaderChance)}</span> of them, {r.leader} is the better button.{' '}
            {r.call === 'clear'
              ? `That clears the ${callPct}% that many teams ask for before they act.`
              : r.call === 'leaning'
                ? `That is a lean, not a verdict: short of the ${callPct}% that many teams ask for before they act.`
                : 'That is barely better than a coin toss, and it is an honest answer: so far the test does not say.'}{' '}
            It is a statement about the buttons, not about how unusual this sample was.
          </p>

          <p>
            The real rates were <span className="num">{rate(result.truth.rateA)}</span> for A and{' '}
            <span className="num">{rate(result.truth.rateB)}</span> for B, so B really{' '}
            {r.bReallyHigher ? 'is' : 'is not'} the better button, by{' '}
            <span className="num">{points(Math.abs(r.trueGap))}</span> points
            {r.bReallyHigher ? '' : ' the wrong way'}, and that gap {r.truthInside ? 'is' : 'is not'} inside the
            interval above.
          </p>

          <h2>When could we have called it?</h2>

          <p>
            That answer needed all {result.tally.visitors.toLocaleString('en-GB')} visitors of each version. Here is the
            same probability after fewer.
          </p>

          <Figure
            caption={`The chance that B is better, recomputed after more and more visitors of each version. The shaded bands are the ${callPct}% lines on either side, where most teams would stop and call it.`}
            wide
          >
            <StripFigure result={result} />
          </Figure>

          <p>
            Look at the left end first. Among the first hundred visitors of each version the chance that B is better
            ranged from <span className="num">{pct(r.earlyLow)}</span> to{' '}
            <span className="num">{pct(r.earlyHigh)}</span>. The buttons did not change between those points. A few
            dozen clicks are simply a small sample, and early on the line is telling you about the visitors more than
            about the buttons.
          </p>

          {r.dip === undefined ? null : (
            <p>
              {r.dip.peakVisitors <= 100 ? 'Some of those highs did not last. ' : 'The wobbles do not stop there. '}At{' '}
              <span className="num">{r.dip.peakVisitors}</span> visitors of each version the chance for {r.leader} stood
              at <span className="num">{pct(r.dip.peak)}</span>; by <span className="num">{r.dip.lowVisitors}</span> it
              had slipped to <span className="num">{pct(r.dip.low)}</span>. Anyone watching a dashboard would have seen
              a promising number and then a less convincing one, from the same two buttons.
            </p>
          )}

          <p>
            {r.firstCall === undefined ? (
              <>
                On this run the line never crossed the {callPct}% lines in either direction. Nobody could honestly have
                called it, and what a thousand visitors buy is the answer above: {pct(r.leaderChance)} for {r.leader}.
              </>
            ) : (
              <>
                The line first crossed {r.firstCall.side === 'B' ? 'the upper' : 'the lower'} {callPct}% line at{' '}
                <span className="num">{r.firstCall.visitors.toLocaleString('en-GB')}</span> visitors of each version.{' '}
                {r.firstCall.stayed
                  ? 'It never came back inside, so on this run a team that stopped at the first crossing would have been right to.'
                  : `It did come back inside afterwards, and ends at ${pct(r.better)}, so a team that stopped at the first crossing would have called it with more certainty than the rest of the data supported.`}
              </>
            )}
          </p>

          <p>
            Looking is free. It is stopping at the first good-looking number that costs you: if every test is stopped
            the first time its line pokes past 95%, some of the buttons you ship will have done nothing. The simple
            remedy is to decide how many visitors you need before the test, and to read the line before then as a guide,
            not as a result.
          </p>

          <p>
            Press <em>Run again</em> for a new pair of buttons and a new crowd of visitors. Some runs end with a clear
            winner. Some never leave the middle of the figure, and that is a result too: &ldquo;we cannot tell
            yet&rdquo; is sometimes all a thousand visitors can say.
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
