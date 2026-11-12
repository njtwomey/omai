import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { CoinArt, FlipTape, StripFigure } from './Figures.tsx'
import { readCoin, widthOf } from './reading.ts'

const fixed = (v: number, digits = 2): string => v.toFixed(digits)
/** A probability as a percentage, 'less than 1%' when it rounds to nothing, so the sentence around it can stay the same. */
const pct = (v: number): string => (v < 0.005 ? 'less than 1%' : `${Math.round(v * 100)}%`)
const heads = (n: number): string => `${n} ${n === 1 ? 'head' : 'heads'}`
const range = (lo: number, hi: number): string => `${fixed(lo)} to ${fixed(hi)}`

/** 'Is this coin fair?': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const r = result === undefined ? undefined : readCoin(result)
  return (
    <StoryPage>
      <p>
        Somebody hands you a coin and says it might not be fair. You cannot tell by looking. A coin that lands heads six
        times in ten and one that lands heads half the time sit identically on a table. The only thing to do is flip it
        and write down what comes up, and the only question is how many flips are enough.
      </p>

      <Figure caption="The coin, and the thing we want to know about it: where on this line its chance of heads falls. Nobody has told us, and this page has not flipped anything yet.">
        <CoinArt />
      </Figure>

      <p>
        Whatever the coin does is governed by one number, its <em>bias</em>: the long-run fraction of heads, from 0
        (never) to 1 (always). A fair coin has a bias of 0.5. We will call a coin fair enough if its bias is between
        0.45 and 0.55, which is the stripe in the pictures below. The page will make a coin with a secret bias, flip it
        a hundred times, and then do the one thing worth doing with the flips: say what they make plausible.
      </p>

      <h2>A belief is a curve</h2>

      <p>
        Instead of guessing one value for the bias, we keep a curve over all of them: how plausible each value from 0 to
        1 is, given the flips so far. Before the first flip nothing favours any value, so the curve is flat. Each flip
        then reshapes it, lifting the biases that would have made that flip likely and lowering the rest. Heads pull the
        curve to the right and tails to the left, and since every flip adds a little evidence the curve narrows.
      </p>

      <Aside>
        This curve is the <em>posterior</em>, the belief after the data. The flat one before is the <em>prior</em>. A
        flat prior is a Beta(1, 1); it is the plainest way to say that you have no idea.
      </Aside>

      <Figure
        caption={
          result === undefined || r === undefined ? (
            'Six moments in the life of one coin: before any flip, then after 1, 3, 10, 30 and 100. They are empty for now; the dashed line is the flat belief everyone starts from, and the stripe is what we called fair.'
          ) : (
            <>
              The belief about the coin's bias at six moments. The darker part of each curve holds the middle 90% of the
              belief, and the stripe is &lsquo;fair&rsquo;. Each panel is scaled to its own height, so compare the
              widths, not the peaks. The dashed line in the last panel is the bias the coin really had.
            </>
          )
        }
      >
        <StripFigure result={result} />
      </Figure>

      <p>
        When you press play, the page flips the coin a hundred times with a seeded random key. It then asks omai for the
        belief about the bias at each of the six moments. The model is the smallest one there is: a single unknown, the
        bias, and every flip a draw from it. A Beta prior and coin flips fit together exactly (the posterior is another
        Beta curve, with the heads added to one of its numbers and the tails to the other), and omai has an engine for
        exactly that case, which gives the answer in closed form. There is no sampling and no approximation here; the
        answer is a distribution we can put questions to.
      </p>

      <Aside>
        omai also has samplers and variational engines that approximate curves like this one. The <code>engine</code>{' '}
        option in the code at the bottom is where the exact one is asked for.
      </Aside>

      <Play run={run} label="Flip the coin 100 times" estimate="about a second" />

      {r === undefined || result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the coin is made and flipped in a background worker, and
          the six panels above fill in.
        </p>
      ) : (
        <Reveal>
          <h2>What the flips said</h2>

          <Figure caption="The hundred flips, in the order they came, cut where the strip above looks at them. Filled marks are heads; the ringed ones are the flips after which the strip looks.">
            <FlipTape flips={result.flips} />
          </Figure>

          <p>
            The first panel is the starting point, and its one number is dull but worth reading: with a flat belief, the
            chance that the bias is within a hair of fair is simply the width of that stripe,{' '}
            <span className="num">{pct(r.start.fair)}</span>. It is a baseline for everything that follows.
          </p>

          <p>
            The first flip came up {r.firstWasHeads ? 'heads' : 'tails'}, and the curve leans toward it, but only
            gently: the 90% interval, the range holding nine tenths of the belief, still runs from{' '}
            <span className="num">{fixed(r.one.lo)}</span> to <span className="num">{fixed(r.one.hi)}</span>. After
            three flips, with <span className="num">{heads(r.three.heads)}</span> among them, it runs from{' '}
            <span className="num">{fixed(r.three.lo)}</span> to <span className="num">{fixed(r.three.hi)}</span>, a
            width of <span className="num">{fixed(widthOf(r.three))}</span> on a scale that only goes to 1. That is not
            the method failing. Three flips really do tell you very little, and the curve says exactly how little.
          </p>

          <p>
            Ten flips ({heads(r.ten.heads)}) give an interval of{' '}
            <span className="num">{range(r.ten.lo, r.ten.hi)}</span>, with{' '}
            <span className="num">{pct(r.ten.fair)}</span> of the belief in the fair stripe. Thirty (
            {heads(r.thirty.heads)}) give <span className="num">{range(r.thirty.lo, r.thirty.hi)}</span> and{' '}
            <span className="num">{pct(r.thirty.fair)}</span>. The curve is narrowing, but slowly.
          </p>

          <h2>What a hundred flips buy</h2>

          <p>
            After all hundred, <span className="num">{heads(r.last.heads)}</span> have come up, the belief is centred
            near <span className="num">{fixed(r.last.mean)}</span>, and the 90% interval is{' '}
            <span className="num">{range(r.last.lo, r.last.hi)}</span>, a width of{' '}
            <span className="num">{fixed(widthOf(r.last))}</span>. The curve puts{' '}
            <span className="num">{pct(r.last.fair)}</span> of its belief in the fair stripe.{' '}
            {r.verdict === 'unlikely'
              ? `So the flips lean ${r.leans} and the answer to the question is probably no. It is not a certain no, which the data cannot give, but it is not a close call either.`
              : r.verdict === 'open'
                ? `So the flips lean ${r.leans}, but the answer is still open. Fair is a live possibility, and so are coins that are not fair; a hundred flips have not separated them.`
                : 'So this coin might well be fair, and that is as far as a hundred flips can go. The interval is wider than the stripe, so even a run that looks perfectly fair cannot prove it; it can only shrink the doubt.'}
          </p>

          <p>
            The coin&apos;s real bias, which nothing in the computation was told, was{' '}
            <span className="num">{fixed(result.truth, 3)}</span>.{' '}
            {r.inside
              ? 'That sits inside the final interval, which is what a 90% interval is for: it should hold the truth about nine times in ten.'
              : 'That falls outside the final interval. A 90% interval misses about one time in ten, and this is one of those times; press the button again and it will usually hold.'}
          </p>

          <p>
            Notice what the extra flips bought. Going from ten flips to a hundred narrowed the interval by a factor of{' '}
            <span className="num">{fixed(r.narrowing, 1)}</span>, not ten. Doubt shrinks slowly, roughly with the square
            root of the number of flips, so each new bit of certainty costs more than the last. That is the honest
            answer to &ldquo;how many flips are enough?&rdquo;: it depends how sure you need to be, and the curve will
            tell you whether you are there.
          </p>

          <p>
            Press <em>Run again</em> for another coin and another hundred flips. The early panels differ from run to run
            by the luck of the first few flips; the last ones differ because the coins do.
          </p>

          <p>
            Everything else in this collection is this picture with more unknowns. The next story keeps two of these
            curves side by side and asks which of them is higher:{' '}
            <Link to={storyHref('which-button-wins')}>Which button wins?</Link>
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
