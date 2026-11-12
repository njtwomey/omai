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
  return (
    <StoryPage>
      <p>
        Someone gives you a coin and says it might not be fair. You cannot tell by looking, so you flip it and write
        down what comes up. The question is how many flips are enough.
      </p>

      <Figure caption="A coin, and what we want to know about it: where on this line its chance of heads falls.">
        <CoinArt />
      </Figure>

      <p>
        A coin&rsquo;s behaviour is set by one number, its <em>bias</em>: the long-run fraction of heads, from 0 (never
        heads) to 1 (always heads). A fair coin has a bias of 0.5. We will call a coin fair enough if its bias is
        between 0.45 and 0.55, and the figures show this range as a stripe.
      </p>

      <p>
        The page makes a coin with a secret bias and flips it 100 times. Then it shows which biases the flips make
        plausible.
      </p>

      <h2>A belief is a curve</h2>

      <p>
        We do not guess one value for the bias. We keep a curve that shows how plausible each value from 0 to 1 is,
        given the flips so far. Before the first flip nothing favours any value, so the curve is flat.
      </p>

      <p>
        Each flip raises the biases that would have made it likely and lowers the others. Heads move the curve to the
        right and tails to the left. Every flip adds evidence, so the curve gets narrower.
      </p>

      <Aside>
        The curve after the data is the <em>posterior</em>. The flat curve before is the <em>prior</em>. A flat prior is
        a Beta(1, 1), which says we have no idea which bias is likely.
      </Aside>

      <p>
        The model has one unknown, the bias, and every flip is a draw from it. A Beta prior and coin flips fit together
        exactly: the posterior is another Beta curve, with the heads added to one of its numbers and the tails to the
        other. omai has an engine for this case that gives the answer in closed form, with no sampling and no
        approximation.
      </p>

      <p>
        We look at the belief at six moments: before any flip, then after 1, 3, 10, 30 and 100 flips. Press play to flip
        the coin.
      </p>

      <Play run={run} label="Flip the coin 100 times" estimate="about a second" />

      <Results run={run}>
        {(result) => {
          const r = readCoin(result)
          return (
            <>
              <h2>What the flips said</h2>

              <Figure caption="The 100 flips. Filled marks are heads. The ringed ones are the flips after which the belief is read.">
                <FlipTape flips={result.flips} />
              </Figure>

              <Figure
                caption={
                  <>
                    The belief at six moments. The darker part of each curve holds the middle 90%, and the stripe is
                    &lsquo;fair&rsquo;. Compare the widths, not the peaks. The dashed line in the last panel is the true
                    bias.
                  </>
                }
              >
                <StripFigure result={result} />
              </Figure>

              <p>
                In the first panel the belief is flat, so the chance that the bias is inside the fair stripe is the
                stripe&rsquo;s width, <span className="num">{pct(r.start.fair)}</span>.
              </p>

              <p>
                The first flip came up {r.firstWasHeads ? 'heads' : 'tails'}. The curve leans toward it, but only a
                little. The 90% interval, the range that holds nine tenths of the belief, still runs from{' '}
                <span className="num">{fixed(r.one.lo)}</span> to <span className="num">{fixed(r.one.hi)}</span>. After
                three flips, with <span className="num">{heads(r.three.heads)}</span> among them, it runs from{' '}
                <span className="num">{fixed(r.three.lo)}</span> to <span className="num">{fixed(r.three.hi)}</span>, a
                width of <span className="num">{fixed(widthOf(r.three))}</span> on a scale that only goes to 1.
              </p>

              <p>
                After ten flips ({heads(r.ten.heads)}) the interval is{' '}
                <span className="num">{range(r.ten.lo, r.ten.hi)}</span>, and{' '}
                <span className="num">{pct(r.ten.fair)}</span> of the belief is in the fair stripe. After 30 flips (
                {heads(r.thirty.heads)}) the interval is <span className="num">{range(r.thirty.lo, r.thirty.hi)}</span>,
                and <span className="num">{pct(r.thirty.fair)}</span> is in the stripe. The curve narrows, but slowly.
              </p>

              <h2>What 100 flips tell us</h2>

              <p>
                After all 100 flips, <span className="num">{heads(r.last.heads)}</span> have come up. The belief is
                centred near <span className="num">{fixed(r.last.mean)}</span>, and the 90% interval is{' '}
                <span className="num">{range(r.last.lo, r.last.hi)}</span>, a width of{' '}
                <span className="num">{fixed(widthOf(r.last))}</span>. The curve puts{' '}
                <span className="num">{pct(r.last.fair)}</span> of its belief in the fair stripe.{' '}
                {r.verdict === 'unlikely'
                  ? `The flips lean ${r.leans}, and the answer is probably no. It is not a certain no, which data cannot give, but it is not a close call.`
                  : r.verdict === 'open'
                    ? `The flips lean ${r.leans}, but the answer is still open. Fair is possible, and so are coins that are not fair. 100 flips have not separated them.`
                    : 'This coin might well be fair. 100 flips cannot prove that. They can only shrink the doubt.'}
              </p>

              <p>
                The coin&rsquo;s real bias, which the computation was not told, was{' '}
                <span className="num">{fixed(result.truth, 3)}</span>.{' '}
                {r.inside
                  ? 'This is inside the final interval. A 90% interval should hold the true value about nine times in ten.'
                  : 'This is outside the final interval. A 90% interval misses about one time in ten, and this is one of those times. Run again, and it will usually hold.'}
              </p>

              <p>
                Going from 10 flips to 100 narrowed the interval by a factor of{' '}
                <span className="num">{fixed(r.narrowing, 1)}</span>, not 10. The doubt shrinks roughly with the square
                root of the number of flips, so each gain costs more than the last. How many flips are enough depends on
                how sure you need to be.
              </p>

              <p>
                Press <em>Run again</em> for another coin. The early panels differ because of the first few flips, and
                the last panels because the coins differ.
              </p>

              <p>
                The next story keeps two of these curves side by side and asks which is higher:{' '}
                <Link to={storyHref('which-button-wins')}>Which button wins?</Link>
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
