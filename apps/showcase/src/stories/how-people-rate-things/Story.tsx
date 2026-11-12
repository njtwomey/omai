import type { ReactNode } from 'react'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex } from '../../kit/Maths.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { drawRatings, GENEROSITY_SD, ITEMS, RATERS } from './data.ts'
import { CutsFigure, ExpectedFigure, GivenFigure } from './Figures.tsx'
import { readData, readResult } from './reading.ts'
import { REFERENCE } from './reference.ts'

// The ratings are made from a fixed seed, so the page describes them before anything runs.
const RATINGS = drawRatings()
const DATA = readData(RATINGS, RATERS, ITEMS)

const fixed = (v: number, digits = 2): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(100 * v)}%`

/** 'How people rate things': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        {RATERS} people each rated some of {ITEMS} items with a score from 1 to 5. Every rater scored an item at most
        once. There are <span className="num">{DATA.total}</span> ratings in all, and every item has between{' '}
        <span className="num">{DATA.itemMin}</span> and <span className="num">{DATA.itemMax}</span> of them. Nothing
        else is known about the raters or the items.
      </p>

      <GivenFigureBlock counts={DATA.counts} />

      <p>
        The raters do not score alike. The most generous rater gave an average of{' '}
        <span className="num">{fixed(DATA.raterHigh, 1)}</span> and the least generous an average of{' '}
        <span className="num">{fixed(DATA.raterLow, 1)}</span>. The items differ too, but an item has only a few
        ratings. An item&rsquo;s average mixes two things, how good the item is and how generous the people who happened
        to rate it were. A 4 from a generous rater says less than a 4 from a strict one.
      </p>

      <h2>The model</h2>

      <p>
        Each item has a quality <Tex>q</Tex> and each rater has a generosity <Tex>g</Tex>. Neither can be seen. When
        rater <Tex>r</Tex> rates item <Tex>i</Tex>, they form a score <Tex>s</Tex>: the quality plus the generosity plus
        some noise. The score is then turned into a rating by four cut points <Tex>c_1 &lt; c_2 &lt; c_3 &lt; c_4</Tex>.
        A score below <Tex>c_1</Tex> is rated 1, a score between <Tex>c_1</Tex> and <Tex>c_2</Tex> is rated 2, and so on
        up to 5 for a score above <Tex>c_4</Tex>.
      </p>

      <Equation
        label="Each rating is a latent score cut into five ordered bins"
        reading="The score s equals the item's quality q plus the rater's generosity g plus noise that is normal with mean 0 and spread 1. The rating is k when s lies between the cut points c k minus 1 and c k."
      >
        {String.raw`s = q_i + g_r + \varepsilon, \quad \varepsilon \sim \Gauss(0, 1), \qquad
          y = k \ \text{ when } \ c_{k-1} < s \le c_k`}
      </Equation>

      <p>
        The chance of a rating <Tex>k</Tex>, given the three quantities that decide it, is then the area of the noise
        bell curve between two cut points.
      </p>

      <Equation
        label="The chance of a rating"
        reading="The probability of rating k is Phi of c k minus q minus g, minus Phi of c k minus 1 minus q minus g, where Phi is the standard normal distribution function."
      >
        {String.raw`P(y = k) = \Phi(c_k - q_i - g_r) - \Phi(c_{k-1} - q_i - g_r)`}
      </Equation>

      <p>
        The model is told a little before it sees any rating. Every quality starts as a bell curve with a mean of 0 and
        a spread of 1. Every generosity starts with a mean of 0 and a spread of {GENEROSITY_SD}. The cut points start
        wide, with a spread of 3. The cut points are the same for every rating. What a 3 means is not decided by one
        rater or one item, so every rating carries a little information about where each cut point is.
      </p>

      <Aside>
        Adding the same number to every quality and every cut point changes no rating&rsquo;s chance. The starting bell
        curves on the qualities keep this from being a free choice, but only loosely. The doubt about this shared shift
        comes back at the end.
      </Aside>

      <h2>The messages</h2>

      <p>
        The method is expectation propagation. The model is a graph. The qualities, the generosities and the cut points
        are nodes. Each rating is a factor, and it is joined to three nodes: its item&rsquo;s quality, its rater&rsquo;s
        generosity, and the two cut points on either side of its rating. A rating of 1 has only the cut point above it,
        and a rating of 5 only the one below. Every node holds a bell curve, a mean and a spread. A message is also a
        bell curve. It is stored as two numbers, the precision (one over the variance) and the precision times the mean.
        Written like this, the product of bell curves is a sum of the two numbers, and dividing one bell curve by
        another is a difference.
      </p>

      <p>One rating is handled in these steps.</p>

      <ol className="mb-5 list-decimal">
        <li>
          Take the three beliefs the rating is joined to and divide out the message this rating sent before. What is
          left is the cavity: what the other ratings say. Without this step a rating would be counted twice.
        </li>
        <li>
          Multiply the cavity by the exact chance of this rating from the equation above. This product is the tilted
          distribution. It is not a bell curve, because the chance is a difference of two normal distribution functions.
        </li>
        <li>
          Find the mean and the spread of the tilted distribution for each of the three nodes. This is moment matching.
          The library does it by numerical integration.
        </li>
        <li>
          Divide the matched bell curve by the cavity. The result is the new message from this rating to the node.
        </li>
        <li>
          Each node adds up the messages it received, together with its starting bell curve. The sum is its new belief.
        </li>
      </ol>

      <p>
        A rating of 5 from a generous rater says the score was above <Tex>c_4</Tex>. This message raises the
        item&rsquo;s quality a little, raises the rater&rsquo;s generosity a little, and lowers <Tex>c_4</Tex>, each by
        as much as the cavity leaves room for. Every rating that is a 4 or a 5 sends a message to <Tex>c_4</Tex>, and
        the messages add. A cut point is in many ratings, so its precision grows with the number of ratings. An item is
        in only a few ratings, and its precision grows more slowly.
      </p>

      <p>
        The steps repeat over all the ratings until no belief changes by more than a small amount. The run below also
        fits the model to the first 10 ratings, the first 25 and the first 50, so the cut points can be seen as ratings
        arrive. It takes the ratings in an order drawn from the seed.
      </p>

      <Play run={run} label="Fit the model" estimate="about 20 seconds, in the background" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          const last = result.history.at(-1)?.after ?? r.total
          return (
            <>
              <h2>The cut points</h2>

              <p>
                The message passing on all <span className="num">{r.total}</span> ratings stopped after{' '}
                <span className="num">{result.final.sweeps}</span> sweeps. A sweep is one pass over the ratings. The
                figure shows what the model believed about each cut point after the first 10 ratings, the first 25, the
                first 50 and all of them.
              </p>

              <Figure
                caption="Each coloured line is the mean belief about one cut point, and the band around it is the 90% range. The dotted line is the true cut point. The vertical bar on the right is the 90% range from a sampler, on all the ratings."
                wide
              >
                <CutsFigure result={result} />
              </Figure>

              <p>
                After 10 ratings the average spread of the four cut points was{' '}
                <span className="num">{fixed(r.spreadAfter(10) ?? 0)}</span>. After 25 it was{' '}
                <span className="num">{fixed(r.spreadAfter(25) ?? 0)}</span>, after 50{' '}
                <span className="num">{fixed(r.spreadAfter(50) ?? 0)}</span>, and after {last} it was{' '}
                <span className="num">{fixed(r.spreadAfter(last) ?? 0)}</span>. The cut points are shared, so each new
                rating sent them a message and narrowed them. The early stages move a lot, and they differ from one run
                to the next because the ratings arrive in a different order. The last stage does not: it uses every
                rating.
              </p>

              <h2>The ratings the model expects</h2>

              <p>
                The fitted model can also say how many of each score it expects to see among these ratings. For each
                rating it takes the beliefs about the item, the rater and the cut points, and finds the chance of each
                score. The figure adds these chances up.
              </p>

              <Figure caption="How many times each score was given, and how many times the fitted model expects it.">
                <ExpectedFigure result={result} />
              </Figure>

              <p>
                The largest gap between a given count and an expected count is{' '}
                <span className="num">{fixed(r.largestGap, 1)}</span> ratings. This is a weak test, because the model
                was fitted to these same ratings. It shows that the cut points and the qualities together reproduce how
                often each score is used.
              </p>

              <h2>What expectation propagation got right, and what it did not</h2>

              <p>
                The ratings were made from known qualities, generosities and cut points, so the fit can be compared with
                them. The fitted qualities have a correlation of{' '}
                <span className="num">{fixed(r.qualityCorrelation)}</span> with the true ones, and the fitted
                generosities <span className="num">{fixed(r.generosityCorrelation)}</span>. The root mean square gap
                between a fitted and a true quality is <span className="num">{fixed(r.qualityError)}</span>. The true
                quality is inside its 90% range for <span className="num">{r.qualityInside}</span> of the{' '}
                <span className="num">{r.items}</span> items. For the cut points, the true value is inside the 90% range
                for <span className="num">{r.cutsInside}</span> of the 4. The generosities are recovered less well than
                the qualities because each rater gave only about{' '}
                <span className="num">{fixed(r.total / RATERS, 0)}</span> ratings, so their spreads stay close to the
                starting spread of {GENEROSITY_SD}.
              </p>

              <p>
                {r.meanRatio < 0.85
                  ? 'The spreads of the cut points are too small. '
                  : 'The spreads of the cut points are close to the sampler’s. '}
                A sampler (NUTS, {REFERENCE.chains} chains of {REFERENCE.draws} draws) was run once on these {r.total}{' '}
                ratings. It takes about three minutes, so it is not part of this page. It gives cut point spreads of{' '}
                {REFERENCE.cutSd.map((v) => fixed(v)).join(', ')}. The spreads here are{' '}
                {r.cuts.map((c) => fixed(c.sd)).join(', ')}, which is on average{' '}
                <span className="num">{percent(r.meanRatio)}</span> of the sampler&rsquo;s. The means agree with the
                sampler&rsquo;s to within <span className="num">{fixed(r.maxMeanGap)}</span>.
              </p>

              <p>
                The likely cause is in how the method stores what it knows. Each node has its own bell curve, and the
                method does not record how the nodes move together. Here they move together: raise every quality and
                every cut point by the same amount and the ratings are nearly as likely as before. The sampler sees this
                shared movement, and the cut points come out less certain. Expectation propagation keeps a separate bell
                curve for each cut point, so it reports less doubt than the sampler does. The spread of a single
                item&rsquo;s quality is <span className="num">{percent(r.qualityRatio)}</span> of the sampler&rsquo;s on
                average, much closer, because few ratings share an item.
                {r.cutsInside === r.cuts.length
                  ? ' The true cut points are inside the narrower ranges here, but this is one set of ratings.'
                  : ' Some true cut points fall outside the narrower ranges.'}
              </p>

              <p>
                What the method gives in exchange is speed. A sweep costs one small numerical integral for each rating,
                and the cost grows with the number of ratings. All four fits on this page together took less time than
                the one sampler run, on {ITEMS + RATERS + 4} unknown quantities.
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}

/** The figure of the counts before the run, with its caption. */
function GivenFigureBlock(props: { readonly counts: readonly number[] }): ReactNode {
  return (
    <Figure caption="How many times each score was given, over all the ratings.">
      <GivenFigure counts={props.counts} />
    </Figure>
  )
}
