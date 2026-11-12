import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex } from '../../kit/Maths.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { CENTRES, drawNumbers, PIECES, ROWS, SHARES, WIDTH } from './data.ts'
import { NumbersFigure, PriorFigure, WeightsFigure } from './Figures.tsx'
import { REAL, readPrior, readResult, readTruth } from './reading.ts'

// The numbers are drawn once from a fixed key, so every run fits the same ones.
const NUMBERS = drawNumbers()

const fixed = (v: number, digits = 2): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(100 * v)}%`
const tenths = (v: number): string => `${(100 * v).toFixed(1)}%`
const list = (values: readonly number[], format: (v: number) => string): string => values.map(format).join(', ')

/** 'Breaking a stick': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        A mixture model says that data come from several groups. It has to be told how many. Often nobody knows. A
        stick-breaking model does not fix the number. It lets the data decide.
      </p>

      <h2>The stick</h2>

      <p>
        Take a stick of length 1. Break off a random piece. That piece is the weight of the first group. Take what is
        left and break off a random piece of that. That is the weight of the second group. Keep going.
      </p>

      <p>
        The pieces always add up to 1. Each piece is the share of the data that one group holds. Early pieces tend to be
        large, because each takes a share of a long stick. Later pieces tend to be small, because the stick has grown
        short.
      </p>

      <p>
        The stick could be broken forever, which allows any number of groups. We cut it off at {PIECES} pieces. The last
        piece takes whatever is left.
      </p>

      <Equation
        label={
          <>
            Each break <Tex>b_k</Tex> is a share of what is left, and the weight <Tex>w_k</Tex> is the piece it cuts off
          </>
        }
        reading="Each break b k is uniform between 0 and 1. The weight w k is b k times the product, over the earlier pieces j, of one minus b j."
      >
        {String.raw`b_k \sim \Beta(1, 1), \qquad w_k = b_k \prod_{j < k} (1 - b_j)`}
      </Equation>

      <h2>What we fit</h2>

      <p>
        We have <span className="num">{ROWS}</span> numbers on a line. They came from some groups, and we do not tell
        the model how many. We tell it only that each group is a bell curve of width{' '}
        <span className="num">{WIDTH}</span>. The model has to find a centre and a weight for each piece.
      </p>

      <Equation
        label={
          <>
            The model: <Tex>\mu_k</Tex> is the centre of piece <Tex>k</Tex>, and <Tex>z_i</Tex> is the piece that number{' '}
            <Tex>i</Tex> belongs to
          </>
        }
        reading={`Each centre mu k is normal with mean 0 and spread 4. Each number belongs to a piece z i chosen with chances w. The number y i is normal around the centre of its piece with spread ${WIDTH}.`}
      >
        {String.raw`\mu_k \sim \Gauss(0, 4^2), \qquad z_i \sim \Cat(w), \qquad y_i \sim \Gauss(\mu_{z_i},\, ${WIDTH}^2)`}
      </Equation>

      <p>
        We use expectation propagation. Each number sends a message to the pieces. A message is a bell curve that says
        how well the number fits each piece, given that piece&rsquo;s centre and weight. The pieces send their messages
        back. This repeats until no message changes. One round of messages is called a sweep. A piece that the data do
        not need ends up with a weight near 0.
      </p>

      <p>Press play to break some sticks and fit the numbers.</p>

      <Play run={run} label="Break the sticks" estimate="a few seconds" />

      <Results run={run}>
        {(result) => {
          const prior = readPrior(result.prior)
          const r = readResult(result)
          const truth = readTruth(result)
          return (
            <>
              <h2>Sticks from the prior</h2>

              <p>
                First, <span className="num">{prior.draws}</span> sticks broken at random, with no data. Each bar is one
                stick. Each colour is one piece, from the first to the last.
              </p>

              <Figure caption="Five sticks broken at random, before any data. Each colour is one piece, from the first to the eighth.">
                <PriorFigure result={result} />
              </Figure>

              <p>
                The first piece holds <span className="num">{tenths(prior.first)}</span> of the stick on average. The
                fifth holds <span className="num">{tenths(prior.fifth)}</span>. The last four pieces hold{' '}
                <span className="num">{tenths(prior.tail)}</span> together.{' '}
                {prior.first > prior.fifth ? 'The weights drop off, so a few groups hold most of the data. ' : ''}
                <em>Run again</em> breaks new sticks.
              </p>

              <h2>The numbers</h2>

              <p>
                The numbers came from {CENTRES.length} groups. The model is not told that. Here they are, counted in
                slots of width 1.
              </p>

              <Figure caption="The 80 numbers. Each grey dot under the bars is one number.">
                <NumbersFigure numbers={NUMBERS} result={undefined} />
              </Figure>

              <h2>The weights the model finds</h2>

              <p>
                The method settled after <span className="num">{result.sweeps}</span> sweeps. Each piece now has a mean
                weight, read from the belief. The bars show it. The thin line on each bar is one standard deviation of
                the belief about that weight.
              </p>

              <Figure caption="Blue: the mean weight of each piece, with one standard deviation. Grey: the true share of the group that piece found.">
                <WeightsFigure result={result} />
              </Figure>

              <p>
                <span className="num">{r.real.length}</span> of the {PIECES} pieces hold at least{' '}
                <span className="num">{percent(REAL)}</span> of the stick. Together they hold{' '}
                <span className="num">{percent(r.held)}</span>. The other pieces hold{' '}
                <span className="num">{percent(r.rest)}</span> together. The model used{' '}
                <span className="num">{r.real.length}</span> groups and left the rest almost empty.
              </p>

              <h2>The groups</h2>

              <p>
                Each piece with real weight is a bell curve. It sits at the piece&rsquo;s centre, and its height is the
                piece&rsquo;s weight.
              </p>

              <Figure caption="The numbers with the curve of each piece that holds real weight, and the sum of all the curves.">
                <NumbersFigure numbers={NUMBERS} result={result} />
              </Figure>

              <p>
                The curves are centred at{' '}
                <span className="num">
                  {list(
                    r.real.map((p) => p.centre),
                    (v) => fixed(v, 1),
                  )}
                </span>
                . The numbers were drawn from groups centred at{' '}
                <span className="num">{list(CENTRES, (v) => fixed(v, 1))}</span>, with shares of{' '}
                <span className="num">{list(SHARES, percent)}</span>. The nearest curve is at most{' '}
                <span className="num">{fixed(r.worst, 1)}</span> from each true centre.{' '}
                {r.real.length === r.groups
                  ? `The model found ${r.groups} groups, which is how many there are.`
                  : r.real.length > r.groups
                    ? `The model found ${r.real.length} groups, more than there are. A group may be shared by two pieces, or a small piece may sit between groups.`
                    : `The model found ${r.real.length} groups, fewer than there are. Two groups may have been merged.`}
              </p>

              <p>
                For <span className="num">{truth.matched}</span> pieces a true group is near enough to compare. The true
                share is within two standard deviations of the piece&rsquo;s weight for{' '}
                <span className="num">{truth.within}</span> of them.
              </p>

              <h2>How far to trust it</h2>

              <p>
                Expectation propagation keeps one bell curve for each break and each centre, so the spreads above may be
                a little narrow. We did not check them against a sampler. The method also depends on the order the
                numbers are handed over in. We tried 14 orders. In 11 of them the model found exactly 4 pieces with real
                weight. In the other 3 it found 5 or 6, with a small piece between two groups. Which piece takes which
                group also changes from run to run.
              </p>

              <p>
                Press <em>Run again</em> to break new sticks and hand the numbers over in a new order. For a mixture
                with a fixed number of groups, see{' '}
                <Link to={storyHref('two-humps')} className="underline underline-offset-2">
                  Two kinds of wait at Old Faithful
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
