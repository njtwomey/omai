import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Equation, Tex } from '../../kit/Maths.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { drawPoints, GAP_ROWS, NUTS_CHECK } from './data.ts'
import { CurvesFigure, DoubtFigure, FieldFigure, JointFigure, PointsFigure } from './Figures.tsx'
import { type OddsReading, readData, readResult } from './reading.ts'

// The points are drawn once from a fixed key, so the page shows them before anything runs.
const POINTS = drawPoints()
const DATA = readData(POINTS)

const fixed = (v: number, digits = 2): string => v.toFixed(digits)

/** What one more unit of a measurement does to the odds of yes, in a sentence that reads the weight's interval. */
function Odds(props: { readonly name: string; readonly odds: OddsReading }): ReactNode {
  const { name, odds } = props
  return (
    <p>
      The weight of {name} has a mean of <span className="num">{fixed(odds.weight.mean)}</span>. The middle 90% of its
      density lies between <span className="num">{fixed(odds.weight.lo)}</span> and{' '}
      <span className="num">{fixed(odds.weight.hi)}</span>. One more unit of {name} multiplies the odds of yes by about{' '}
      <span className="num">{fixed(odds.factor)}</span>, and by anything from{' '}
      <span className="num">{fixed(odds.low)}</span> to <span className="num">{fixed(odds.high)}</span> across that
      range. {odds.raises ? 'So a higher value makes yes more likely.' : 'So a higher value makes yes less likely.'}{' '}
      {odds.sure
        ? 'The whole range is on one side of 1, so this measurement does change the odds.'
        : 'The range includes 1, so the data do not show for sure that this measurement matters.'}
    </p>
  )
}

/** 'Logistic regression': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        Here are <span className="num">{DATA.count}</span> points. Each has two measurements, which fix where it sits on
        the plane. Each is also marked yes or no: <span className="num">{DATA.yes}</span> are yes and{' '}
        <span className="num">{DATA.no}</span> are no.
      </p>

      <Figure caption="The 40 points, marked yes or no.">
        <PointsFigure points={POINTS} />
      </Figure>

      <p>
        A new point arrives with its two measurements and no mark. We want the chance that it is a yes. We also want to
        know how sure that chance is.
      </p>

      <h2>A chance that rises across a line</h2>

      <p>
        Logistic regression gives each point a score <Tex>s</Tex>. The score is an offset <Tex>w_0</Tex>, plus a weight{' '}
        <Tex>w_1</Tex> times measurement 1 (<Tex>x_1</Tex>), plus a weight <Tex>w_2</Tex> times measurement 2 (
        <Tex>x_2</Tex>). A higher score makes yes more likely. An S-shaped curve turns the score into a chance. The
        chance is near 0 for a very low score, near 1 for a very high score and one half at a score of 0.
      </p>

      <Equation
        label="The score, and the chance of yes"
        reading="The score s is w zero plus w one times x one plus w two times x two. The chance of yes is the sigmoid of s, which is one over one plus e to the minus s."
      >
        {String.raw`s = w_0 + w_1 x_1 + w_2 x_2, \qquad p(\text{yes}) = \sigmoid(s) = \frac{1}{1 + e^{-s}}`}
      </Equation>

      <p>
        The places where the score is 0, <Tex>s = 0</Tex>, form a straight line across the plane. This is the boundary.
        The chance of yes rises smoothly as a point moves away from the boundary on the yes side.
      </p>

      <p>
        The offset and the two weights are not known. Many sets of these three numbers fit the 40 points about equally
        well. Each set gives one line.
      </p>

      <h2>A belief about the numbers</h2>

      <p>
        The Bayesian way keeps every set of the three numbers, each with a degree of belief. We start with a bell curve
        for each number, centred on 0 with a spread of 2: <Tex>\Gauss(0, 2^2)</Tex>. Each marked point then narrows the
        belief. The result is called the posterior: a belief about the three numbers together.
      </p>

      <p>
        We find the posterior with expectation propagation. Each marked point sends a message about the three numbers. A
        message is a bell curve that says what that point alone suggests. The method takes one message at a time and
        replaces it with a better one, using what all the other messages say. It repeats this until no message changes.
        One round over all the points is called a sweep.
      </p>

      <p>
        The three numbers are kept as one vector, not as three separate numbers. So the final bell curve has a mean for
        each number and also says how the numbers vary together.
      </p>

      <p>
        We draw no samples from it. The chance of yes at each place, and the density of each weight, are worked out
        directly from the bell curve.
      </p>

      <p>Press play to fit the model.</p>

      <Play run={run} label="Fit the model" estimate="under a second" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          return (
            <>
              <h2>The chance of yes</h2>

              <p>
                {result.settled ? (
                  <>
                    The method settled after <span className="num">{result.sweeps}</span> sweeps.
                  </>
                ) : (
                  <>
                    The method did not settle. It stopped after <span className="num">{result.sweeps}</span> sweeps, the
                    most it was allowed. Read its belief, the weights and the spreads below as where it stopped.
                  </>
                )}{' '}
                Its belief is one bell curve over the offset and the two weights. At each place the score is the offset
                plus the weights times the measurements, so the score also has a bell curve. Its mean and spread come
                from the belief&rsquo;s mean and covariance, including how the offset and the weights vary together.
              </p>

              <p>
                The chance of yes at a place is the S-shaped curve averaged over that bell curve. This is the expected
                chance. It averages over every offset and every pair of weights the belief allows. A rule called
                Gauss-Hermite quadrature does the averaging: it adds up the S-shaped curve at 100 chosen points.
              </p>

              <Figure
                caption="The expected chance of yes at each place: blue for no, orange for yes, grey for one half. The lines mark the chances 0.2, 0.4, 0.5 (thick), 0.6 and 0.8."
                wide
              >
                <FieldFigure points={POINTS} result={result} />
              </Figure>

              <p>
                <span className="num">{result.right}</span> of the <span className="num">{result.count}</span> points
                are on the side of the thick line that matches their mark.
              </p>

              <p>
                The 0.5 line is straight. The other lines spread apart where the model has seen little data. That is the
                extra doubt from averaging over all the weights. Along the row where <Tex>{`x_2 = ${GAP_ROWS[0]}`}</Tex>
                , in the middle of the points, the 0.2 and 0.8 lines are{' '}
                <span className="num">{fixed(result.gaps.centre, 1)}</span> apart. Along the row where{' '}
                <Tex>{`x_2 = ${GAP_ROWS[1]}`}</Tex>, near the top of the figure, they are{' '}
                <span className="num">{fixed(result.gaps.edge, 1)}</span> apart.
              </p>

              <Aside>
                The weights&rsquo; belief is the bell curve that expectation propagation found, so the averaging is over
                that bell curve.
              </Aside>

              <h2>How unsure the score is</h2>

              <p>The spread of the score at a place says how unsure the model is about it. The next map shows it.</p>

              <Figure
                caption="The spread of the score at each place. A stronger blue means more doubt. The points are drawn on top."
                wide
              >
                <DoubtFigure points={POINTS} result={result} />
              </Figure>

              <p>
                In the middle of the points the spread is <span className="num">{fixed(r.doubtMiddle)}</span>. At the
                four corners of the map it averages <span className="num">{fixed(r.doubtCorners)}</span>.{' '}
                {r.doubtCorners > r.doubtMiddle
                  ? 'The model is surer of the score where it has seen points.'
                  : 'The spread is not larger away from the points here.'}
              </p>

              <h2>The weights</h2>

              <p>
                The belief about the two weights is a density. The next figure shows it as filled contours. A stronger
                blue means a higher density. The dot is the mean.
              </p>

              <Figure caption="The joint density of the two weights. A stronger blue means a higher density. The diamond marks the true pair.">
                <JointFigure result={result} />
              </Figure>

              <p>
                The correlation between the two weights is <span className="num">{fixed(result.correlation)}</span>.{' '}
                {Math.abs(result.correlation) > 0.3
                  ? 'The contours are tilted, so the weights are not independent. The method kept this because it treated the three numbers as one vector.'
                  : 'The contours are close to round, so the weights are nearly independent.'}
              </p>

              <p>
                The next figure shows each of the three numbers on its own. The shaded part of each curve holds its
                middle 90%.
              </p>

              <Figure caption="The density of the offset and of each weight. The shaded part is the middle 90%. The dashed lines mark the true values.">
                <CurvesFigure result={result} />
              </Figure>

              <p>
                The points were drawn with an offset of <span className="num">{fixed(result.truth.offset, 1)}</span> and
                weights of <span className="num">{fixed(result.truth.first, 1)}</span> and{' '}
                <span className="num">{fixed(result.truth.second, 1)}</span>. These are the true values, marked in the
                last two figures. The true offset is {r.inside.offset ? 'inside' : 'outside'} its middle 90%. The true
                weight of measurement 1 is {r.inside.first ? 'inside' : 'outside'} its range, and the true weight of
                measurement 2 is {r.inside.second ? 'inside' : 'outside'} its range.
              </p>

              <h2>What a weight means</h2>

              <p>
                Adding one unit to a measurement adds its weight to the score. That multiplies the odds of yes by{' '}
                <Tex>{'e^{w}'}</Tex>, where <Tex>w</Tex> is the weight and <Tex>e</Tex> is about 2.72.
              </p>

              <Equation
                label="The odds of yes"
                reading="The chance of yes over the chance of no is e to the s, which is e to the w zero, times e to the w one x one, times e to the w two x two."
              >
                {String.raw`\frac{p}{1 - p} = e^{s} = e^{w_0}\, e^{w_1 x_1}\, e^{w_2 x_2}`}
              </Equation>

              <Aside>
                The odds of yes are the chance of yes divided by the chance of no. A chance of 0.8 is odds of 4 to 1.
              </Aside>

              <Odds name="measurement 1" odds={r.first} />

              <Odds name="measurement 2" odds={r.second} />

              <h2>How far to trust the spread</h2>

              <p>
                Expectation propagation fits one bell curve to the belief about the three numbers. The exact belief in
                logistic regression is slightly skewed, so the bell curve is a little too narrow. We checked this
                against a sampler (NUTS) on <span className="num">{NUTS_CHECK.dataSets}</span> data sets like this one.
                The means agreed closely. The spreads from expectation propagation were smaller, by up to about{' '}
                <span className="num">{Math.round(100 * NUTS_CHECK.narrower)}</span> percent. So the densities and
                ranges above are a little narrower than they should be. The expected chance suffers less: at{' '}
                <span className="num">{NUTS_CHECK.places}</span> places it matched an average over the sampler&rsquo;s
                draws to within <span className="num">{NUTS_CHECK.chanceWithin}</span>.
              </p>

              <p>
                Press <em>Run again</em> to hand the points over in a new order. The beliefs should come out the same.
                For a boundary that curves, see{' '}
                <Link to={storyHref('where-is-the-boundary')} className="underline underline-offset-2">
                  Where is the boundary?
                </Link>{' '}
                For the same idea with a line through numbers instead of marks, see{' '}
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
