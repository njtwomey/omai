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
import { CHECKED, DAMPING, FEATURES, FIRST_SEED, measurements, NOISE, OUTPUTS, POINTS, SWEEPS } from './data.ts'
import { CurvesFigure, PairFigure, RecoveryFigure } from './Figures.tsx'
import { readResult } from './reading.ts'

// The data of the first run are drawn before play, so the page can show them; they are made from a fixed seed.
const FIRST_DATA = measurements(FIRST_SEED)

const fixed = (v: number, digits = 2): string => v.toFixed(digits)
const OLD_PERCENT = Math.round(DAMPING * 100)

/** 'Finding hidden coordinates': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(
    () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
    FIRST_SEED,
  )
  return (
    <StoryPage>
      <p>
        Twenty points have each been measured in five ways. That is five numbers for every point, and{' '}
        <span className="num">{POINTS * OUTPUTS}</span> numbers in all. The figure shows two views of them: the first
        measurement against the second, and the third against the fourth. In both views the points lie along a curve,
        with a little scatter around it.
      </p>

      <Figure caption="Twenty points, each measured five ways. Left: the first measurement against the second. Right: the third against the fourth. Each dot is one point.">
        <PairFigure y={FIRST_DATA} />
      </Figure>

      <p>
        A curve in this picture suggests a simple cause. Suppose every point has a position along one line, and each of
        the five measurements depends smoothly on that position. Then one number per point would be enough to say where
        it is, and all five measurements would follow from it. Nobody measured that number. The task is to find it from
        the five measurements alone.
      </p>

      <p>
        A hidden number like this is called a latent variable. A model that finds latent variables and describes each
        measurement as a smooth function of them is called a Gaussian process latent variable model.
      </p>

      <Aside>
        A Gaussian process is a probability distribution over smooth functions. Instead of saying what one number is
        likely to be, it says which curves are likely.
      </Aside>

      <h2>The model</h2>

      <p>
        We write each measurement as a function of the point&rsquo;s coordinate, plus noise of a known size,{' '}
        <span className="num">{fixed(NOISE)}</span>. The function is not given. We build it from {FEATURES} waves. Each
        wave is a cosine, and its frequency and its phase are drawn at random once, before anything else. A measurement
        is a weighted sum of the {FEATURES} waves, evaluated at the point&rsquo;s coordinate. With the frequencies drawn
        in this way, such sums behave like draws from a Gaussian process, and with more waves they match it more
        closely.
      </p>

      <Equation
        label={
          <>
            Measurement <Tex>n</Tex> of point <Tex>i</Tex>, from the coordinate <Tex>x_i</Tex> and the weights{' '}
            <Tex>{'w_{nk}'}</Tex>
          </>
        }
        reading={`The measurement y i n is the sum over the ${FEATURES} waves k of the weight w n k times the cosine of omega k times x i plus phi k, plus noise with spread ${fixed(NOISE)}.`}
      >
        {String.raw`y_{in} = \sum_{k=1}^{${FEATURES}} w_{nk} \cos(\omega_k x_i + \phi_k) + \varepsilon_{in},
          \qquad \varepsilon_{in} \sim \Gauss(0, ${fixed(NOISE)}^2)`}
      </Equation>

      <p>
        Two things are unknown. One is the coordinate of every point, <span className="num">{POINTS}</span> numbers. The
        other is the weight of every measurement on every wave, <span className="num">{OUTPUTS * FEATURES}</span>{' '}
        numbers. Both start with a bell curve centred on zero. The point is to find the coordinates and the weights
        together, so that the sums of waves pass close to the measurements.
      </p>

      <h2>What the method does</h2>

      <p>
        The method is expectation propagation. The model is a graph. Each point has a node for its coordinate. Each
        measurement has a node for its {FEATURES} weights. For every point there is one factor. It joins the
        point&rsquo;s coordinate to the {OUTPUTS} nodes of weights. A message is a bell curve, sent from a factor to a
        node. A bell curve has a mean and a spread. The method runs in these steps.
      </p>

      <ol className="mb-5 list-decimal">
        <li>Every node starts with its bell curve.</li>
        <li>A factor takes the bell curves of the nodes it joins. It removes its own earlier messages from them.</li>
        <li>
          The factor lays a fine grid of possible coordinates for its point. At each value on the grid, the {OUTPUTS}{' '}
          measurements are a plain weighted sum of waves. With the bell curves of the weights, the factor can work out
          how well that value explains them, and what the weights would then be.
        </li>
        <li>
          The factor sends back bell curves with the same mean and spread as what it found. The curve for the coordinate
          counts each grid value by how well it explains the measurements. The curve for each node of weights mixes the
          weights found at all the grid values. Matching the mean and the spread gives the method its name.
        </li>
        <li>Every node multiplies the messages it receives. The product is its new bell curve.</li>
        <li>
          One pass over all the factors is a sweep. Each new message is mixed with the old one, {OLD_PERCENT}% old and{' '}
          {100 - OLD_PERCENT}% new, so a step cannot overshoot. The run stops after {SWEEPS} sweeps, or sooner if no
          message changes.
        </li>
      </ol>

      <p>
        The weights are not the result we want. When the sweeps end, we read off the bell curve of every coordinate,
        line the coordinates up with the true ones, and draw the function that each measurement has become.
      </p>

      <Play run={run} label="Look for the coordinates" estimate="about 4 seconds" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          return (
            <>
              <h2>What the method found</h2>

              <p>
                The run took <span className="num">{r.sweeps}</span> sweeps.{' '}
                {r.settled
                  ? 'The messages stopped changing before the sweeps ran out.'
                  : 'The messages were still changing when the sweeps ran out, so what follows is where the method was at that point, not a settled answer.'}{' '}
                The figure compares the coordinate that the method found for each point with the true one, which the
                data were made from and the method never saw. The model cannot say which way a coordinate points, or how
                long it is, so before the comparison the found coordinates are turned over if that helps, stretched and
                shifted to lie as near the true ones as they can.
              </p>

              <Figure
                caption="The coordinate found for each point against its true coordinate, with a bar of one spread each side. A dot on the line is a point found exactly."
                wide
              >
                <RecoveryFigure result={result} />
              </Figure>

              {r.found ? (
                <p>
                  The found coordinates follow the true ones. The correlation is{' '}
                  <span className="num">{fixed(r.correlation, 3)}</span>, and the typical distance from the line is{' '}
                  <span className="num">{fixed(r.miss)}</span>, where the true coordinates run from{' '}
                  <span className="num">{fixed(r.low, 1)}</span> to <span className="num">{fixed(r.high, 1)}</span>. The
                  method gave each coordinate a spread, <span className="num">{fixed(r.spread)}</span> on average, which
                  is {r.spreadIs} the typical miss. Of the {r.points} true coordinates,{' '}
                  <span className="num">{r.inside}</span> lie inside the 90% range of the coordinate found for them.
                </p>
              ) : (
                <p>
                  This run did not find the coordinate. The correlation is{' '}
                  <span className="num">{fixed(r.correlation, 3)}</span>, and the typical distance from the line is{' '}
                  <span className="num">{fixed(r.miss)}</span>, where the true coordinates run from{' '}
                  <span className="num">{fixed(r.low, 1)}</span> to <span className="num">{fixed(r.high, 1)}</span>. The
                  method gave each coordinate a spread, <span className="num">{fixed(r.spread)}</span> on average, which
                  is {r.spreadIs} the typical miss. Of the {r.points} true coordinates,{' '}
                  <span className="num">{r.inside}</span> lie inside the 90% range of the coordinate found for them.
                  Press <em>Run again</em> to draw new points and try again.
                </p>
              )}

              <h2>The functions it learned</h2>

              <p>
                Each measurement is now a function of the found coordinate. The figure draws them one above the other.
                The dots are the measurements. The thick line is the function the method learned, a weighted sum of the
                waves. The thin dashed line is the function that the data were made from.
              </p>

              <Figure
                caption="Each measurement against the coordinate the method found. Dots: the measurements. Thick line: the function the method learned. Dashed line: the function the data were made from."
                wide
              >
                <CurvesFigure result={result} />
              </Figure>

              <p>
                At the points, the learned functions are <span className="num">{fixed(r.residual, 3)}</span> from the
                measurements, on average. The noise put into the data was <span className="num">{fixed(r.noise)}</span>.{' '}
                {r.found && r.curvesFit
                  ? 'The learned functions follow the dashed lines, and they pass about as close to the measurements as the noise allows.'
                  : r.curvesFit
                    ? 'The learned functions pass about as close to the measurements as the noise allows, but they are functions of the wrong coordinate, so they do not follow the dashed lines.'
                    : 'The learned functions miss the measurements by more than the noise, which is what a poor solution looks like.'}{' '}
                Only the best guess is drawn. A band around each line would need the method&rsquo;s spreads for the
                functions, and the messages to the weights are the least reliable part of the answer.
              </p>

              <h2>What to be careful about</h2>

              <ul className="mb-5 list-disc">
                <li>
                  <strong>The messages are bell curves.</strong> The best coordinate for a point can have two likely
                  places. A bell curve holds one place, or a wide spread that covers both. The weights of different
                  measurements are also treated one measurement at a time, so the messages leave out how they move
                  together.
                </li>
                <li>
                  <strong>The coordinate is only found up to its direction, length and position.</strong> Turn a
                  coordinate over and it describes the same curve read the other way. That is why the comparison above
                  needed the alignment.
                </li>
                <li>
                  <strong>Some runs find a poor solution.</strong> In a check of {CHECKED.runs} runs on different
                  points, {CHECKED.poor} found one. With only {FEATURES} waves, some random waves cannot follow the
                  measurements. Even so, a poor solution can pass close to the measurements, and one of the two stated
                  small spreads. Only the true coordinates show that a run failed.
                </li>
              </ul>

              <h2>Why this is useful</h2>

              <p>
                Five numbers per point became one. Measurements often depend on a few hidden causes, and finding them
                gives a way to draw, compare and order the points. Here the functions are sums of {FEATURES} waves. The
                smooth functions of a Gaussian process are the subject of{' '}
                <Link to={storyHref('functions-from-a-process')} className="underline underline-offset-2">
                  Functions drawn from a Gaussian process
                </Link>
                . The same message passing, on a different model, is in{' '}
                <Link to={storyHref('who-is-better')} className="underline underline-offset-2">
                  Who is better?
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
