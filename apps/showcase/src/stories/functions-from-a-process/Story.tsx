import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import { LENGTH_SCALE, NOISE, type Result } from './compute.ts'
import source from './compute.ts?raw'
import { FunctionsFigure } from './Figures.tsx'
import { readResult } from './reading.ts'

const fixed = (v: number, digits = 2): string => v.toFixed(digits)

/** 'Functions drawn from a Gaussian process': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const r = result === undefined ? undefined : readResult(result)
  return (
    <StoryPage>
      <p>
        Fitting a curve to data means choosing one. You pick a line, or a polynomial of some degree, or a spline with so
        many knots, and the data then decide the few numbers inside it. The choice of shape was made before the first
        point arrived, and it is easy to forget that it was a choice.
      </p>

      <p>
        There is another move. Don't choose a curve: describe a <em>population</em> of curves, put a probability on
        every smooth function there could be, and let the data thin the population out. What is left is not one answer
        but all the functions that are still plausible, and where they disagree is where the data have nothing to say.
      </p>

      <Figure
        caption="An empty frame, x from −3 to 3 and f(x) from −3 to 3. Every function in this story lives in it."
        wide
      >
        <FunctionsFigure />
      </Figure>

      <h2>A probability on whole functions</h2>

      <p>
        A function has a value at every <em>x</em>, which is a lot of numbers to put a probability on. The trick is that
        we only ever look at it in finitely many places, a grid of a hundred and twenty-one points across the frame
        above, and a Gaussian process says that the values at any such places are jointly normal. All that is left to
        decide is how they relate: the average value (zero, here) and how strongly the value at one <em>x</em> goes
        along with the value at another.
      </p>

      <p>
        That second part is the <em>kernel</em>. The one used here is the squared exponential: two values at nearby{' '}
        <em>x</em> move together almost perfectly, and two values far apart hardly at all, with{' '}
        <span className="num">{fixed(LENGTH_SCALE)}</span> as the distance at which the link has faded noticeably. That
        single sentence is the whole taste of the process. It says the functions are smooth, and that they bend on a
        scale of about half a unit of <em>x</em>. It says nothing about where they go.
      </p>

      <Aside>
        A Gaussian process needs no sampler. Everything it does, from drawing a function to conditioning on data, is
        linear algebra on a covariance matrix, and a Cholesky factorisation of it is the engine under both.
      </Aside>

      <p>
        When you press play, the first thing that happens is the simplest: eight functions are drawn straight from this
        prior, before any data exist. Then five points arrive. They are read, with a little noise (a standard deviation
        of <span className="num">{fixed(NOISE, 1)}</span>), off one more function from the same process, which is hidden
        from everything after. The process is then conditioned on the five readings, exactly, in one step, and eight
        functions are drawn again.
      </p>

      <Play run={run} label="Draw the functions" estimate="under a second" />

      {r === undefined || result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the functions are drawn in a background worker and appear
          below.
        </p>
      ) : (
        <Reveal>
          <h2>Before any data</h2>

          <Figure
            caption="Eight functions drawn from the prior (thin lines) and the band holding 90% of all such functions at each x (shaded)."
            wide
          >
            <FunctionsFigure grid={result.grid} crowd={result.prior} />
          </Figure>

          <p>
            Eight functions, all different, all of the same character: smooth, never jagged, swinging up and down and
            rarely straying more than a couple of units from zero. None is preferred. The shaded band is as wide at one
            end as at the other, <span className="num">{fixed(r.priorWidth)}</span> from edge to edge, because the
            process has not been told anything that could make it surer about one place than another.
          </p>

          <h2>After five points</h2>

          <Figure
            caption="The same process conditioned on the five points. Eight new functions (thin lines), their mean (thick), and the band holding 90% of them at each x (shaded)."
            wide
          >
            <FunctionsFigure grid={result.grid} crowd={result.posterior} showMean data={result.data} />
          </Figure>

          <p>
            The population has been thinned. Every one of the eight functions now goes through the points, give or take
            the noise: the farthest any of them gets from a reading, at the nearest place on the grid, is{' '}
            <span className="num">{fixed(r.largestMiss)}</span>. For the eight functions of the prior, the same measure
            is <span className="num">{fixed(r.priorMiss)}</span>.
          </p>

          <p>
            Away from the points, the functions are free to disagree, and they do. At the five points the band is only{' '}
            <span className="num">{fixed(r.atPoints)}</span> wide on average. Between them it opens up: at <em>x</em> ={' '}
            <span className="num">{fixed(r.betweenAt, 1)}</span>, the widest place between the outer points, it is{' '}
            <span className="num">{fixed(r.between)}</span> wide, and the eight functions there run from{' '}
            <span className="num">{fixed(r.betweenLow)}</span> to <span className="num">{fixed(r.betweenHigh)}</span>.{' '}
            {r.widest > r.between + 0.05 ? (
              <>
                The band is widest of all at <em>x</em> = <span className="num">{fixed(r.widestAt, 1)}</span>, beyond
                the outer points, where it is <span className="num">{fixed(r.widest)}</span> wide, against{' '}
                <span className="num">{fixed(r.priorWidth)}</span> for the prior.
              </>
            ) : (
              <>
                Beyond the outer points it can be no wider than the prior's{' '}
                <span className="num">{fixed(r.priorWidth)}</span>, which it approaches as the data fall away.
              </>
            )}
          </p>

          <h2>What to take from it</h2>

          <p>
            The thick line is the mean of the posterior, the single best guess if one is wanted. It is also, notice, the
            average of many functions that all disagree somewhere. The band is not an error bar bolted on afterwards. It
            came out of the same calculation as the mean, because conditioning on data changes the whole distribution of
            functions, not just its middle.
          </p>

          <p>
            The only thing that was chosen was the kernel and its length scale, and they decide how the functions look
            both before and after the data. A smaller length scale gives rougher functions and a band that closes up
            only very near the points. In{' '}
            <Link to={storyHref('bayesian-linear-regression')}>A line, and how sure we are</Link> the crowd was lines,
            which have two numbers to be unsure about. Here the crowd is of whole curves, and the only requirement is
            that they be smooth.
          </p>

          <p>
            Press <em>Run again</em>: the seed changes, so the functions, the five points and where they fall are all
            new, and the band closes at the new points and opens between them in the new places.
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
