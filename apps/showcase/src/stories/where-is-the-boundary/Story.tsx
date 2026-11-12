import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { drawPoints } from './data.ts'
import { DoubtFigure, MapFigure, PointsFigure } from './Figures.tsx'
import { FAR, NEAR, readData, readResult, UNSURE } from './reading.ts'

// The points are the same every time, so the page shows them before anything runs.
const POINTS = drawPoints()
const DATA = readData(POINTS)

const fixed = (v: number, digits = 2): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(100 * v)}%`

/** 'Where is the boundary?': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const r = result === undefined ? undefined : readResult(result)
  return (
    <StoryPage>
      <p>
        Here are <span className="num">{DATA.count}</span> points in a plane. Each belongs to one of two classes: class
        1 in blue (<span className="num">{DATA.first}</span> points) and class 2 in orange (
        <span className="num">{DATA.second}</span> points). The two classes form curved bands that wrap around each
        other, and the bands overlap a little where they meet.
      </p>

      <Figure caption="Forty points of two classes.">
        <PointsFigure points={POINTS} />
      </Figure>

      <p>
        A new point arrives, and we want to know its class. A yes or no is not enough. We want the chance that it
        belongs to class 2, for any place in the plane. Near the middle of a band the chance should be close to 0 or 1.
        Where the bands meet it should be near one half. Far from every point we have seen nothing, so it should be near
        one half there as well, but for a different reason. A picture of the chance over the whole plane would show all
        of this.
      </p>

      <h2>A smooth hidden score</h2>

      <p>
        We give every place in the plane a hidden score. A positive score means class 2 is likely and a negative score
        means class 1 is likely. A score of 0 is a toss-up. The chance of class 2 follows the score along an S-shaped
        curve: near 0 for large negative scores, 1 for large positive scores and one half at 0.
      </p>

      <p>
        We do not know the score at any place, but we say it is smooth: two places that are close together have similar
        scores. How close is close is called the length scale, and it is found from the data. A smooth hidden function
        with a bell curve at every place is called a Gaussian process, and this model is a Gaussian process classifier.
      </p>

      <Aside>
        A bell curve is a normal distribution, described by its mean (the centre) and its spread (the width). A Gaussian
        process gives every place its own bell curve for the score, and links them so that nearby places move together.
      </Aside>

      <h2>How the messages work</h2>

      <p>
        The method is expectation propagation. Each point has a site: a bell curve, with a mean and a spread, for what
        that point&rsquo;s class says about its score. A message is a bell curve sent from one part of the model to
        another. The steps are these.
      </p>

      <ol className="mb-5 list-decimal">
        <li>At the start every site says nothing.</li>
        <li>
          The smoothness rule joins all the scores. Together with the sites, it gives each point a current bell curve
          for its score.
        </li>
        <li>
          Take one point. Remove its own site from its current bell curve. What is left is what all the other points say
          about this point&rsquo;s score.
        </li>
        <li>
          Now use the point&rsquo;s class. If it is class 2, the score should be positive, and the S-shaped curve says
          how strongly. Combining that with the bell curve from step 3 gives a shape that is not a bell curve. Replace
          it by the bell curve with the same mean and the same spread. The difference between this bell curve and the
          one from step 3 is the point&rsquo;s new site.
        </li>
        <li>Do this for every point. One pass over all the points is a sweep. Sweep again until no site changes.</li>
        <li>
          To find the chance at a new place, use the sites to get the mean and the spread of the score there. Then
          average the S-shaped curve over that bell curve.
        </li>
      </ol>

      <p>
        Before the sweeps start, the length scale is chosen: the method tries several values and keeps the one that
        makes the observed classes most probable.
      </p>

      <Play run={run} label="Find the boundary" estimate="about a second, in the background" />

      {result === undefined || r === undefined ? null : (
        <Reveal>
          <h2>The map</h2>

          <p>
            The sites settled after <span className="num">{result.updates}</span> single-point updates, which is{' '}
            <span className="num">{Math.round(result.updates / r.count)}</span> sweeps of {r.count} points. The length
            scale that was chosen is <span className="num">{fixed(r.lengthScale)}</span>. The map shows the chance of
            class 2 at each place.
          </p>

          <Figure
            caption="The chance that a point at each place is class 2: blue for class 1, orange for class 2, grey for a toss-up. The solid line is where the chance is one half. The dashed lines mark the chances 0.25 and 0.75."
            wide
          >
            <MapFigure result={result} />
          </Figure>

          <p>
            <span className="num">{r.right}</span> of the <span className="num">{r.count}</span> points are on the side
            of the solid line that belongs to their own class. The line winds between the two bands, as the classes do.
            Between the dashed lines, where the chance is between <span className="num">{UNSURE[0]}</span> and{' '}
            <span className="num">{UNSURE[1]}</span>, lies <span className="num">{percent(r.unsureShare)}</span> of the
            map. <span className="num">{r.unsureNear}</span> of those cells are within {NEAR} of a point, and{' '}
            <span className="num">{r.unsureFar}</span> are farther than {FAR} from every point.
          </p>

          <p>
            At the four corners of the map, which are far from every point, the chances are{' '}
            <span className="num">{fixed(r.corners[0] ?? 0)}</span> (bottom left),{' '}
            <span className="num">{fixed(r.corners[1] ?? 0)}</span> (bottom right),{' '}
            <span className="num">{fixed(r.corners[2] ?? 0)}</span> (top left) and{' '}
            <span className="num">{fixed(r.corners[3] ?? 0)}</span> (top right).{' '}
            {r.corners.every((c) => Math.abs(c - 0.5) < 0.15)
              ? 'They are close to one half, as they should be: nothing was seen there.'
              : 'Some of them are not close to one half, because the score is still extended some way beyond the points.'}
          </p>

          <h2>Two reasons for one half</h2>

          <p>
            A chance of one half can mean two different things. It can mean that the score is known to be near 0, as it
            is on the boundary between the bands. It can also mean that the score is not known at all, as it is far from
            every point. The second map shows how unsure the model is about the score itself, as the spread of the
            score&rsquo;s bell curve.
          </p>

          <Figure
            caption="The spread of the score's bell curve at each place. A stronger blue means more doubt about the score. The points are drawn on top."
            wide
          >
            <DoubtFigure result={result} />
          </Figure>

          <p>
            Near the points (within {NEAR}) the average spread of the score is{' '}
            <span className="num">{fixed(r.doubtNear)}</span>. Far from every point (farther than {FAR}) it is{' '}
            <span className="num">{fixed(r.doubtFar)}</span>. The model is surer of the score where it has seen points.
            Along the boundary the chance is one half because the classes overlap there. Far from the points it is one
            half because the model has not seen anything there, and the second map is what tells the two cases apart.
          </p>

          <p>
            The spread here is that of a single bell curve for each place, which is all that the messages carry. It is a
            summary of how sure the method is, and it is not a guarantee.
          </p>

          <h2>Why this is useful</h2>

          <p>
            A classifier that gives a chance, and says where it is unsure, tells us what to do next. A place with a high
            doubt is a good place to collect another point. A place with a chance near one half and a low doubt is one
            where the classes really overlap, and more points there will not remove the overlap.
          </p>

          <p>
            The smooth score is the same object as in{' '}
            <Link to={storyHref('functions-from-a-process')} className="underline underline-offset-2">
              Functions from a process
            </Link>
            , where it is drawn before any class is attached to it. Press <em>Run again</em> to hand the points over in
            a new order. The sweeps then take a different route, and the map should come out the same.
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
