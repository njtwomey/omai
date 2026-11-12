import type { ReactNode } from 'react'
import { Aside, Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { simulate } from './data.ts'
import { CloudFigure, LinesFigure } from './Figures.tsx'
import { readResult } from './reading.ts'

// The data are twelve fixed points, drawn here so the page shows them before anything runs.
const DATA = simulate()

const fixed = (v: number, digits = 2): string => v.toFixed(digits)

/** 'A line, and how sure we are': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const result = run.result
  const r = result === undefined ? undefined : readResult(result, DATA)
  return (
    <StoryPage>
      <p>
        Here are twelve measurements. Something was set along the bottom, call it <em>x</em>, and something was read off
        in response, call it <em>y</em>. The points drift upward from left to right, and none of them sits exactly on a
        line.
      </p>

      <Figure
        caption={
          result === undefined
            ? 'Twelve noisy points. The axes already reach past the data on the right; keep an eye on that empty space.'
            : 'The same points with sampled lines (thin), their average (solid), the range holding 90% of them at each x (shaded), and the line the data were drawn from (dashed).'
        }
        wide
      >
        <LinesFigure data={DATA} result={result} />
      </Figure>

      <p>
        The first thing anyone does with a scatter like this is fit a line: find the intercept and slope that pass
        closest to the points. That gives one line, and one line says nothing about how far to trust it. Twelve noisy
        points are compatible with many lines, a little steeper, a little flatter, shifted up or down, and the data
        cannot tell most of them apart.
      </p>

      <h2>A posterior is a crowd of lines</h2>

      <p>
        The Bayesian answer keeps the whole crowd. We begin by saying what we believed before looking: the intercept and
        the slope are probably not enormous (a normal centred on zero, with a spread of 5, is generous without being
        silly), and the scatter of the points about the line is positive and probably not huge. Then the data vote. A
        line that passes close to many points becomes more plausible; a line that misses them becomes less. What is left
        is a probability distribution over intercept, slope and noise: the <em>posterior</em>. One draw from it is one
        line, chosen with the probability that the prior and the data together give it.
      </p>

      <p>
        Nobody can write that distribution down in closed form once the model gets interesting, so we sample from it
        instead. The sampler here is NUTS, a Hamiltonian Monte Carlo method: it wanders over the space of (intercept,
        slope, noise) in long, informed steps and keeps the places it has been.
      </p>

      <Aside>
        NUTS needs the gradient of the log probability. The model, written out at the end of this page, is never told
        how to differentiate itself; omai gets the gradient from its own automatic differentiation.
      </Aside>

      <Play run={run} label="Draw the crowd" estimate="a few seconds" />

      {r === undefined || result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the sampler runs in a background worker and the lines
          appear in the figure above.
        </p>
      ) : (
        <Reveal>
          <h2>What came back</h2>

          <p>
            The thin lines in the figure are draws from the posterior, {result.draws.toLocaleString('en-GB')} of them
            kept, of which it shows {result.fan.length}. The solid line is their average and the shaded band holds the
            middle 90% of them at each <em>x</em>. Where the lines bunch, the data are sure of themselves. Where they
            fan out, they are not.
          </p>

          <p>
            Look at how the fan breathes. Around <em>x</em> = <span className="num">{fixed(r.centre, 1)}</span>, in the
            middle of the points, the band is only <span className="num">{fixed(r.centreWidth)}</span> wide. At{' '}
            <em>x</em> = <span className="num">{fixed(r.edge, 0)}</span>, far to the right of the last point, it is{' '}
            <span className="num">{fixed(r.edgeWidth)}</span> wide, about{' '}
            <span className="num">{fixed(r.ratio, 1)}</span> times more. Nothing was measured out there. The lines agree
            about the region the data cover and disagree about the region they do not, which is how an honest
            extrapolation ought to look.
          </p>

          <Figure caption="The same lines, as points. Each dot is one sampled line, placed by its intercept and slope. The marked dot is the line the data were drawn from.">
            <CloudFigure result={result} />
          </Figure>

          <p>
            Seen this way, the crowd is a tilted cloud. A steeper line needs a lower intercept to keep passing through
            the points, so the two numbers rise and fall against each other. That tilt is why the lines in the first
            figure pivot about the middle of the data instead of sliding up and down together: you can swing a line
            about the centre of the cloud and it still fits, and the farther you stand from the centre, the more that
            swing matters.
          </p>

          <h2>What to take from it</h2>

          <p>
            A single fit would have given a number for the slope. The posterior gives the number and the doubt: the
            slope is about <span className="num">{fixed(r.slope.mean)}</span>, and nine in ten of the plausible slopes
            lie between <span className="num">{fixed(r.slope.lo)}</span> and{' '}
            <span className="num">{fixed(r.slope.hi)}</span>. The line the data were drawn from has a slope of 0.7,
            which is {r.slopeInside ? 'inside' : 'outside'} that range. The noise comes out at about{' '}
            <span className="num">{fixed(r.noise.mean)}</span> (the truth is 0.9), with a plausible range from{' '}
            <span className="num">{fixed(r.noise.lo)}</span> to <span className="num">{fixed(r.noise.hi)}</span>. With
            twelve points that range is wide, and the posterior says so.
          </p>

          <p>
            Press <em>Run again</em>. The sampler gets a new seed, so every draw is different, and the picture hardly
            moves. That is a quiet check on the sampler: a different set of draws tells the same story.
          </p>
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}
