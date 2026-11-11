import type { Config } from 'omai-engine/trace'
import type { ReactNode } from 'react'
import { ExamplePage } from '../../shared/ExamplePage.tsx'
import { HierarchicalCheck, HierarchicalObserve, hierarchicalPanels } from '../../shared/HierarchicalPanels.tsx'
import { schoolsExample, schoolsProducer } from './method.ts'

// The reference: a long NUTS run of the non-centred form of the same data, in a worker of its own once the chosen run has ended.
const REFERENCE: Config = { method: 'nuts', numChains: 2, numWarmup: 500, numDraws: 1500 }

export default function EightSchoolsPage(): ReactNode {
  return (
    <ExamplePage
      id="eight-schools"
      example={schoolsExample as never}
      producer={schoolsProducer}
      panels={hierarchicalPanels('Partial pooling: estimates against the raw ones')}
      reference={REFERENCE}
      observe={(view) => (
        <HierarchicalObserve
          view={view}
          prefix="schools"
          estimatesNote={
            <>
              Each school's raw estimate (diamond) is pulled towards the population mean (dashed line) to its posterior
              mean (dot): the schools with the largest standard errors, on the right, have the least data and move
              furthest. With no pooling nothing moves, with complete pooling everything lands on the line.
            </>
          }
          pairsNote={
            <>
              With the centred model the posterior is a funnel: where the spread tau is small the school effect is tied
              to the mean, and the sampler's divergent transitions (red) pile up at the neck. The non-centred model
              samples z instead, whose posterior has no funnel, and the divergences go.
            </>
          }
        />
      )}
      check={(view) => (
        <HierarchicalCheck
          view={view}
          prefix="schools"
          explain={
            <>
              The reference is a long NUTS run (2 chains, 500 warm-up, 1500 draws) of the non-centred model on the same
              data, in a worker of its own once this run has ended (for complete and no pooling, the same model, which
              has no funnel). The check puts the chosen run's posterior against it, coordinate by coordinate: the
              difference of the means in the reference's spreads, and in Monte Carlo errors for samplers. Centred and
              non-centred are the same posterior, so a sampler that works agrees within a few Monte Carlo errors. Where
              it does not (the centred model's tau when it diverges) or cannot (a mean-field fit is too narrow) the
              numbers say by how much.
            </>
          }
        />
      )}
    />
  )
}
