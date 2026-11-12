import type { Config } from 'omai-engine/trace'
import type { ReactNode } from 'react'
import { ExamplePage } from '../../../shared/ExamplePage.tsx'
import { HierarchicalCheck, HierarchicalObserve, hierarchicalPanels } from '../../../shared/HierarchicalPanels.tsx'
import { radonExample, radonProducer } from './method.ts'

// The reference: a long NUTS run of the same data (non-centred where the model is partial pooling), in a worker of its own.
const REFERENCE: Config = { method: 'nuts', numChains: 2, numWarmup: 300, numDraws: 700 }

export default function RadonPage(): ReactNode {
  return (
    <ExamplePage
      id="radon"
      example={radonExample as never}
      producer={radonProducer}
      panels={hierarchicalPanels('Shrinkage: county intercepts against county size')}
      reference={REFERENCE}
      observe={(view) => (
        <HierarchicalObserve
          view={view}
          prefix="radon"
          logX
          estimatesNote={
            <>
              Each county's raw estimate (diamond, from its own houses alone) is pulled towards the population mean
              (dashed line) to its posterior mean (dot) with its interval: the counties with the fewest houses, on the
              left, have the least data behind them and borrow most from the rest. The crosses are the true intercepts
              of the generated data. Unpooled leaves every county at its raw estimate, pooled puts every county on one
              value.
            </>
          }
          pairsNote={
            <>
              The hyperparameters (and the intercept of the smallest county): the spread tau of the counties is only
              weakly identified, and with the centred model the smallest county's intercept and log tau form a funnel
              whose neck draws the sampler's divergent transitions (red).
            </>
          }
        />
      )}
      check={(view) => (
        <HierarchicalCheck
          view={view}
          prefix="radon"
          explain={
            <>
              The reference is a long NUTS run (2 chains, 300 warm-up, 700 draws) of the same data, non-centred where
              the model is partial pooling, in a worker of its own once this run has ended. The check puts the chosen
              run's posterior against it, coordinate by coordinate: the difference of the means in the reference's
              spreads, and in Monte Carlo errors for samplers.
            </>
          }
        />
      )}
    />
  )
}
