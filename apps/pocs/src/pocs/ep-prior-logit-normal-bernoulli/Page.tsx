import { epAtom } from 'omai-engine/bayesian-examples'
import type { ReactNode } from 'react'
import { EpAtomPage } from '../../shared/EpAtomPage.tsx'

export default function Page(): ReactNode {
  return <EpAtomPage atom={epAtom('prior-logit-normal-bernoulli')} />
}
