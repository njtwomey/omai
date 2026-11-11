import { epAtom } from 'omai-engine/probprog/ep'
import type { ReactNode } from 'react'
import { EpAtomPage } from '../../shared/EpAtomPage.tsx'

export default function Page(): ReactNode {
  return <EpAtomPage atom={epAtom('bivariate-correlation-learned')} />
}
