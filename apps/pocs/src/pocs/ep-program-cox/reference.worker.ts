import { nutsReference } from './reference.ts'

type Request = { counts: number[]; prior: { amplitude: number; lengthScale: number } }

self.onmessage = (e: MessageEvent<Request>): void => {
  self.postMessage(nutsReference(e.data.counts, e.data.prior, { chains: 2, warmup: 300, draws: 1000 }))
}
