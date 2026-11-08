import { run } from '../qlearning/main.tsx'
import { episodes } from 'mdml-engine/trace'
const o = run()
const r = episodes(o.result).map((e) => e.return)
console.log(r.slice(0, 40).join(' '))
console.log('min after 20', Math.min(...r.slice(20)), 'count < -300', r.filter((v) => v < -300).length, 'count<-200', r.filter(v=>v<-200).length, 'count<-100', r.filter(v=>v<-100).length, 'of', r.length)
console.log(r.slice(100).filter(v=>v<-100).length, Math.min(...r.slice(100)))
