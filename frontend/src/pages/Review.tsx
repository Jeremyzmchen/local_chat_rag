import { useParams } from 'react-router-dom'

export default function Review() {
  const { id } = useParams()
  return (
    <div className="p-6">
      <h1 className="font-headline font-black text-2xl text-on-surface uppercase tracking-tighter">
        Review — {id}
      </h1>
    </div>
  )
}
