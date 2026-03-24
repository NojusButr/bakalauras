import { useEffect, useState } from 'react'
import type { GeoJsonObject } from 'geojson'

export type TrafficStatus = 'idle' | 'loading' | 'loaded' | 'error'

export function useTrafficData(city: string) {
  const [trafficData, setTrafficData] = useState<GeoJsonObject | null>(null)
  const [status, setStatus] = useState<TrafficStatus>('idle')
  const [lastUpdated, setLastUpdated] = useState<string | null>(null)
  const [snapshots, setSnapshots] = useState<string[]>([])
  const [selectedSnapshot, setSelectedSnapshot] = useState<string | null>(null)

  async function fetchSnapshotList() {
    try {
      const res = await fetch(
        `http://localhost:8000/traffic/snapshot/${encodeURIComponent(city.toLowerCase())}/list`
      )
      if (!res.ok) return
      const data = await res.json()
      const list: string[] = data.snapshots ?? []
      setSnapshots(list)
      return list
    } catch {
      return []
    }
  }

  async function fetchSnapshot(filename: string | null) {
    setStatus('loading')
    try {
      const url = filename
        ? `http://localhost:8000/traffic/snapshot/${encodeURIComponent(city.toLowerCase())}/${filename}`
        : `http://localhost:8000/traffic/snapshot/${encodeURIComponent(city.toLowerCase())}/latest`

      const res = await fetch(url)
      if (!res.ok) {
        if (res.status === 404) { setStatus('idle'); setTrafficData(null); return }
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
      setTrafficData(data)
      setLastUpdated(data?.metadata?.timestamp ?? null)
      setSelectedSnapshot(filename)
      setStatus('loaded')
    } catch (err) {
      console.error('Failed to fetch traffic snapshot:', err)
      setStatus('error')
    }
  }

  async function triggerSnapshot() {
    setStatus('loading')
    try {
      const res = await fetch(
        `http://localhost:8000/traffic/snapshot/${encodeURIComponent(city.toLowerCase())}`,
        { method: 'POST' }
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setTrafficData(data)
      setLastUpdated(data?.metadata?.timestamp ?? null)
      setStatus('loaded')
      // Refresh snapshot list
      await fetchSnapshotList()
    } catch (err) {
      console.error('Failed to trigger snapshot:', err)
      setStatus('error')
    }
  }

  // On city change: reset and load snapshot list + latest
  useEffect(() => {
    setTrafficData(null)
    setStatus('idle')
    setLastUpdated(null)
    setSnapshots([])
    setSelectedSnapshot(null)

    fetchSnapshotList().then(list => {
      if (list && list.length > 0) {
        fetchSnapshot(null) // load latest
      }
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city])

  return {
    trafficData,
    status,
    lastUpdated,
    snapshots,
    selectedSnapshot,
    fetchSnapshot,
    triggerSnapshot,
  }
}