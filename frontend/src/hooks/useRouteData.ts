import { useEffect, useState } from 'react'
import type { GeoJsonObject } from 'geojson'

export type RouteMode = 'classic' | 'traffic' | 'both'

export function useRouteData(
  startPoint: [number, number] | null,
  endPoint: [number, number] | null,
  city: string,
  mode: RouteMode = 'both',
  snapshot: string | null = null,
  simulation: string | null = null,
) {
  const [classicRoute, setClassicRoute] = useState<GeoJsonObject | null>(null)
  const [trafficRoute, setTrafficRoute] = useState<GeoJsonObject | null>(null)

  function clearRoutes() {
    setClassicRoute(null)
    setTrafficRoute(null)
  }

  useEffect(() => {
    if (!startPoint || !endPoint) return

    const body = {
      start: startPoint,
      end: endPoint,
      city,
      snapshot: snapshot ?? undefined,
      simulation: simulation ?? undefined,
    }

    if (mode === 'both') {
      fetch('http://localhost:8000/route/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
        .then(r => r.json())
        .then(data => {
          setClassicRoute(data.classic)
          setTrafficRoute(data.traffic)
        })
        .catch(err => console.error('Route compare failed:', err))

    } else if (mode === 'classic') {
      fetch('http://localhost:8000/route/classic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
        .then(r => r.json())
        .then(data => { setClassicRoute(data); setTrafficRoute(null) })
        .catch(err => console.error('Classic route failed:', err))

    } else {
      fetch('http://localhost:8000/route/traffic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
        .then(r => r.json())
        .then(data => { setTrafficRoute(data); setClassicRoute(null) })
        .catch(err => console.error('Traffic route failed:', err))
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startPoint, endPoint, city, mode, snapshot, simulation])

  return { classicRoute, trafficRoute, clearRoutes }
}
