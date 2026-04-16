import { useState, useCallback } from 'react'
import type { GeoJsonObject } from 'geojson'

export type RouteMode = 'classic' | 'traffic' | 'gnn' | 'classifier' | 'both'

export interface DegradedInfo {
  data_pct: number
  mode?: string
  total_edges: number
  edges_with_traffic: number
  edges_stripped: number
  edges_remaining: number
}

export function useRouteData(city: string) {
  const [classicRoute, setClassicRoute]       = useState<GeoJsonObject | null>(null)
  const [trafficRoute, setTrafficRoute]       = useState<GeoJsonObject | null>(null)
  const [gnnRoute, setGnnRoute]               = useState<GeoJsonObject | null>(null)
  const [classifierRoute, setClassifierRoute] = useState<GeoJsonObject | null>(null)
  const [degradedInfo, setDegradedInfo]       = useState<DegradedInfo | null>(null)
  const [degradedGeojson, setDegradedGeojson] = useState<GeoJsonObject | null>(null)
  const [isLoading, setIsLoading]             = useState(false)

  function clearRoutes() {
    setClassicRoute(null); setTrafficRoute(null); setGnnRoute(null); setClassifierRoute(null)
    setDegradedInfo(null); setDegradedGeojson(null)
  }

  const fetchRoutes = useCallback(async (
    startPoint: [number, number],
    endPoint: [number, number],
    mode: RouteMode,
    snapshot: string | null,
    simulation: string | null,
    dataPct: number,
    degradeMode: string = 'random',
    corridorWidth: number = 500,
    zoneRadius: number = 2000,
  ) => {
    setIsLoading(true)
    const body = {
      start: startPoint, end: endPoint, city,
      snapshot: snapshot ?? undefined,
      simulation: simulation ?? undefined,
      data_pct: dataPct,
      degrade_mode: degradeMode,
      corridor_width: corridorWidth,
      zone_radius: zoneRadius,
    }

    try {
      if (mode === 'both') {
        const r = await fetch('http://localhost:8000/route/compare', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        const data = await r.json()
        setClassicRoute(data.classic)
        setTrafficRoute(data.traffic)
        setGnnRoute(data.gnn ?? null)
        setClassifierRoute(data.classifier ?? null)
        setDegradedInfo(data.degraded_info ?? null)
        setDegradedGeojson(data.degraded_geojson ?? null)
      } else {
        const endpoint = { classic: 'classic', traffic: 'traffic', gnn: 'gnn', classifier: 'classifier' }[mode]
        const r = await fetch(`http://localhost:8000/route/${endpoint}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        const data = await r.json()
        setClassicRoute(mode === 'classic' ? data : null)
        setTrafficRoute(mode === 'traffic' ? data : null)
        setGnnRoute(mode === 'gnn' ? data : null)
        setClassifierRoute(mode === 'classifier' ? data : null)
        setDegradedInfo(null)
        setDegradedGeojson(null)
      }
    } catch (err) {
      console.error('Route fetch failed:', err)
    } finally {
      setIsLoading(false)
    }
  }, [city])

  return { classicRoute, trafficRoute, gnnRoute, classifierRoute, degradedInfo, degradedGeojson, isLoading, clearRoutes, fetchRoutes }
}