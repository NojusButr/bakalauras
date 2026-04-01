import { useEffect, useState } from 'react'
import type { GeoJsonObject } from 'geojson'

export type SimulationStatus = 'idle' | 'loading' | 'loaded' | 'error'

export interface PresetInfo {
  key: string
  name: string
  description: string
  event_count: number
}

export interface SimulationMeta {
  filename?: string
  scenario_name: string
  scenario_description: string
  timestamp: string
  total_edges_affected: number
  direct_edges?: number
  propagated_edges?: number
  propagation_depth?: number
  propagation_decay?: number
  is_simulation: boolean
}

export function useSimulation(city: string) {
  const [simData, setSimData]       = useState<GeoJsonObject | null>(null)
  const [status, setStatus]         = useState<SimulationStatus>('idle')
  const [presets, setPresets]       = useState<PresetInfo[]>([])
  const [simulations, setSimulations] = useState<SimulationMeta[]>([])
  const [activeMeta, setActiveMeta] = useState<SimulationMeta | null>(null)

  // Propagation params — controlled by sliders in SimulationPanel
  const [propDepth, setPropDepth]   = useState(3)
  const [propDecay, setPropDecay]   = useState(0.55)

  const cityKey = city.toLowerCase()

  async function fetchPresets() {
    try {
      const res = await fetch(`http://localhost:8000/simulation/presets/${cityKey}`)
      if (!res.ok) return
      const data = await res.json()
      setPresets(data.presets ?? [])
    } catch { /* silent */ }
  }

  async function fetchSimulationList() {
    try {
      const res = await fetch(`http://localhost:8000/simulation/list/${cityKey}`)
      if (!res.ok) return
      const data = await res.json()
      setSimulations(data.simulations ?? [])
    } catch { /* silent */ }
  }

  async function runPreset(presetKey: string) {
    setStatus('loading')
    try {
      const params = new URLSearchParams({
        propagation_depth: String(propDepth),
        propagation_decay: String(propDecay),
      })
      const res = await fetch(
        `http://localhost:8000/simulation/preset/${cityKey}/${presetKey}?${params}`,
        { method: 'POST' }
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setSimData(data)
      setActiveMeta(data.metadata)
      setStatus('loaded')
      fetchSimulationList()
    } catch (err) {
      console.error('Simulation failed:', err)
      setStatus('error')
    }
  }

  async function loadSavedSimulation(filename: string) {
    setStatus('loading')
    try {
      const res = await fetch(`http://localhost:8000/simulation/${cityKey}/${filename}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setSimData(data)
      setActiveMeta(data.metadata)
      setStatus('loaded')
    } catch (err) {
      console.error('Failed to load simulation:', err)
      setStatus('error')
    }
  }

  function clearSimulation() {
    setSimData(null)
    setActiveMeta(null)
    setStatus('idle')
  }

  useEffect(() => {
    clearSimulation()
    fetchPresets()
    fetchSimulationList()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city])

  return {
    simData, status, presets, simulations, activeMeta,
    propDepth, setPropDepth,
    propDecay, setPropDecay,
    runPreset, loadSavedSimulation, clearSimulation,
  }
}