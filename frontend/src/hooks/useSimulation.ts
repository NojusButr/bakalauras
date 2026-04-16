import { useEffect, useState, useRef, useCallback } from 'react'
import type { GeoJsonObject } from 'geojson'

export type SimulationStatus = 'idle' | 'loading' | 'loaded' | 'error' | 'previewing'

export interface PresetInfo {
  key: string
  name: string
  description: string
  event_count: number
  user_created: boolean
}

export interface CrisisEvent {
  type: 'construction' | 'congestion' | 'damage'
  center: [number, number]
  radius_m: number
  speed_reduction_pct: number
  description: string
  edges_affected?: number
}

export interface SimulationMeta {
  filename?: string
  saved_as?: string
  scenario_name: string
  scenario_description: string
  timestamp: string
  total_edges_affected: number
  direct_edges?: number
  propagated_edges?: number
  propagation_depth?: number
  propagation_decay?: number
  is_simulation: boolean
  is_preview?: boolean
  events?: CrisisEvent[]
}

export function useSimulation(city: string) {
  const [simData, setSimData] = useState<GeoJsonObject | null>(null)
  const [status, setStatus] = useState<SimulationStatus>('idle')
  const [presets, setPresets] = useState<PresetInfo[]>([])
  const [simulations, setSimulations] = useState<SimulationMeta[]>([])
  const [activeMeta, setActiveMeta] = useState<SimulationMeta | null>(null)
  const [propDepth, setPropDepth] = useState(3)
  const [propDecay, setPropDecay] = useState(0.55)
  const [activeEvents, setActiveEvents] = useState<CrisisEvent[]>([])
  const [activePresetKey, setActivePresetKey] = useState<string | null>(null)
  const previewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const abortRef = useRef<AbortController | null>(null)

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

  const runPreview = useCallback(async (events: CrisisEvent[], depth: number, decay: number) => {
    if (events.length === 0) return
    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setStatus('previewing')
    try {
      const res = await fetch(`http://localhost:8000/simulation/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city: cityKey, events, propagation_depth: depth, propagation_decay: decay }),
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      console.log('Preview response:', {
  propagation_depth: data.metadata?.propagation_depth,
  propagated_edges: data.metadata?.propagated_edges,
  direct_edges: data.metadata?.direct_edges,
  total_features: data.features?.length,
})
      setSimData(data)
      setActiveMeta(data.metadata)
      setStatus('loaded')
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      console.error('Preview failed:', err)
      setStatus('error')
    }
  }, [cityKey])

  useEffect(() => {
    if (activeEvents.length === 0) return
    if (previewTimerRef.current) clearTimeout(previewTimerRef.current)
    previewTimerRef.current = setTimeout(() => {
      runPreview(activeEvents, propDepth, propDecay)
    }, 300)
    return () => { if (previewTimerRef.current) clearTimeout(previewTimerRef.current) }
  }, [propDepth, propDecay, activeEvents, runPreview])

  async function runPreset(presetKey: string) {
    setStatus('loading')
    setActivePresetKey(presetKey)
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
      setActiveMeta({ ...data.metadata, filename: data.metadata?.saved_as })
      setActiveEvents(data.metadata?.events ?? [])
      setStatus('loaded')
      fetchSimulationList()
    } catch (err) {
      console.error('Simulation failed:', err)
      setStatus('error')
    }
  }

  async function runCustom(events: CrisisEvent[], name = 'custom', description = '') {
    setStatus('loading')
    setActivePresetKey(null)
    try {
      const res = await fetch(`http://localhost:8000/simulation/custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          city: cityKey, events,
          scenario_name: name, scenario_description: description,
          propagation_depth: propDepth, propagation_decay: propDecay,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setSimData(data)
      setActiveMeta({ ...data.metadata, filename: data.metadata?.saved_as })
      setActiveEvents(events)
      setStatus('loaded')
      fetchSimulationList()
    } catch (err) {
      console.error('Custom simulation failed:', err)
      setStatus('error')
    }
  }

  async function loadSavedSimulation(filename: string) {
    setStatus('loading')
    setActivePresetKey(null)
    try {
      const res = await fetch(`http://localhost:8000/simulation/${cityKey}/${filename}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setSimData(data)
      setActiveMeta({ ...data.metadata, filename })
      setActiveEvents(data.metadata?.events ?? [])
      setStatus('loaded')
    } catch (err) {
      console.error('Failed to load simulation:', err)
      setStatus('error')
    }
  }

  // ── User preset management ─────────────────────────────────────────────
  async function saveAsPreset(name: string, description: string) {
    if (activeEvents.length === 0) return
    const key = name.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '')
    try {
      const res = await fetch(`http://localhost:8000/simulation/presets/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          city: cityKey, key, name, description, events: activeEvents,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      await fetchPresets()
    } catch (err) {
      console.error('Failed to save preset:', err)
    }
  }

  async function deletePreset(presetKey: string) {
    try {
      const res = await fetch(
        `http://localhost:8000/simulation/presets/${cityKey}/${presetKey}`,
        { method: 'DELETE' }
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      await fetchPresets()
    } catch (err) {
      console.error('Failed to delete preset:', err)
    }
  }

  function clearSimulation() {
    setSimData(null); setActiveMeta(null); setActiveEvents([]); setActivePresetKey(null); setStatus('idle')
  }

  function addEvent(event: CrisisEvent) {
    const newEvents = [...activeEvents, event]
    setActiveEvents(newEvents)
    runPreview(newEvents, propDepth, propDecay)
  }

  function removeEvent(index: number) {
    const newEvents = activeEvents.filter((_, i) => i !== index)
    setActiveEvents(newEvents)
    if (newEvents.length === 0) clearSimulation()
    else runPreview(newEvents, propDepth, propDecay)
  }

  function updateEvent(index: number, event: CrisisEvent) {
    const newEvents = [...activeEvents]
    newEvents[index] = event
    setActiveEvents(newEvents)
  }

  useEffect(() => {
    clearSimulation(); fetchPresets(); fetchSimulationList()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city])

  return {
    simData, status, presets, simulations, activeMeta,
    propDepth, setPropDepth, propDecay, setPropDecay,
    activeEvents, activePresetKey,
    runPreset, runCustom, loadSavedSimulation, clearSimulation,
    addEvent, removeEvent, updateEvent,
    saveAsPreset, deletePreset,
  }
}