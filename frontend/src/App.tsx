import { useState, useEffect } from 'react'
import './App.css'
import InfoPanel from './components/InfoPanel'
import RouteMap from './components/RouteMap'
import SimulationPanel from './components/SimulationPanel'
import { useGraphData } from './hooks/useGraphData'
import { useRouteData, type RouteMode } from './hooks/useRouteData'
import { useTrafficData } from './hooks/useTrafficData'
import { useSimulation, type CrisisEvent } from './hooks/useSimulation'

export type MapMode = 'routing' | 'simulation'

function App() {
  const [city, setCity] = useState('Vilnius')
  const [startPoint, setStartPoint] = useState<[number, number] | null>(null)
  const [endPoint, setEndPoint] = useState<[number, number] | null>(null)
  const [showTraffic, setShowTraffic] = useState(false)
  const [showIncidents, setShowIncidents] = useState(true)
  const [showSimulation, setShowSimulation] = useState(true)
  const [routeMode, setRouteMode] = useState<RouteMode>('both')
  const [dataPct, setDataPct] = useState(100)
  const [degradeMode, setDegradeMode] = useState('random')
  const [corridorWidth, setCorridorWidth] = useState(500)
  const [zoneRadius, setZoneRadius] = useState(2000)
  const [showDegraded, setShowDegraded] = useState(true)
  const [mapMode, setMapMode] = useState<MapMode>('routing')
  const [placingEventType, setPlacingEventType] = useState<string>('damage')
  const [defaultSpeed, setDefaultSpeed] = useState(0)

  const graphData = useGraphData(city)
  const {
    trafficData, status: trafficStatus, lastUpdated: trafficLastUpdated,
    snapshots, selectedSnapshot, fetchSnapshot, triggerSnapshot,
  } = useTrafficData(city)
  const {
    simData, status: simStatus, presets, simulations, activeMeta,
    propDepth, setPropDepth, propDecay, setPropDecay,
    activeEvents,
    runPreset, runCustom, loadSavedSimulation, clearSimulation,
    addEvent, removeEvent, updateEvent,
    saveAsPreset, deletePreset,
  } = useSimulation(city)

  // const routingSimulation = activeMeta?.filename ?? null
  // const routingSnapshot = routingSimulation ? null : selectedSnapshot

  const {
    classicRoute, trafficRoute, gnnRoute, classifierRoute,
    degradedInfo, degradedGeojson, isLoading: routeLoading, clearRoutes, fetchRoutes,
  } = useRouteData(city)

  useEffect(() => {
    setStartPoint(null); setEndPoint(null); clearRoutes(); clearSimulation()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city])

  const handleRoute = async () => {
    if (!startPoint || !endPoint) return
    // Auto-save unsaved simulation first
    if (activeEvents.length > 0 && !activeMeta?.filename) {
      await runCustom(activeEvents, activeMeta?.scenario_name || 'custom', '')
    }
    const sim = activeMeta?.filename ?? null
    const snap = sim ? null : selectedSnapshot
    fetchRoutes(startPoint, endPoint, routeMode, snap, sim, dataPct, degradeMode, corridorWidth, zoneRadius)
  }

  const handleMapClick = (latlng: { lat: number; lng: number }) => {
    if (mapMode === 'simulation') {
      const newEvent: CrisisEvent = {
        type: placingEventType as CrisisEvent['type'],
        center: [latlng.lat, latlng.lng],
        radius_m: placingEventType === 'congestion' ? 800 : placingEventType === 'damage' ? 80 : 150,
        speed_reduction_pct: placingEventType === 'damage' ? 0 : defaultSpeed,
        description: '',
      }
      addEvent(newEvent)
      return
    }
    if (!startPoint) {
      setStartPoint([latlng.lat, latlng.lng])
    } else if (!endPoint) {
      setEndPoint([latlng.lat, latlng.lng])
    } else {
      setStartPoint([latlng.lat, latlng.lng])
      setEndPoint(null)
      clearRoutes()
    }
  }

  return (
    <div className="map-container">
      <div className="map-mode-toggle">
        <button className={`mode-btn ${mapMode === 'routing' ? 'active' : ''}`}
          onClick={() => setMapMode('routing')}>Routing</button>
        <button className={`mode-btn ${mapMode === 'simulation' ? 'active' : ''}`}
          onClick={() => setMapMode('simulation')}>Simulation</button>
      </div>

      <div className={`info-panel ${mapMode === 'simulation' ? 'placing-event' : ''}`}>
        <InfoPanel
          city={city} onCityChange={c => { if (c !== city) setCity(c) }}
          startPoint={startPoint} endPoint={endPoint}
          classicRoute={classicRoute} trafficRoute={trafficRoute} gnnRoute={gnnRoute}
          classifierRoute={classifierRoute}
          onClear={() => { setStartPoint(null); setEndPoint(null); clearRoutes() }}
          showTraffic={showTraffic} onToggleTraffic={() => setShowTraffic(v => !v)}
          showIncidents={showIncidents} onToggleIncidents={() => setShowIncidents(v => !v)}
          trafficData={trafficData} trafficStatus={trafficStatus}
          trafficLastUpdated={trafficLastUpdated}
          snapshots={snapshots} selectedSnapshot={selectedSnapshot}
          onSelectSnapshot={fetchSnapshot} onTriggerSnapshot={triggerSnapshot}
          routeMode={routeMode} onRouteModeChange={setRouteMode}
          dataPct={dataPct} onDataPctChange={setDataPct}
          onRoute={handleRoute} routeLoading={routeLoading}
          degradedInfo={degradedInfo}
          showDegraded={showDegraded} onToggleDegraded={() => setShowDegraded(v => !v)}
          degradeMode={degradeMode} onDegradeModeChange={setDegradeMode}
          corridorWidth={corridorWidth} onCorridorWidthChange={setCorridorWidth}
          zoneRadius={zoneRadius} onZoneRadiusChange={setZoneRadius}
        />
        <SimulationPanel
          presets={presets} simulations={simulations} status={simStatus}
          activeMeta={activeMeta} showSimulation={showSimulation}
          onToggleSimulation={() => setShowSimulation(v => !v)}
          onRunPreset={runPreset} onRunCustom={runCustom}
          onLoadSaved={loadSavedSimulation} onClear={clearSimulation}
          propDepth={propDepth} propDecay={propDecay}
          onPropDepthChange={setPropDepth} onPropDecayChange={setPropDecay}
          activeEvents={activeEvents} onAddEvent={addEvent}
          onRemoveEvent={removeEvent} onUpdateEvent={updateEvent}
          mapMode={mapMode} placingEventType={placingEventType}
          onPlacingEventTypeChange={setPlacingEventType}
          defaultSpeed={defaultSpeed} onDefaultSpeedChange={setDefaultSpeed}
          onSaveAsPreset={saveAsPreset} onDeletePreset={deletePreset}
        />
      </div>

      <RouteMap
        key={city} city={city} graphData={graphData} trafficData={trafficData}
        showTraffic={showTraffic} showIncidents={showIncidents}
        simData={simData} showSimulation={showSimulation}
        classicRoute={classicRoute} trafficRoute={trafficRoute} gnnRoute={gnnRoute}
        classifierRoute={classifierRoute}
        degradedGeojson={showDegraded ? degradedGeojson : null}
        startPoint={startPoint} endPoint={endPoint}
        onMapClick={handleMapClick}
        placingEvent={mapMode === 'simulation' ? placingEventType : null}
      />
    </div>
  )
}

export default App