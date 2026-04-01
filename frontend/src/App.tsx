import { useState, useEffect } from 'react'
import './App.css'
import InfoPanel from './components/InfoPanel'
import RouteMap from './components/RouteMap'
import SimulationPanel from './components/SimulationPanel'
import { useGraphData } from './hooks/useGraphData'
import { useRouteData, type RouteMode } from './hooks/useRouteData'
import { useTrafficData } from './hooks/useTrafficData'
import { useSimulation } from './hooks/useSimulation'

function App() {
  const [city, setCity]                   = useState('Vilnius')
  const [startPoint, setStartPoint]       = useState<[number, number] | null>(null)
  const [endPoint, setEndPoint]           = useState<[number, number] | null>(null)
  const [showTraffic, setShowTraffic]     = useState(false)
  const [showIncidents, setShowIncidents] = useState(true)
  const [showSimulation, setShowSimulation] = useState(true)
  const [routeMode, setRouteMode]         = useState<RouteMode>('both')

  const graphData = useGraphData(city)

  const {
    trafficData, status: trafficStatus, lastUpdated: trafficLastUpdated,
    snapshots, selectedSnapshot, fetchSnapshot, triggerSnapshot,
  } = useTrafficData(city)

  const {
    simData, status: simStatus, presets, simulations, activeMeta,
    propDepth, setPropDepth, propDecay, setPropDecay,
    runPreset, loadSavedSimulation, clearSimulation,
  } = useSimulation(city)

  const routingSimulation = activeMeta?.filename ?? null
  const routingSnapshot   = routingSimulation ? null : selectedSnapshot

  const { classicRoute, trafficRoute, clearRoutes } = useRouteData(
    startPoint, endPoint, city, routeMode, routingSnapshot, routingSimulation
  )

  useEffect(() => {
    setStartPoint(null)
    setEndPoint(null)
    clearRoutes()
    clearSimulation()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city])

  const handleMapClick = (latlng: { lat: number; lng: number }) => {
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
      {/* Single overlay panel — internally scrollable */}
      <div className="info-panel">
        <InfoPanel
          city={city}
          onCityChange={c => { if (c !== city) setCity(c) }}
          startPoint={startPoint}
          endPoint={endPoint}
          classicRoute={classicRoute}
          trafficRoute={trafficRoute}
          onClear={() => { setStartPoint(null); setEndPoint(null); clearRoutes() }}
          showTraffic={showTraffic}
          onToggleTraffic={() => setShowTraffic(v => !v)}
          showIncidents={showIncidents}
          onToggleIncidents={() => setShowIncidents(v => !v)}
          trafficData={trafficData}
          trafficStatus={trafficStatus}
          trafficLastUpdated={trafficLastUpdated}
          snapshots={snapshots}
          selectedSnapshot={selectedSnapshot}
          onSelectSnapshot={fetchSnapshot}
          onTriggerSnapshot={triggerSnapshot}
          routeMode={routeMode}
          onRouteModeChange={setRouteMode}
        />
        <SimulationPanel
          presets={presets}
          simulations={simulations}
          status={simStatus}
          activeMeta={activeMeta}
          showSimulation={showSimulation}
          onToggleSimulation={() => setShowSimulation(v => !v)}
          onRunPreset={runPreset}
          onLoadSaved={loadSavedSimulation}
          onClear={clearSimulation}
          propDepth={propDepth}
          propDecay={propDecay}
          onPropDepthChange={setPropDepth}
          onPropDecayChange={setPropDecay}
        />
      </div>

      <RouteMap
        key={city}
        city={city}
        graphData={graphData}
        trafficData={trafficData}
        showTraffic={showTraffic}
        showIncidents={showIncidents}
        simData={simData}
        showSimulation={showSimulation}
        classicRoute={classicRoute}
        trafficRoute={trafficRoute}
        startPoint={startPoint}
        endPoint={endPoint}
        onMapClick={handleMapClick}
      />
    </div>
  )
}

export default App