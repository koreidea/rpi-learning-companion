import { useState, useEffect, useRef } from 'react'

export default function Dashboard({ pin }) {
  const [dashboard, setDashboard] = useState(null)
  const [live, setLive] = useState(null)
  const [loading, setLoading] = useState(true)
  const liveRef = useRef(null)

  useEffect(() => {
    fetchDashboard()
    const dashInterval = setInterval(fetchDashboard, 30000) // Stats every 30s
    return () => clearInterval(dashInterval)
  }, [])

  // Fast poll for live status
  useEffect(() => {
    fetchLive()
    const liveInterval = setInterval(fetchLive, 1000) // Every 1s
    return () => clearInterval(liveInterval)
  }, [])

  // Auto-scroll conversation to bottom
  useEffect(() => {
    if (liveRef.current) {
      liveRef.current.scrollTop = liveRef.current.scrollHeight
    }
  }, [live])

  async function fetchDashboard() {
    try {
      const res = await fetch('/api/dashboard/', {
        headers: { 'X-Parent-PIN': pin },
      })
      const data = await res.json()
      setDashboard(data)
    } catch (err) {
      console.error('Dashboard fetch error:', err)
    } finally {
      setLoading(false)
    }
  }

  async function fetchLive() {
    try {
      const res = await fetch('/api/dashboard/live', {
        headers: { 'X-Parent-PIN': pin },
      })
      const data = await res.json()
      setLive(data)
    } catch (err) {
      // Silently ignore live poll errors
    }
  }

  if (loading) {
    return <div className="text-center py-12 text-gray-500">Loading dashboard...</div>
  }

  if (!dashboard) {
    return <div className="text-center py-12 text-red-500">Could not load dashboard.</div>
  }

  const { device, stats, recent_sessions } = dashboard

  const stateConfig = {
    ready:      { color: 'bg-green-100 text-green-800', dot: 'bg-green-500', label: 'Waiting for wake word' },
    listening:  { color: 'bg-blue-100 text-blue-800', dot: 'bg-blue-500', label: 'Listening...' },
    processing: { color: 'bg-yellow-100 text-yellow-800', dot: 'bg-yellow-500', label: 'Thinking...' },
    speaking:   { color: 'bg-purple-100 text-purple-800', dot: 'bg-purple-500', label: 'Speaking...' },
    setup:      { color: 'bg-gray-100 text-gray-800', dot: 'bg-gray-400', label: 'Setup required' },
    loading:    { color: 'bg-orange-100 text-orange-800', dot: 'bg-orange-500', label: 'Loading models...' },
    error:      { color: 'bg-red-100 text-red-800', dot: 'bg-red-500', label: 'Error' },
  }

  const currentState = live?.state || device.state
  const sc = stateConfig[currentState] || stateConfig.ready
  const isActive = ['listening', 'processing', 'speaking'].includes(currentState)
  const canStop = ['processing', 'speaking'].includes(currentState)

  async function stopResponse() {
    try {
      await fetch('/api/control/stop-response', {
        method: 'POST',
        headers: { 'X-Parent-PIN': pin },
      })
    } catch (err) {
      console.error('Stop response error:', err)
    }
  }

  return (
    <div className="space-y-6">
      {/* Live Status Panel */}
      <div className={`bg-white rounded-xl shadow-sm p-6 border-2 ${isActive ? 'border-blue-300' : 'border-transparent'}`}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Live Status</h2>
          <div className="flex items-center gap-2">
            {canStop && (
              <button
                onClick={stopResponse}
                className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded-full text-sm font-medium transition-colors flex items-center gap-1"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <rect x="5" y="5" width="10" height="10" rx="1" />
                </svg>
                Stop
              </button>
            )}
            <div className={`w-3 h-3 rounded-full ${sc.dot} ${isActive ? 'animate-pulse' : ''}`} />
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${sc.color}`}>
              {sc.label}
            </span>
          </div>
        </div>

        {live?.follow_up && (
          <div className="mb-3 px-3 py-1.5 bg-green-50 text-green-700 rounded-lg text-sm">
            Follow-up mode — no wake word needed
          </div>
        )}

        {/* Conversation area */}
        <div ref={liveRef} className="bg-gray-50 rounded-xl p-4 min-h-[120px] max-h-[300px] overflow-y-auto space-y-3">
          {!live?.transcript && !live?.response && (
            <p className="text-gray-400 text-sm text-center py-6">
              {currentState === 'ready'
                ? 'Say "Hey Jarvis" to start a conversation'
                : currentState === 'listening'
                ? '🎤 Listening...'
                : currentState === 'loading'
                ? 'Loading models, please wait...'
                : 'Waiting...'}
            </p>
          )}

          {live?.transcript && (
            <div className="flex gap-2">
              <span className="text-lg">👦</span>
              <div className="bg-blue-100 text-blue-900 rounded-xl rounded-tl-sm px-4 py-2 text-sm max-w-[85%]">
                {live.transcript}
              </div>
            </div>
          )}

          {/* Camera image when vision request */}
          {live?.image && (
            <div className="flex gap-2">
              <span className="text-lg">📷</span>
              <div className="rounded-xl overflow-hidden border border-gray-200 max-w-[85%]">
                <img
                  src={`data:image/jpeg;base64,${live.image}`}
                  alt="Camera capture"
                  className="w-full rounded-xl"
                />
                {live.detections && live.detections.length > 0 && (
                  <div className="px-3 py-1.5 bg-gray-50 text-xs text-gray-600">
                    Detected: {live.detections.map(d => d.label).join(', ')}
                  </div>
                )}
              </div>
            </div>
          )}

          {live?.response && (
            <div className="flex gap-2 justify-end">
              <div className="bg-white border border-gray-200 text-gray-800 rounded-xl rounded-tr-sm px-4 py-2 text-sm max-w-[85%]">
                {live.response}
              </div>
              <span className="text-lg">🤖</span>
            </div>
          )}

          {currentState === 'processing' && !live?.response && live?.transcript && (
            <div className="flex gap-2 justify-end">
              <div className="bg-white border border-gray-200 text-gray-400 rounded-xl rounded-tr-sm px-4 py-2 text-sm">
                <span className="animate-pulse">Thinking...</span>
              </div>
              <span className="text-lg">🤖</span>
            </div>
          )}
        </div>

        {live?.last_error && (
          <div className="mt-3 px-3 py-2 bg-red-50 text-red-600 rounded-lg text-xs">
            Error: {live.last_error}
          </div>
        )}
      </div>

      {/* Device Status */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-4">Device Info</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="text-xs text-gray-500 uppercase">Mode</label>
            <div className="mt-1 text-sm font-medium">
              {device.mode === 'online' ? 'Online' : 'Offline'}
              {device.mode === 'online' && ` (${device.provider})`}
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-500 uppercase">Microphone</label>
            <div className={`mt-1 text-sm font-medium ${device.mic_enabled ? 'text-green-600' : 'text-red-600'}`}>
              {device.mic_enabled ? 'On' : 'Off'}
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-500 uppercase">Camera</label>
            <div className={`mt-1 text-sm font-medium ${device.camera_enabled ? 'text-green-600' : 'text-red-600'}`}>
              {device.camera_enabled ? 'On' : 'Off'}
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-500 uppercase">Models</label>
            <div className={`mt-1 text-sm font-medium ${device.model_loaded ? 'text-green-600' : 'text-orange-600'}`}>
              {device.model_loaded ? 'Loaded' : 'Loading...'}
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-4">Usage Stats</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Total Sessions" value={stats.total_sessions} />
          <StatCard label="Today" value={stats.sessions_today} />
          <StatCard label="Total Time" value={`${stats.total_duration_minutes}m`} />
          <StatCard label="Avg Session" value={`${stats.avg_session_seconds}s`} />
        </div>
      </div>

      {/* Topics */}
      {stats.topics && Object.keys(stats.topics).length > 0 && (
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold mb-4">Topics Explored</h2>
          <div className="flex flex-wrap gap-2">
            {Object.entries(stats.topics).map(([topic, count]) => (
              <span
                key={topic}
                className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm"
              >
                {topic} ({count})
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Recent Sessions */}
      {recent_sessions && recent_sessions.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold mb-4">Recent Activity</h2>
          <div className="space-y-2">
            {recent_sessions.slice(0, 10).map((session, i) => (
              <div key={i} className="flex justify-between items-center py-2 border-b border-gray-50 last:border-0">
                <div>
                  <span className="text-sm font-medium">{session.topic_category}</span>
                  <span className="text-xs text-gray-400 ml-2">{session.mode}</span>
                </div>
                <div className="text-right">
                  <span className="text-sm text-gray-600">{session.duration_seconds}s</span>
                  <div className="text-xs text-gray-400">
                    {new Date(session.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value }) {
  return (
    <div className="text-center">
      <div className="text-2xl font-bold text-gray-800">{value}</div>
      <div className="text-xs text-gray-500 mt-1">{label}</div>
    </div>
  )
}
