import { useState, useEffect } from 'react'

export default function Dashboard({ pin }) {
  const [dashboard, setDashboard] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboard()
    const interval = setInterval(fetchDashboard, 10000) // Refresh every 10s
    return () => clearInterval(interval)
  }, [])

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

  if (loading) {
    return <div className="text-center py-12 text-gray-500">Loading dashboard...</div>
  }

  if (!dashboard) {
    return <div className="text-center py-12 text-red-500">Could not load dashboard.</div>
  }

  const { device, stats, recent_sessions } = dashboard

  const stateColors = {
    ready: 'bg-green-100 text-green-800',
    listening: 'bg-blue-100 text-blue-800',
    processing: 'bg-yellow-100 text-yellow-800',
    speaking: 'bg-purple-100 text-purple-800',
    setup: 'bg-gray-100 text-gray-800',
    loading: 'bg-orange-100 text-orange-800',
    error: 'bg-red-100 text-red-800',
  }

  return (
    <div className="space-y-6">
      {/* Device Status */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-4">Device Status</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="text-xs text-gray-500 uppercase">State</label>
            <div className={`inline-block mt-1 px-3 py-1 rounded-full text-sm font-medium ${stateColors[device.state] || 'bg-gray-100'}`}>
              {device.state}
            </div>
          </div>
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
