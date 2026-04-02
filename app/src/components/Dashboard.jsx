import { useState, useEffect, useRef } from 'react'

export default function Dashboard({ pin }) {
  const [dashboard, setDashboard] = useState(null)
  const [live, setLive] = useState(null)
  const [loading, setLoading] = useState(true)
  const [isListening, setIsListening] = useState(false)
  const [textInput, setTextInput] = useState('')
  const [sendStatus, setSendStatus] = useState(null)
  const liveRef = useRef(null)
  const recognitionRef = useRef(null)

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
    dancing:    { color: 'bg-pink-100 text-pink-800', dot: 'bg-pink-500', label: 'Dancing!' },
    setup:      { color: 'bg-gray-100 text-gray-800', dot: 'bg-gray-400', label: 'Setup required' },
    loading:    { color: 'bg-orange-100 text-orange-800', dot: 'bg-orange-500', label: 'Loading models...' },
    error:      { color: 'bg-red-100 text-red-800', dot: 'bg-red-500', label: 'Error' },
  }

  const currentState = live?.state || device.state
  const sc = stateConfig[currentState] || stateConfig.ready
  const isActive = ['listening', 'processing', 'speaking'].includes(currentState)
  const canStop = ['processing', 'speaking'].includes(currentState)

  const botEnabled = live?.bot_enabled ?? true
  const volume = live?.volume ?? 80

  async function setVolume(level) {
    try {
      await fetch('/api/control/volume', {
        method: 'PUT',
        headers: { 'X-Parent-PIN': pin, 'Content-Type': 'application/json' },
        body: JSON.stringify({ level }),
      })
    } catch (err) {
      console.error('Volume set error:', err)
    }
  }

  async function toggleBot() {
    try {
      await fetch('/api/control/mic', {
        method: 'PUT',
        headers: { 'X-Parent-PIN': pin, 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !botEnabled }),
      })
    } catch (err) {
      console.error('Toggle bot error:', err)
    }
  }

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

  async function sendTextToBot(text) {
    if (!text.trim()) return
    setSendStatus('sending')
    try {
      const res = await fetch('/api/control/send-text', {
        method: 'POST',
        headers: { 'X-Parent-PIN': pin, 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })
      const data = await res.json()
      if (data.status === 'sent') {
        setSendStatus('sent')
        setTextInput('')
        setTimeout(() => setSendStatus(null), 2000)
      } else {
        setSendStatus('error')
      }
    } catch (err) {
      console.error('Send text error:', err)
      setSendStatus('error')
    }
  }

  async function triggerWake() {
    try {
      await fetch('/api/control/wake', {
        method: 'POST',
        headers: { 'X-Parent-PIN': pin },
      })
    } catch (err) {
      console.error('Wake trigger error:', err)
    }
  }

  function toggleSpeechRecognition() {
    if (isListening) {
      // Stop listening
      if (recognitionRef.current) {
        recognitionRef.current.abort()
        recognitionRef.current = null
      }
      setIsListening(false)
      return
    }

    // Start speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Use the text input instead.')
      return
    }

    const recognition = new SpeechRecognition()
    recognition.lang = 'en-IN'
    recognition.interimResults = false
    recognition.maxAlternatives = 1
    recognition.continuous = false

    recognition.onresult = (event) => {
      const text = event.results[0][0].transcript
      recognition.stop()
      recognitionRef.current = null
      setIsListening(false)
      sendTextToBot(text)
    }

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error)
      recognition.abort()
      recognitionRef.current = null
      setIsListening(false)
    }

    recognition.onend = () => {
      recognitionRef.current = null
      setIsListening(false)
    }

    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
  }

  return (
    <div className="space-y-6">
      {/* Live Status Panel */}
      <div className={`bg-white rounded-xl shadow-sm p-6 border-2 ${!botEnabled ? 'border-gray-300' : isActive ? 'border-blue-300' : 'border-transparent'}`}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Live Status</h2>
          <div className="flex items-center gap-2">
            {canStop && botEnabled && (
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
            {botEnabled ? (
              <>
                <div className={`w-3 h-3 rounded-full ${sc.dot} ${isActive ? 'animate-pulse' : ''}`} />
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${sc.color}`}>
                  {sc.label}
                </span>
              </>
            ) : (
              <span className="px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-500">
                Companion Off
              </span>
            )}
          </div>
        </div>

        {live?.follow_up && (
          <div className="mb-3 px-3 py-1.5 bg-green-50 text-green-700 rounded-lg text-sm">
            Follow-up mode — no wake word needed
          </div>
        )}

        {/* Conversation area */}
        <div ref={liveRef} className="bg-gray-50 rounded-xl p-4 min-h-[120px] max-h-[400px] overflow-y-auto space-y-3">
          {(!live?.messages || live.messages.length === 0) && !live?.transcript && !live?.response && (
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

          {/* Past conversation messages (completed exchanges) */}
          {live?.messages?.map((msg, i) => (
            msg.role === 'user' ? (
              <div key={`h-${i}`}>
                <div className="flex gap-2">
                  <span className="text-lg">👦</span>
                  <div className="bg-blue-100 text-blue-900 rounded-xl rounded-tl-sm px-4 py-2 text-sm max-w-[85%]">
                    {msg.content}
                  </div>
                </div>
                {msg.image && (
                  <div className="flex gap-2 mt-1">
                    <span className="text-lg">📷</span>
                    <div className="rounded-xl overflow-hidden border border-gray-200 max-w-[60%]">
                      <img
                        src={`data:image/jpeg;base64,${msg.image}`}
                        alt="Camera capture"
                        className="w-full rounded-xl"
                      />
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div key={`h-${i}`} className="flex gap-2 justify-end">
                <div className="bg-white border border-gray-200 text-gray-800 rounded-xl rounded-tr-sm px-4 py-2 text-sm max-w-[85%]">
                  {msg.content}
                </div>
                <span className="text-lg">🤖</span>
              </div>
            )
          ))}

          {/* Current live interaction (in-progress)
              Only show if not already the last message in history (avoids duplication) */}
          {(() => {
            const msgs = live?.messages || []
            const lastUserMsg = msgs.filter(m => m.role === 'user').pop()
            const isDuplicate = lastUserMsg && live?.transcript && lastUserMsg.content === live.transcript
            if (isDuplicate) return null

            return (
              <>
                {live?.transcript && (
                  <div className="flex gap-2">
                    <span className="text-lg">👦</span>
                    <div className="bg-blue-100 text-blue-900 rounded-xl rounded-tl-sm px-4 py-2 text-sm max-w-[85%]">
                      {live.transcript}
                    </div>
                  </div>
                )}

                {live?.image && (
                  <div className="flex gap-2">
                    <span className="text-lg">📷</span>
                    <div className="rounded-xl overflow-hidden border border-gray-200 max-w-[60%]">
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
              </>
            )
          })()}
        </div>

        {live?.last_error && (
          <div className="mt-3 px-3 py-2 bg-red-50 text-red-600 rounded-lg text-xs">
            Error: {live.last_error}
          </div>
        )}
      </div>

      {/* Companion ON/OFF Toggle */}
      <div className={`rounded-xl shadow-sm p-4 flex items-center justify-between ${botEnabled ? 'bg-green-50 border border-green-200' : 'bg-gray-50 border border-gray-200'}`}>
        <div className="flex items-center gap-3">
          <span className="text-2xl">{botEnabled ? '🟢' : '⚫'}</span>
          <div>
            <div className="font-semibold text-sm">{botEnabled ? 'Companion is ON' : 'Companion is OFF'}</div>
            <div className="text-xs text-gray-500">{botEnabled ? 'Listening and responding to your child' : 'Not listening — tap to turn on'}</div>
          </div>
        </div>
        <button
          onClick={toggleBot}
          className={`relative inline-flex h-9 w-[72px] items-center rounded-full transition-colors focus:outline-none shadow-inner ${
            botEnabled ? 'bg-green-500' : 'bg-gray-300'
          }`}
        >
          <span
            className={`inline-block h-7 w-7 transform rounded-full bg-white shadow-md transition-transform ${
              botEnabled ? 'translate-x-10' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* Remote Mic — Speak from phone */}
      <div className="bg-white rounded-xl shadow-sm p-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-lg">📱</span>
          <span className="font-semibold text-sm">Remote Voice</span>
          <span className="text-xs text-gray-400">Speak from your phone to the bot</span>
        </div>
        <div className="flex gap-2">
          {/* Big mic button */}
          <button
            onClick={toggleSpeechRecognition}
            className={`flex-shrink-0 w-14 h-14 rounded-full flex items-center justify-center text-white text-2xl shadow-lg transition-all ${
              isListening
                ? 'bg-red-500 animate-pulse scale-110'
                : 'bg-blue-500 hover:bg-blue-600 active:scale-95'
            }`}
          >
            {isListening ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" viewBox="0 0 20 20" fill="currentColor">
                <rect x="5" y="5" width="10" height="10" rx="1" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
              </svg>
            )}
          </button>
          {/* Text input fallback */}
          <div className="flex-1 flex gap-2">
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') sendTextToBot(textInput) }}
              placeholder="Or type a message..."
              className="flex-1 px-3 py-2 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
            />
            <button
              onClick={() => sendTextToBot(textInput)}
              disabled={!textInput.trim() || sendStatus === 'sending'}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white rounded-xl text-sm font-medium transition-colors"
            >
              Send
            </button>
          </div>
        </div>
        {isListening && (
          <div className="mt-2 text-center text-sm text-red-500 animate-pulse">
            Listening... speak now
          </div>
        )}
        {sendStatus === 'sent' && (
          <div className="mt-2 text-center text-sm text-green-600">
            Sent to bot!
          </div>
        )}
        {sendStatus === 'error' && (
          <div className="mt-2 text-center text-sm text-red-600">
            Failed to send. Try again.
          </div>
        )}
        {/* Wake word button */}
        <div className="mt-3 flex justify-center">
          <button
            onClick={triggerWake}
            className="px-4 py-1.5 bg-green-100 hover:bg-green-200 text-green-700 rounded-full text-xs font-medium transition-colors"
          >
            Wake Bot
          </button>
        </div>
      </div>

      {/* Volume Control */}
      <div className="bg-white rounded-xl shadow-sm p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-lg">{volume === 0 ? '🔇' : volume < 40 ? '🔈' : volume < 70 ? '🔉' : '🔊'}</span>
            <span className="font-semibold text-sm">Volume</span>
          </div>
          <span className="text-sm font-medium text-gray-600">{volume}%</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setVolume(Math.max(0, volume - 10))}
            className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 text-gray-600 text-lg font-bold transition-colors"
          >
            −
          </button>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={volume}
            onChange={(e) => setVolume(Number(e.target.value))}
            className="flex-1 h-2 rounded-full appearance-none bg-gray-200 accent-blue-500 cursor-pointer"
          />
          <button
            onClick={() => setVolume(Math.min(100, volume + 10))}
            className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 text-gray-600 text-lg font-bold transition-colors"
          >
            +
          </button>
        </div>
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
