import { useState, useEffect } from 'react'

export default function Settings({ pin }) {
  const [settings, setSettings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [restarting, setRestarting] = useState(false)

  useEffect(() => {
    fetchSettings()
  }, [])

  async function fetchSettings() {
    try {
      const res = await fetch('/api/settings/', {
        headers: { 'X-Parent-PIN': pin },
      })
      const data = await res.json()
      setSettings(data)
    } catch (err) {
      console.error('Settings fetch error:', err)
    } finally {
      setLoading(false)
    }
  }

  async function updateSetting(endpoint, body) {
    setSaving(true)
    try {
      const res = await fetch(`/api/settings/${endpoint}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'X-Parent-PIN': pin,
        },
        body: JSON.stringify(body),
      })
      const data = await res.json()

      // If the server is restarting (e.g., language change), show overlay and wait
      if (data.restarting) {
        setRestarting(true)
        // Wait for the service to come back up (poll every 2s)
        const waitForRestart = async () => {
          await new Promise((r) => setTimeout(r, 3000))
          for (let i = 0; i < 30; i++) {
            try {
              const check = await fetch('/api/settings/', {
                headers: { 'X-Parent-PIN': pin },
              })
              if (check.ok) {
                setRestarting(false)
                await fetchSettings()
                return
              }
            } catch {}
            await new Promise((r) => setTimeout(r, 2000))
          }
          setRestarting(false)
          await fetchSettings()
        }
        waitForRestart()
        return
      }

      await fetchSettings()
    } catch (err) {
      console.error('Update error:', err)
    } finally {
      setSaving(false)
    }
  }

  async function toggleHardware(type, enabled) {
    try {
      await fetch(`/api/control/${type}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'X-Parent-PIN': pin,
        },
        body: JSON.stringify({ enabled }),
      })
      await fetchSettings()
    } catch (err) {
      console.error('Toggle error:', err)
    }
  }

  if (loading) {
    return <div className="text-center py-12 text-gray-500">Loading settings...</div>
  }

  if (!settings) {
    return <div className="text-center py-12 text-red-500">Could not load settings.</div>
  }

  if (restarting) {
    return (
      <div className="text-center py-16">
        <div className="animate-spin w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full mx-auto mb-4" />
        <p className="text-gray-600 font-medium">Restarting with new voice model...</p>
        <p className="text-xs text-gray-400 mt-2">This takes about 10 seconds.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Mode Toggle */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-4">AI Mode</h2>
        <div className="flex gap-3">
          <button
            onClick={() => updateSetting('mode', { mode: 'offline' })}
            className={`flex-1 py-3 rounded-xl font-medium transition-colors ${
              settings.mode === 'offline'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Offline
          </button>
          <button
            onClick={() => updateSetting('mode', { mode: 'online' })}
            className={`flex-1 py-3 rounded-xl font-medium transition-colors ${
              settings.mode === 'online'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Online
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-3">
          {settings.mode === 'offline'
            ? 'All processing happens on the device. No data leaves the Pi.'
            : 'Questions are sent to a cloud AI for better responses. Text only — no audio.'}
        </p>
      </div>

      {/* Provider Selection (only when online) */}
      {settings.mode === 'online' && (
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold mb-4">AI Provider</h2>
          <div className="grid grid-cols-3 gap-3">
            {['openai', 'gemini', 'claude'].map((provider) => (
              <button
                key={provider}
                onClick={() => updateSetting('provider', { provider })}
                className={`py-3 rounded-xl font-medium text-sm transition-colors ${
                  settings.provider === provider
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {provider === 'openai' ? 'ChatGPT' : provider === 'gemini' ? 'Gemini' : 'Claude'}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Cloud Speech Recognition (only when online) */}
      {settings.mode === 'online' && (
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold mb-4">Cloud Speech Recognition</h2>
          <Toggle
            label={settings.hardware.cloud_stt ? 'Enabled' : 'Disabled'}
            description={
              settings.hardware.cloud_stt
                ? 'Speech sent to OpenAI for faster recognition (~0.5s). Audio is not stored.'
                : 'Speech processed on device. Private but slower (~3s).'
            }
            enabled={settings.hardware.cloud_stt}
            onChange={(v) => updateSetting('cloud-stt', { enabled: v })}
          />
          <p className="text-xs text-amber-600 mt-3">
            When enabled, your child's voice audio is sent to OpenAI for transcription. Audio is processed and immediately discarded — it is not stored.
          </p>
        </div>
      )}

      {/* Hardware Controls */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-4">Hardware</h2>
        <div className="space-y-4">
          <Toggle
            label="Microphone"
            description="Voice input for learning interactions"
            enabled={settings.hardware.mic_enabled}
            onChange={(v) => toggleHardware('mic', v)}
          />
          <Toggle
            label="Camera"
            description="Visual features: object recognition, reading"
            enabled={settings.hardware.camera_enabled}
            onChange={(v) => toggleHardware('camera', v)}
          />
        </div>
      </div>

      {/* Child Settings */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-4">Child Settings</h2>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="text-sm font-medium text-gray-600">Language</label>
            <select
              value={settings.child.language || 'en'}
              onChange={(e) => updateSetting('child', { language: e.target.value })}
              className="w-full mt-1 border rounded-lg px-3 py-2"
            >
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="te">Telugu</option>
            </select>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-600">Age Min</label>
            <select
              value={settings.child.age_min}
              onChange={(e) => updateSetting('child', { age_min: Number(e.target.value) })}
              className="w-full mt-1 border rounded-lg px-3 py-2"
            >
              {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-600">Age Max</label>
            <select
              value={settings.child.age_max}
              onChange={(e) => updateSetting('child', { age_max: Number(e.target.value) })}
              className="w-full mt-1 border rounded-lg px-3 py-2"
            >
              {[3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </div>
        </div>
        <p className="text-xs text-gray-400 mt-3">
          Changing language will automatically restart the bot to load the appropriate voice model.
        </p>
      </div>
    </div>
  )
}

function Toggle({ label, description, enabled, onChange }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <div className="font-medium text-sm">{label}</div>
        <div className="text-xs text-gray-400">{description}</div>
      </div>
      <button
        onClick={() => onChange(!enabled)}
        className={`relative w-12 h-7 rounded-full transition-colors ${
          enabled ? 'bg-blue-600' : 'bg-gray-300'
        }`}
      >
        <div
          className={`absolute top-0.5 w-6 h-6 bg-white rounded-full shadow transition-transform ${
            enabled ? 'translate-x-5' : 'translate-x-0.5'
          }`}
        />
      </button>
    </div>
  )
}
