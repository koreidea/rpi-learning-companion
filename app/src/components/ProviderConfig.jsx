import { useState, useEffect } from 'react'

export default function ProviderConfig({ pin }) {
  const [settings, setSettings] = useState(null)
  const [keys, setKeys] = useState({ openai: '', gemini: '', claude: '' })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

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
    }
  }

  async function saveKeys() {
    setSaving(true)
    setSaved(false)
    try {
      const body = {}
      if (keys.openai) body.openai = keys.openai
      if (keys.gemini) body.gemini = keys.gemini
      if (keys.claude) body.claude = keys.claude

      await fetch('/api/settings/api-keys', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'X-Parent-PIN': pin,
        },
        body: JSON.stringify(body),
      })
      setSaved(true)
      setKeys({ openai: '', gemini: '', claude: '' })
      await fetchSettings()
    } catch (err) {
      console.error('Save keys error:', err)
    } finally {
      setSaving(false)
    }
  }

  const providers = [
    {
      id: 'openai',
      name: 'OpenAI (ChatGPT)',
      description: 'GPT-4o-mini — fast, affordable, great for kids',
      placeholder: 'sk-...',
    },
    {
      id: 'gemini',
      name: 'Google Gemini',
      description: 'Gemini 1.5 Flash — fast and capable',
      placeholder: 'AI...',
    },
    {
      id: 'claude',
      name: 'Anthropic Claude',
      description: 'Claude Haiku — safe and thoughtful',
      placeholder: 'sk-ant-...',
    },
  ]

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-2">AI Provider API Keys</h2>
        <p className="text-sm text-gray-500 mb-6">
          Enter API keys for the cloud AI providers you want to use in online mode.
          Keys are stored encrypted on the device.
        </p>

        <div className="space-y-5">
          {providers.map((p) => (
            <div key={p.id}>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {p.name}
              </label>
              <p className="text-xs text-gray-400 mb-2">{p.description}</p>
              <div className="flex gap-2">
                <input
                  type="password"
                  placeholder={
                    settings?.api_keys?.[p.id] && settings.api_keys[p.id] !== ''
                      ? `Current: ${settings.api_keys[p.id]}`
                      : p.placeholder
                  }
                  value={keys[p.id]}
                  onChange={(e) => setKeys({ ...keys, [p.id]: e.target.value })}
                  className="flex-1 border rounded-lg px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 flex items-center gap-4">
          <button
            onClick={saveKeys}
            disabled={saving || (!keys.openai && !keys.gemini && !keys.claude)}
            className="bg-blue-600 text-white rounded-xl px-6 py-2.5 font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            {saving ? 'Saving...' : 'Save Keys'}
          </button>
          {saved && <span className="text-green-600 text-sm">Keys saved successfully.</span>}
        </div>
      </div>

      <div className="bg-amber-50 rounded-xl p-4 text-sm text-amber-800">
        <strong>Privacy note:</strong> API keys are stored encrypted on the device.
        When online mode is active, only the text of the child's question is sent to the provider — never audio recordings.
      </div>
    </div>
  )
}
