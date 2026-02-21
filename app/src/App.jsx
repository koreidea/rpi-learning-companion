import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import ConsentGate from './components/ConsentGate'
import Dashboard from './components/Dashboard'
import Settings from './components/Settings'
import DataManagement from './components/DataManagement'
import ProviderConfig from './components/ProviderConfig'
import Layout from './components/Layout'

const API_BASE = '/api'

function App() {
  const [consentStatus, setConsentStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [pin, setPin] = useState('')

  useEffect(() => {
    checkConsent()
  }, [])

  async function checkConsent() {
    try {
      const res = await fetch(`${API_BASE}/consent/status`)
      const data = await res.json()
      setConsentStatus(data)
    } catch (err) {
      console.error('Failed to check consent:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-lg text-gray-600">Connecting to Learning Buddy...</div>
      </div>
    )
  }

  // If no consent yet, show the consent/setup flow
  if (!consentStatus?.setup_complete) {
    return <ConsentGate onComplete={() => checkConsent()} />
  }

  // If consent given but no PIN entered this session, prompt for PIN
  if (!pin) {
    return <PinPrompt onSubmit={setPin} />
  }

  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard pin={pin} />} />
          <Route path="/settings" element={<Settings pin={pin} />} />
          <Route path="/providers" element={<ProviderConfig pin={pin} />} />
          <Route path="/data" element={<DataManagement pin={pin} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

function PinPrompt({ onSubmit }) {
  const [value, setValue] = useState('')
  const [error, setError] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    // Verify PIN by trying to access a protected endpoint
    try {
      const res = await fetch('/api/settings/', {
        headers: { 'X-Parent-PIN': value },
      })
      if (res.ok) {
        onSubmit(value)
      } else {
        setError('Invalid PIN. Please try again.')
      }
    } catch {
      setError('Could not connect to the device.')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="bg-white rounded-2xl shadow-lg p-8 w-full max-w-sm">
        <h1 className="text-2xl font-bold text-center mb-2">Learning Buddy</h1>
        <p className="text-gray-500 text-center mb-6">Enter your parent PIN</p>
        <form onSubmit={handleSubmit}>
          <input
            type="password"
            inputMode="numeric"
            maxLength={6}
            value={value}
            onChange={(e) => setValue(e.target.value.replace(/\D/g, ''))}
            placeholder="Enter PIN"
            className="w-full text-center text-2xl tracking-widest border-2 border-gray-200 rounded-xl py-3 px-4 mb-4 focus:border-blue-500 focus:outline-none"
          />
          {error && <p className="text-red-500 text-sm text-center mb-4">{error}</p>}
          <button
            type="submit"
            disabled={value.length < 4}
            className="w-full bg-blue-600 text-white rounded-xl py-3 font-semibold hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Unlock
          </button>
        </form>
      </div>
    </div>
  )
}

export default App
