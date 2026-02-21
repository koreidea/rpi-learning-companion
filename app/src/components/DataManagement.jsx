import { useState } from 'react'

export default function DataManagement({ pin }) {
  const [showErase, setShowErase] = useState(false)
  const [erasePin, setErasePin] = useState('')
  const [erasing, setErasing] = useState(false)
  const [exportData, setExportData] = useState(null)
  const [exporting, setExporting] = useState(false)

  async function handleExport() {
    setExporting(true)
    try {
      const res = await fetch('/api/data/export', {
        headers: { 'X-Parent-PIN': pin },
      })
      const data = await res.json()
      setExportData(data)
    } catch (err) {
      console.error('Export error:', err)
    } finally {
      setExporting(false)
    }
  }

  async function handleErase() {
    if (!erasePin) return
    setErasing(true)
    try {
      const res = await fetch('/api/data/erase-all', {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'X-Parent-PIN': pin,
        },
        body: JSON.stringify({ pin: erasePin }),
      })
      const data = await res.json()
      if (data.status === 'erased') {
        // Reload the page to go back to setup
        window.location.reload()
      }
    } catch (err) {
      console.error('Erase error:', err)
    } finally {
      setErasing(false)
    }
  }

  async function handleRevokeConsent() {
    try {
      await fetch('/api/consent/revoke', { method: 'POST' })
      window.location.reload()
    } catch (err) {
      console.error('Revoke error:', err)
    }
  }

  return (
    <div className="space-y-6">
      {/* Data Info */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-4">Your Data</h2>
        <div className="space-y-3 text-sm text-gray-600">
          <div className="flex justify-between">
            <span>Voice recordings stored</span>
            <span className="font-medium text-green-600">None (never saved)</span>
          </div>
          <div className="flex justify-between">
            <span>Conversation text stored</span>
            <span className="font-medium text-green-600">None (never saved)</span>
          </div>
          <div className="flex justify-between">
            <span>Session metadata stored</span>
            <span className="font-medium">Timestamps & topic categories only</span>
          </div>
          <div className="flex justify-between">
            <span>Data location</span>
            <span className="font-medium">On device only (encrypted)</span>
          </div>
        </div>
      </div>

      {/* Export Data */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-2">Export Data</h2>
        <p className="text-sm text-gray-500 mb-4">
          View all data stored about your child's usage. Only metadata is stored — no conversation content.
        </p>
        <button
          onClick={handleExport}
          disabled={exporting}
          className="bg-gray-100 text-gray-700 rounded-xl px-5 py-2.5 font-medium hover:bg-gray-200 transition-colors"
        >
          {exporting ? 'Loading...' : 'View Stored Data'}
        </button>

        {exportData && (
          <div className="mt-4 bg-gray-50 rounded-lg p-4 text-xs font-mono overflow-auto max-h-60">
            <pre>{JSON.stringify(exportData, null, 2)}</pre>
          </div>
        )}
      </div>

      {/* Revoke Consent */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-2">Revoke Consent</h2>
        <p className="text-sm text-gray-500 mb-4">
          Stop all interactions and return to setup mode. Your data remains until you delete it.
        </p>
        <button
          onClick={handleRevokeConsent}
          className="bg-orange-100 text-orange-700 rounded-xl px-5 py-2.5 font-medium hover:bg-orange-200 transition-colors"
        >
          Revoke Consent
        </button>
      </div>

      {/* Delete All Data */}
      <div className="bg-white rounded-xl shadow-sm p-6 border-2 border-red-100">
        <h2 className="text-lg font-semibold text-red-700 mb-2">Delete All Data</h2>
        <p className="text-sm text-gray-500 mb-4">
          Permanently delete ALL data and reset the device. This cannot be undone.
          The device will return to first-time setup mode.
        </p>

        {!showErase ? (
          <button
            onClick={() => setShowErase(true)}
            className="bg-red-100 text-red-700 rounded-xl px-5 py-2.5 font-medium hover:bg-red-200 transition-colors"
          >
            Delete Everything
          </button>
        ) : (
          <div className="bg-red-50 rounded-xl p-4">
            <p className="text-sm text-red-700 font-medium mb-3">
              Enter your PIN to confirm permanent deletion:
            </p>
            <div className="flex gap-3">
              <input
                type="password"
                inputMode="numeric"
                maxLength={6}
                value={erasePin}
                onChange={(e) => setErasePin(e.target.value.replace(/\D/g, ''))}
                placeholder="Enter PIN"
                className="flex-1 border-2 border-red-200 rounded-lg px-3 py-2 text-center tracking-widest focus:border-red-500 focus:outline-none"
              />
              <button
                onClick={handleErase}
                disabled={erasing || erasePin.length < 4}
                className="bg-red-600 text-white rounded-lg px-5 py-2 font-medium hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {erasing ? 'Deleting...' : 'Confirm Delete'}
              </button>
              <button
                onClick={() => { setShowErase(false); setErasePin('') }}
                className="border border-gray-300 rounded-lg px-4 py-2 text-gray-600 hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
