import { useState } from 'react'

export default function ConsentGate({ onComplete }) {
  const [step, setStep] = useState(1) // 1: info, 2: consent, 3: PIN setup
  const [pin, setPin] = useState('')
  const [confirmPin, setConfirmPin] = useState('')
  const [ageMin, setAgeMin] = useState(3)
  const [ageMax, setAgeMax] = useState(6)
  const [consentChecked, setConsentChecked] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit() {
    if (pin !== confirmPin) {
      setError('PINs do not match.')
      return
    }
    if (pin.length < 4) {
      setError('PIN must be at least 4 digits.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const res = await fetch('/api/consent/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pin,
          consent_given: true,
          child_age_min: ageMin,
          child_age_max: ageMax,
        }),
      })
      const data = await res.json()

      if (data.status === 'setup_complete') {
        onComplete()
      } else {
        setError(data.error || 'Setup failed.')
      }
    } catch (err) {
      setError('Could not connect to device.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-lg p-8 w-full max-w-lg">
        <h1 className="text-2xl font-bold text-center mb-1">Learning Buddy</h1>
        <p className="text-gray-500 text-center mb-8">First-time setup</p>

        {/* Step 1: Information */}
        {step === 1 && (
          <div>
            <h2 className="text-lg font-semibold mb-4">Welcome!</h2>
            <p className="text-gray-600 mb-4">
              Learning Buddy is a voice-based AI companion that helps your child learn through conversation.
              Before we begin, we need your consent as required by the Digital Personal Data Protection Act (DPDP).
            </p>
            <h3 className="font-semibold mb-2">What data is collected:</h3>
            <ul className="text-gray-600 text-sm mb-4 space-y-1">
              <li>- Session timestamps and duration</li>
              <li>- Topic categories (e.g., "math", "animals")</li>
            </ul>
            <h3 className="font-semibold mb-2">What is NOT collected:</h3>
            <ul className="text-gray-600 text-sm mb-6 space-y-1">
              <li>- Voice recordings (audio is processed in memory and discarded)</li>
              <li>- Conversation text (discarded after the response is generated)</li>
              <li>- Personal information of any kind</li>
            </ul>
            <button
              onClick={() => setStep(2)}
              className="w-full bg-blue-600 text-white rounded-xl py-3 font-semibold hover:bg-blue-700 transition-colors"
            >
              Continue
            </button>
          </div>
        )}

        {/* Step 2: Consent */}
        {step === 2 && (
          <div>
            <h2 className="text-lg font-semibold mb-4">Parental Consent</h2>
            <div className="bg-gray-50 rounded-xl p-4 mb-4 text-sm text-gray-600">
              <p className="mb-2">By giving consent, you agree that:</p>
              <ul className="space-y-1">
                <li>- Your child will interact with an AI learning companion</li>
                <li>- Session metadata (timestamps, topic categories) will be stored on this device</li>
                <li>- No data is sent to external servers in offline mode</li>
                <li>- In online mode, the child's question text (not audio) is sent to the selected AI provider</li>
                <li>- You can revoke consent and delete all data at any time</li>
              </ul>
            </div>

            <div className="mb-4">
              <label className="font-medium text-sm">Child's age range:</label>
              <div className="flex gap-3 mt-2">
                <select
                  value={ageMin}
                  onChange={(e) => setAgeMin(Number(e.target.value))}
                  className="border rounded-lg px-3 py-2 flex-1"
                >
                  {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((a) => (
                    <option key={a} value={a}>{a} years</option>
                  ))}
                </select>
                <span className="self-center text-gray-400">to</span>
                <select
                  value={ageMax}
                  onChange={(e) => setAgeMax(Number(e.target.value))}
                  className="border rounded-lg px-3 py-2 flex-1"
                >
                  {[3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((a) => (
                    <option key={a} value={a}>{a} years</option>
                  ))}
                </select>
              </div>
            </div>

            <label className="flex items-start gap-3 mb-6 cursor-pointer">
              <input
                type="checkbox"
                checked={consentChecked}
                onChange={(e) => setConsentChecked(e.target.checked)}
                className="mt-1 w-5 h-5 rounded border-gray-300"
              />
              <span className="text-sm text-gray-700">
                I am the parent/guardian and I give consent for my child to use Learning Buddy.
                I understand how data is collected and stored.
              </span>
            </label>

            <div className="flex gap-3">
              <button
                onClick={() => setStep(1)}
                className="flex-1 border border-gray-300 rounded-xl py-3 font-medium text-gray-600 hover:bg-gray-50 transition-colors"
              >
                Back
              </button>
              <button
                onClick={() => setStep(3)}
                disabled={!consentChecked}
                className="flex-1 bg-blue-600 text-white rounded-xl py-3 font-semibold hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                Continue
              </button>
            </div>
          </div>
        )}

        {/* Step 3: PIN setup */}
        {step === 3 && (
          <div>
            <h2 className="text-lg font-semibold mb-4">Set a Parent PIN</h2>
            <p className="text-gray-500 text-sm mb-4">
              This PIN protects the settings. You'll need it to access this dashboard.
            </p>
            <input
              type="password"
              inputMode="numeric"
              maxLength={6}
              value={pin}
              onChange={(e) => setPin(e.target.value.replace(/\D/g, ''))}
              placeholder="Create PIN (4-6 digits)"
              className="w-full text-center text-xl tracking-widest border-2 border-gray-200 rounded-xl py-3 px-4 mb-3 focus:border-blue-500 focus:outline-none"
            />
            <input
              type="password"
              inputMode="numeric"
              maxLength={6}
              value={confirmPin}
              onChange={(e) => setConfirmPin(e.target.value.replace(/\D/g, ''))}
              placeholder="Confirm PIN"
              className="w-full text-center text-xl tracking-widest border-2 border-gray-200 rounded-xl py-3 px-4 mb-4 focus:border-blue-500 focus:outline-none"
            />
            {error && <p className="text-red-500 text-sm text-center mb-4">{error}</p>}
            <div className="flex gap-3">
              <button
                onClick={() => setStep(2)}
                className="flex-1 border border-gray-300 rounded-xl py-3 font-medium text-gray-600 hover:bg-gray-50 transition-colors"
              >
                Back
              </button>
              <button
                onClick={handleSubmit}
                disabled={loading || pin.length < 4}
                className="flex-1 bg-blue-600 text-white rounded-xl py-3 font-semibold hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Setting up...' : 'Complete Setup'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
