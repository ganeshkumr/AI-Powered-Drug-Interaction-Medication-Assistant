import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Shield, AlertCircle, CheckCircle2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import Button from '../common/Button'
import DrugSearch from '../medication/DrugSearch'
import MedicationChip from '../medication/MedicationChip'

const QuickCheckModal = ({ onClose }) => {
  const navigate = useNavigate()
  const [selectedDrugs, setSelectedDrugs] = useState([])
  const [useProfile, setUseProfile] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleAddDrug = (drug) => {
    if (!selectedDrugs.includes(drug)) {
      setSelectedDrugs([...selectedDrugs, drug])
      setError(null)
    }
  }

  const handleRemoveDrug = (drugToRemove) => {
    setSelectedDrugs(selectedDrugs.filter(drug => drug !== drugToRemove))
  }

  const handleCheckRisk = async () => {
    if (selectedDrugs.length < 1) {
      setError('Please add at least one medication')
      return
    }

    if (useProfile) {
      // Redirect to login if not authenticated
      const token = localStorage.getItem('token')
      if (!token) {
        navigate('/login', { state: { from: '/dashboard', drugs: selectedDrugs } })
        return
      }
    }

    setLoading(true)
    setError(null)

    try {
      // For quick check without profile
      const response = await fetch('http://localhost:5000/api/quick-check', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          drugs: selectedDrugs,
          use_profile: useProfile
        })
      })

      if (response.ok) {
        const data = await response.json()
        // Navigate to results page with data
        navigate('/results', { state: { result: data, drugs: selectedDrugs } })
      } else {
        const errorData = await response.json()
        setError(errorData.error || 'Failed to check interactions')
      }
    } catch (err) {
      console.error('Quick check error:', err)
      setError('Network error. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Info Banner */}
      <div className="bg-primary-50 border border-primary-100 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <Shield className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm text-primary-700">
              <strong>Quick Check:</strong> Get instant results without signing in.
              For more accurate, personalized checks based on your health profile,{' '}
              <button
                onClick={() => navigate('/register')}
                className="underline font-medium hover:text-primary-800"
              >
                create an account
              </button>
              .
            </p>
          </div>
        </div>
      </div>

      {/* Drug Search */}
      <div>
        <label className="block text-sm font-medium text-neutral-text mb-2">
          Add medications to check
        </label>
        <DrugSearch
          onSelect={handleAddDrug}
          placeholder="Search for a medication (e.g., Aspirin, Ibuprofen)..."
        />
        <p className="text-xs text-gray-500 mt-2">
          Type at least 2 characters to search
        </p>
      </div>

      {/* Selected Drugs */}
      {selectedDrugs.length > 0 && (
        <div>
          <label className="block text-sm font-medium text-neutral-text mb-3">
            Selected medications ({selectedDrugs.length})
          </label>
          <div className="flex flex-wrap gap-2">
            <AnimatePresence>
              {selectedDrugs.map((drug) => (
                <MedicationChip
                  key={drug}
                  drug={drug}
                  onRemove={() => handleRemoveDrug(drug)}
                />
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}

      {/* Profile Check Toggle */}
      <div className="bg-gray-50 rounded-lg p-4">
        <label className="flex items-start space-x-3 cursor-pointer">
          <input
            type="checkbox"
            checked={useProfile}
            onChange={(e) => setUseProfile(e.target.checked)}
            className="mt-1 w-4 h-4 text-primary border-gray-300 rounded focus:ring-primary"
          />
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium text-neutral-text">
                Check with my health profile
              </span>
              <span className="text-xs bg-accent text-white px-2 py-0.5 rounded-full">
                More Accurate
              </span>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              Include your age, conditions, and current medications for personalized analysis
              {!localStorage.getItem('token') && ' (requires sign in)'}
            </p>
          </div>
        </label>
      </div>

      {/* Error Message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-danger-50 border border-danger-200 rounded-lg p-4 flex items-start space-x-3"
        >
          <AlertCircle className="w-5 h-5 text-danger flex-shrink-0 mt-0.5" />
          <p className="text-sm text-danger-700">{error}</p>
        </motion.div>
      )}

      {/* Action Buttons */}
      <div className="flex items-center justify-between pt-4 border-t border-gray-200">
        <div className="flex items-center space-x-2 text-xs text-gray-500">
          <CheckCircle2 className="w-4 h-4" />
          <span>Your data is not stored</span>
        </div>
        
        <div className="flex items-center space-x-3">
          <Button
            variant="secondary"
            onClick={onClose}
            disabled={loading}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleCheckRisk}
            loading={loading}
            disabled={selectedDrugs.length === 0}
            icon={<Activity className="w-4 h-4" />}
          >
            Check Risk
          </Button>
        </div>
      </div>
    </div>
  )
}

export default QuickCheckModal
