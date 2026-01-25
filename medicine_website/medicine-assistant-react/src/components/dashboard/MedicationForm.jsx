import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Loader, CheckCircle, AlertCircle, Calendar } from 'lucide-react'
import { medicationAPI } from '../../services/api'
import Card from '../common/Card'
import Button from '../common/Button'
import DrugSearch from '../medication/DrugSearch'
import WarningModal from '../common/WarningModal'

const MedicationForm = () => {
  const [drugName, setDrugName] = useState('')
  const [dosage, setDosage] = useState('')
  const [dosageUnit, setDosageUnit] = useState('mg')
  const [frequency, setFrequency] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [loading, setLoading] = useState(false)
  const [checkResult, setCheckResult] = useState(null)
  const [showWarningModal, setShowWarningModal] = useState(false)

  const handleCheck = async () => {
    if (!drugName) {
      alert('Please enter a drug name')
      return
    }

    setLoading(true)
    setCheckResult(null)

    try {
      const response = await medicationAPI.checkBeforeAdding({
        drug_name: drugName,
        dosage_amount: dosage,
        dosage_unit: dosageUnit,
        frequency
      })
      setCheckResult(response.data)
      
      // Show warning modal for unsafe medications (non-blocking)
      if (response.data.verdict?.includes('UNSAFE') || (response.data.gnn_risk && response.data.gnn_risk > 70)) {
        setShowWarningModal(true)
      }
    } catch (error) {
      console.error('Check failed:', error)
      setCheckResult({
        verdict: 'ERROR',
        ai_response: 'Failed to check medication. Please try again.',
        gnn_risk: 0
      })
    } finally {
      setLoading(false)
    }
  }

  const handleAdd = async () => {
    if (!checkResult) {
      alert('Please check medication safety first')
      return
    }

    // Allow user to add medication even if unsafe (non-blocking warning)
    try {
      await medicationAPI.addMedication({
        drug_name: drugName,
        dosage_amount: dosage,
        dosage_unit: dosageUnit,
        frequency,
        start_date: startDate,
        end_date: endDate
      })
      
      // Reset form
      setDrugName('')
      setDosage('')
      setDosageUnit('mg')
      setFrequency('')
      setStartDate('')
      setEndDate('')
      setCheckResult(null)
      
      alert('Medication added successfully!')
      window.location.reload() // Refresh to show new medication
    } catch (error) {
      console.error('Add medication failed:', error)
      alert('Failed to add medication')
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-200 dark:border-slate-600"
    >
      <h3 className="text-2xl font-bold text-slate-800 dark:text-white mb-4">
        💊 Add New Medication
      </h3>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-neutral-text mb-2">
            Drug Name *
          </label>
          <DrugSearch
            onSelect={(drug) => setDrugName(drug)}
            placeholder="Search for a medication (e.g., Lisinopril)..."
          />
          {drugName && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-2 text-sm text-primary font-medium"
            >
              Selected: {drugName}
            </motion.div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-neutral-text mb-2">
              Dosage Amount *
            </label>
            <input
              type="number"
              value={dosage}
              onChange={(e) => setDosage(e.target.value)}
              placeholder="e.g., 10"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-neutral-text mb-2">
              Unit *
            </label>
            <select
              value={dosageUnit}
              onChange={(e) => setDosageUnit(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
            >
              <option value="mg">mg (milligrams)</option>
              <option value="g">g (grams)</option>
              <option value="ml">ml (milliliters)</option>
              <option value="mcg">mcg (micrograms)</option>
              <option value="IU">IU (International Units)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-neutral-text mb-2">
              Frequency *
            </label>
            <select
              value={frequency}
              onChange={(e) => setFrequency(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
            >
              <option value="">Select frequency</option>
              <option value="Once daily">Once daily</option>
              <option value="Twice daily">Twice daily</option>
              <option value="Three times daily">Three times daily</option>
              <option value="Four times daily">Four times daily</option>
              <option value="Every 4 hours">Every 4 hours</option>
              <option value="Every 6 hours">Every 6 hours</option>
              <option value="Every 8 hours">Every 8 hours</option>
              <option value="Every 12 hours">Every 12 hours</option>
              <option value="As needed">As needed</option>
              <option value="Weekly">Weekly</option>
              <option value="Monthly">Monthly</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-neutral-text mb-2 flex items-center space-x-2">
              <Calendar className="w-4 h-4" />
              <span>Start Date</span>
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-neutral-text mb-2 flex items-center space-x-2">
              <Calendar className="w-4 h-4" />
              <span>End Date (Optional)</span>
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
            />
            <p className="text-xs text-gray-500 mt-1">
              Leave empty for ongoing medication
            </p>
          </div>
        </div>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleCheck}
          disabled={loading}
          className="w-full py-3 bg-primary hover:bg-primary-600 text-white rounded-lg font-medium disabled:opacity-50 flex items-center justify-center space-x-2 shadow-soft hover:shadow-soft-lg transition-all"
        >
          {loading ? (
            <>
              <Loader className="w-5 h-5 animate-spin" />
              <span>Checking...</span>
            </>
          ) : (
            <>
              <CheckCircle className="w-5 h-5" />
              <span>Check Safety</span>
            </>
          )}
        </motion.button>

        <AnimatePresence>
          {checkResult && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-3"
            >
              {/* GNN Risk Score */}
              {checkResult.gnn_risk !== undefined && (
                <div className="p-4 bg-gradient-to-br from-primary-50 to-accent-50 rounded-card border border-primary-100 shadow-soft">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-semibold text-primary mb-1">
                        🤖 AI Risk Prediction
                      </p>
                      <p className={`text-3xl font-bold font-heading ${
                        checkResult.gnn_risk > 70 ? 'text-status-danger' :
                        checkResult.gnn_risk > 40 ? 'text-status-warning' :
                        'text-status-safe'
                      }`}>
                        {checkResult.gnn_risk}%
                      </p>
                      <p className="text-xs text-gray-600 mt-1">
                        Interaction Risk Score
                      </p>
                    </div>
                    <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                      checkResult.gnn_risk > 70 ? 'bg-red-100' :
                      checkResult.gnn_risk > 40 ? 'bg-yellow-100' :
                      'bg-green-100'
                    }`}>
                      <span className={`text-2xl ${
                        checkResult.gnn_risk > 70 ? 'text-status-danger' :
                        checkResult.gnn_risk > 40 ? 'text-status-warning' :
                        'text-status-safe'
                      }`}>
                        {checkResult.gnn_risk > 70 ? '⚠️' :
                         checkResult.gnn_risk > 40 ? '⚡' :
                         '✓'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* AI Analysis */}
              <div className={`p-4 rounded-card border-l-4 shadow-soft ${
                checkResult.verdict?.includes('SAFE')
                  ? 'bg-green-50 dark:bg-green-900/20 border-status-safe'
                  : 'bg-red-50 dark:bg-red-900/20 border-status-danger'
              }`}>
                <div className="flex items-start space-x-2 mb-3">
                  {checkResult.verdict?.includes('SAFE') ? (
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <p className="font-bold text-lg mb-2">{checkResult.verdict}</p>
                    <div 
                      className="text-sm prose prose-sm max-w-none dark:prose-invert"
                      dangerouslySetInnerHTML={{ __html: checkResult.ai_response?.replace(/\n/g, '<br>') }}
                    />
                  </div>
                </div>

                {checkResult.verdict?.includes('SAFE') && (
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleAdd}
                    className="w-full py-3 bg-status-safe hover:bg-green-700 text-white rounded-lg font-medium flex items-center justify-center space-x-2 shadow-soft hover:shadow-soft-lg transition-all"
                  >
                    <Plus className="w-5 h-5" />
                    <span>Confirm & Add to Profile</span>
                  </motion.button>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Warning Modal for Unsafe Medications */}
      <WarningModal
        isOpen={showWarningModal}
        onClose={() => setShowWarningModal(false)}
        onProceed={() => {
          // User acknowledges the warning and can continue
          console.log('User acknowledged medication safety warning')
        }}
        riskData={{
          overallRisk: checkResult?.gnn_risk,
          overallVerdict: checkResult?.verdict,
          results: checkResult ? [{
            medication: drugName,
            ai_response: checkResult.ai_response,
            gnn_risk: checkResult.gnn_risk,
            verdict: checkResult.verdict
          }] : []
        }}
        medications={[drugName]}
        title="Medication Safety Warning"
        showProceedButton={true}
        proceedButtonText="I Understand the Risks"
      />
    </motion.div>
  )
}

export default MedicationForm
