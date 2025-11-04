import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Loader, CheckCircle, AlertCircle } from 'lucide-react'
import { medicationAPI } from '../../services/api'
import Card from '../common/Card'
import Button from '../common/Button'

const MedicationForm = () => {
  const [drugName, setDrugName] = useState('')
  const [dosage, setDosage] = useState('')
  const [dosageUnit, setDosageUnit] = useState('mg')
  const [frequency, setFrequency] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [loading, setLoading] = useState(false)
  const [checkResult, setCheckResult] = useState(null)

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
    if (!checkResult || checkResult.status === 'UNSAFE') {
      alert('Cannot add unsafe medication')
      return
    }

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
          <label className="block text-sm font-medium mb-1">Drug Name</label>
          <input
            type="text"
            value={drugName}
            onChange={(e) => setDrugName(e.target.value)}
            placeholder="e.g., Lisinopril"
            className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
          />
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Dosage Amount</label>
            <input
              type="number"
              value={dosage}
              onChange={(e) => setDosage(e.target.value)}
              placeholder="e.g., 10"
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Unit</label>
            <select
              value={dosageUnit}
              onChange={(e) => setDosageUnit(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
            >
              <option value="mg">mg</option>
              <option value="g">g</option>
              <option value="ml">ml</option>
              <option value="mcg">mcg</option>
              <option value="IU">IU</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Frequency</label>
            <input
              type="text"
              value={frequency}
              onChange={(e) => setFrequency(e.target.value)}
              placeholder="e.g., Once daily"
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">End Date (Optional)</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
            />
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
    </motion.div>
  )
}

export default MedicationForm
