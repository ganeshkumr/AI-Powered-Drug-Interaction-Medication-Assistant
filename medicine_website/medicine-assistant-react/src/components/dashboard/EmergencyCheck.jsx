import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, Loader } from 'lucide-react'
import { emergencyAPI } from '../../services/api'

const EmergencyCheck = () => {
  const [drug1, setDrug1] = useState('')
  const [drug2, setDrug2] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleCheck = async () => {
    if (!drug1 || !drug2) {
      alert('Please enter both drug names')
      return
    }

    setLoading(true)
    setResult(null)

    try {
      const response = await emergencyAPI.checkInteraction(drug1, drug2)
      setResult(response.data)
    } catch (error) {
      console.error('Emergency check failed:', error)
      setResult({
        status: 'UNSAFE',
        response: 'Emergency check failed. Please try again or consult a healthcare professional.',
        gnn_risk: 0
      })
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'UNSAFE':
        return 'bg-red-100 dark:bg-red-900/20 border-red-500 text-red-800 dark:text-red-400'
      case 'CAUTION':
        return 'bg-yellow-100 dark:bg-yellow-900/20 border-yellow-500 text-yellow-800 dark:text-yellow-400'
      default:
        return 'bg-green-100 dark:bg-green-900/20 border-green-500 text-green-800 dark:text-green-400'
    }
  }

  const getRiskColor = (risk) => {
    if (risk > 70) return 'text-red-600 dark:text-red-400'
    if (risk > 40) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-green-600 dark:text-green-400'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-slate-800 dark:to-slate-700 p-6 rounded-card shadow-soft-lg border border-red-200 dark:border-slate-600"
    >
      <div className="flex items-center mb-4">
        <div className="w-12 h-12 bg-status-danger rounded-full flex items-center justify-center mr-3 shadow-soft">
          <AlertTriangle className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-heading font-bold text-status-danger">
            🚨 Emergency Drug Check
          </h3>
          <p className="text-sm text-gray-600">
            Quick interaction check - no profile needed
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <motion.div whileFocus={{ scale: 1.02 }}>
          <label className="block text-sm font-medium text-red-700 dark:text-red-300 mb-1">
            Drug 1
          </label>
          <input
            type="text"
            value={drug1}
            onChange={(e) => setDrug1(e.target.value)}
            placeholder="e.g., Warfarin"
            className="w-full px-4 py-3 border border-red-300 dark:border-red-700 rounded-lg focus:ring-2 focus:ring-red-500 dark:bg-slate-700"
          />
        </motion.div>

        <motion.div whileFocus={{ scale: 1.02 }}>
          <label className="block text-sm font-medium text-red-700 dark:text-red-300 mb-1">
            Drug 2
          </label>
          <input
            type="text"
            value={drug2}
            onChange={(e) => setDrug2(e.target.value)}
            placeholder="e.g., Aspirin"
            className="w-full px-4 py-3 border border-red-300 dark:border-red-700 rounded-lg focus:ring-2 focus:ring-red-500 dark:bg-slate-700"
          />
        </motion.div>
      </div>

      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleCheck}
        disabled={loading}
        className="w-full py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
      >
        {loading ? (
          <>
            <Loader className="w-5 h-5 animate-spin" />
            <span>Analyzing...</span>
          </>
        ) : (
          <span>🚨 Emergency Check</span>
        )}
      </motion.button>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 p-4 bg-white dark:bg-slate-800 border border-red-200 dark:border-slate-600 rounded-lg"
          >
            {/* Status Badge */}
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              className={`mb-3 p-3 border-l-4 rounded ${getStatusColor(result.status)}`}
            >
              <p className="font-bold text-lg">
                {result.status === 'UNSAFE' && '🚨 DO NOT COMBINE'}
                {result.status === 'CAUTION' && '⚠️ USE WITH CAUTION'}
                {result.status === 'SAFE' && '✅ APPEARS SAFE'}
              </p>
            </motion.div>

            {/* GNN Prediction */}
            {result.gnn_risk !== undefined && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-3 p-3 bg-slate-50 dark:bg-slate-700 rounded border border-slate-200 dark:border-slate-600"
              >
                <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  🤖 AI Prediction:
                </p>
                <p className={`text-3xl font-bold ${getRiskColor(result.gnn_risk)}`}>
                  {result.gnn_risk}%
                </p>
                <p className="text-xs text-slate-600 dark:text-slate-400">
                  Interaction Risk Score
                </p>
              </motion.div>
            )}

            {/* Detailed Response */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="prose prose-sm max-w-none dark:prose-invert"
              dangerouslySetInnerHTML={{ __html: result.response.replace(/\n/g, '<br>') }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default EmergencyCheck