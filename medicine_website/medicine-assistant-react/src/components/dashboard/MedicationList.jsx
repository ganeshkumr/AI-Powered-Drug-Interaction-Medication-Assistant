import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Pill, Trash2, Clock } from 'lucide-react'
import { medicationAPI } from '../../services/api'

const MedicationList = () => {
  const [medications, setMedications] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchMedications()
  }, [])

  const fetchMedications = async () => {
    try {
      const response = await medicationAPI.getMedications()
      setMedications(response.data.medications || [])
    } catch (error) {
      console.error('Failed to fetch medications:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id) => {
    if (!confirm('Are you sure you want to remove this medication?')) return

    try {
      // Add delete API call here when backend is ready
      setMedications(medications.filter(med => med.id !== id))
    } catch (error) {
      console.error('Failed to delete medication:', error)
    }
  }

  if (loading) {
    return (
      <div className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-200 dark:border-slate-600">
        <p className="text-center text-slate-600 dark:text-slate-400">Loading medications...</p>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-200 dark:border-slate-600"
    >
      <h3 className="text-2xl font-bold text-slate-800 dark:text-white mb-4">
        📋 Your Medications
      </h3>

      {medications.length === 0 ? (
        <div className="text-center py-8">
          <Pill className="w-16 h-16 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
          <p className="text-slate-600 dark:text-slate-400">No medications added yet</p>
          <p className="text-sm text-slate-500 dark:text-slate-500">Add your first medication above</p>
        </div>
      ) : (
        <div className="space-y-3">
          {medications.map((med, index) => (
            <motion.div
              key={med.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700 rounded-lg border border-slate-200 dark:border-slate-600 hover:shadow-md transition-shadow"
            >
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
                  <Pill className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h4 className="font-semibold text-slate-800 dark:text-white">
                    {med.drug_name}
                  </h4>
                  <div className="flex items-center space-x-3 text-sm text-slate-600 dark:text-slate-400">
                    <span>{med.dosage}</span>
                    <span className="flex items-center space-x-1">
                      <Clock className="w-3 h-3" />
                      <span>{med.frequency}</span>
                    </span>
                  </div>
                </div>
              </div>

              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleDelete(med.id)}
                className="p-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
              >
                <Trash2 className="w-5 h-5" />
              </motion.button>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  )
}

export default MedicationList
