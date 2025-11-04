import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Pill, Trash2, Clock, Calendar } from 'lucide-react'
import { medicationAPI } from '../../services/api'
import Card from '../common/Card'

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
      setMedications(medications.filter(med => med.id !== id))
    } catch (error) {
      console.error('Failed to delete medication:', error)
    }
  }

  if (loading) {
    return (
      <Card>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <p className="ml-3 text-gray-600">Loading medications...</p>
        </div>
      </Card>
    )
  }

  return (
    <Card>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-heading font-bold text-neutral-text">
          💊 Medication Wallet
        </h3>
        <span className="px-3 py-1 bg-primary-50 text-primary text-sm font-medium rounded-full">
          {medications.length} {medications.length === 1 ? 'medication' : 'medications'}
        </span>
      </div>

      {medications.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-20 h-20 bg-primary-50 rounded-full flex items-center justify-center mx-auto mb-4">
            <Pill className="w-10 h-10 text-primary" />
          </div>
          <p className="text-lg font-medium text-gray-700 mb-2">No medications yet</p>
          <p className="text-sm text-gray-500">Add your first medication above to get started</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {medications.map((med, index) => (
            <motion.div
              key={med.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className="group relative p-4 bg-gradient-to-br from-primary-50 to-white dark:from-slate-700 dark:to-slate-800 rounded-card border border-primary-100 dark:border-slate-600 hover:shadow-soft-lg transition-all"
            >
              {/* Pill Icon */}
              <div className="absolute top-4 right-4">
                <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center shadow-soft">
                  <Pill className="w-5 h-5 text-white" />
                </div>
              </div>

              {/* Content */}
              <div className="pr-12">
                <h4 className="text-lg font-heading font-bold text-neutral-text mb-2">
                  {med.drug_name}
                </h4>
                
                <div className="space-y-2 text-sm">
                  <div className="flex items-center text-gray-600">
                    <span className="font-medium">Dosage:</span>
                    <span className="ml-2">{med.dosage_amount} {med.dosage_unit}</span>
                  </div>
                  
                  <div className="flex items-center text-gray-600">
                    <Clock className="w-4 h-4 mr-2" />
                    <span>{med.frequency}</span>
                  </div>
                  
                  {med.start_date && (
                    <div className="flex items-center text-gray-600">
                      <Calendar className="w-4 h-4 mr-2" />
                      <span>Started: {new Date(med.start_date).toLocaleDateString()}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Delete Button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleDelete(med.id)}
                className="absolute bottom-4 right-4 p-2 text-status-danger hover:bg-red-50 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                aria-label="Delete medication"
              >
                <Trash2 className="w-4 h-4" />
              </motion.button>
            </motion.div>
          ))}
        </div>
      )}
    </Card>
  )
}

export default MedicationList
