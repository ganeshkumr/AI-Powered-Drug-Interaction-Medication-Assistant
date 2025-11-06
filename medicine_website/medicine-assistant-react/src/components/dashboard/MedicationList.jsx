import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Pill, Trash2, Clock, Calendar, Edit2, Bell, BellOff } from 'lucide-react'
import { medicationAPI } from '../../services/api'
import Card from '../common/Card'
import Button from '../common/Button'

const MedicationList = () => {
  const [medications, setMedications] = useState([])
  const [loading, setLoading] = useState(true)
  const [editingId, setEditingId] = useState(null)

  useEffect(() => {
    fetchMedications()
  }, [])

  const fetchMedications = async () => {
    try {
      const response = await medicationAPI.getMedications()
      // Add reminder state to each medication (default false)
      const medsWithReminders = (response.data.medications || []).map(med => ({
        ...med,
        reminderEnabled: med.reminderEnabled || false
      }))
      setMedications(medsWithReminders)
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

  const handleToggleReminder = (id) => {
    setMedications(medications.map(med => 
      med.id === id ? { ...med, reminderEnabled: !med.reminderEnabled } : med
    ))
  }

  const handleEdit = (id) => {
    setEditingId(id)
    // In a real implementation, this would open an edit modal
    alert('Edit functionality coming soon!')
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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <AnimatePresence>
            {medications.map((med, index) => (
              <motion.div
                key={med.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ delay: index * 0.05 }}
                className="group relative bg-white rounded-card-lg border border-gray-200 hover:border-primary-200 hover:shadow-soft-lg transition-all overflow-hidden"
              >
                {/* Header with Pill Icon */}
                <div className="bg-gradient-to-br from-primary-50 to-accent-50 p-4 border-b border-gray-100">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="text-lg font-heading font-bold text-neutral-text mb-1">
                        {med.drug_name}
                      </h4>
                      <div className="flex items-center text-sm text-gray-600">
                        <span className="font-semibold text-primary">
                          {med.dosage_amount} {med.dosage_unit}
                        </span>
                      </div>
                    </div>
                    <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center shadow-soft">
                      <Pill className="w-6 h-6 text-white" />
                    </div>
                  </div>
                </div>

                {/* Content */}
                <div className="p-4 space-y-3">
                  <div className="flex items-center text-sm text-gray-600">
                    <Clock className="w-4 h-4 mr-2 text-primary" />
                    <span>{med.frequency}</span>
                  </div>
                  
                  {med.start_date && (
                    <div className="flex items-center text-sm text-gray-600">
                      <Calendar className="w-4 h-4 mr-2 text-primary" />
                      <span>Started: {new Date(med.start_date).toLocaleDateString()}</span>
                    </div>
                  )}

                  {med.end_date && (
                    <div className="flex items-center text-sm text-gray-600">
                      <Calendar className="w-4 h-4 mr-2 text-accent" />
                      <span>Ends: {new Date(med.end_date).toLocaleDateString()}</span>
                    </div>
                  )}

                  {/* Reminder Toggle */}
                  <div className="pt-3 border-t border-gray-100">
                    <button
                      onClick={() => handleToggleReminder(med.id)}
                      className={`flex items-center justify-between w-full px-3 py-2 rounded-lg transition-colors ${
                        med.reminderEnabled
                          ? 'bg-primary-50 text-primary'
                          : 'bg-gray-50 text-gray-600 hover:bg-gray-100'
                      }`}
                    >
                      <span className="text-sm font-medium">Reminders</span>
                      {med.reminderEnabled ? (
                        <Bell className="w-4 h-4" />
                      ) : (
                        <BellOff className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center border-t border-gray-100 divide-x divide-gray-100">
                  <motion.button
                    whileHover={{ backgroundColor: 'rgba(46, 167, 155, 0.05)' }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleEdit(med.id)}
                    className="flex-1 flex items-center justify-center space-x-2 py-3 text-primary hover:text-primary-600 transition-colors"
                  >
                    <Edit2 className="w-4 h-4" />
                    <span className="text-sm font-medium">Edit</span>
                  </motion.button>
                  
                  <motion.button
                    whileHover={{ backgroundColor: 'rgba(239, 68, 68, 0.05)' }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleDelete(med.id)}
                    className="flex-1 flex items-center justify-center space-x-2 py-3 text-danger hover:text-red-700 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    <span className="text-sm font-medium">Delete</span>
                  </motion.button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </Card>
  )
}

export default MedicationList
