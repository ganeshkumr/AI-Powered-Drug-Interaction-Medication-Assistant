import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Pill, Clock, Calendar, Edit2, Trash2, Sun, Sunset, Moon, Shield, AlertTriangle, XCircle, TrendingUp, List, BarChart3 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { medicationAPI } from '../services/api'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import EmptyState from '../components/common/EmptyState'
import LoadingSpinner from '../components/common/LoadingSpinner'
import MedicationCard from '../components/medication/MedicationCard'

/**
 * MyMedPage Component
 * 
 * Redesigned medication dashboard with improved layout and organization.
 * Features prominent "Add Medication" button and time-based grouping.
 * 
 * Requirements: 6.1, 6.2, 6.3, 6.5
 */
const MyMedPage = () => {
  const [medications, setMedications] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [viewMode, setViewMode] = useState('list') // 'list' or 'timeline'
  const navigate = useNavigate()

  useEffect(() => {
    fetchMedications()
  }, [])

  const fetchMedications = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await medicationAPI.getMedications()
      setMedications(response.data.medications || [])
    } catch (error) {
      console.error('Failed to fetch medications:', error)
      setError('Failed to load medications. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id) => {
    if (!confirm('Are you sure you want to remove this medication?')) return

    try {
      setMedications(medications.filter(med => med.id !== id))
      // In a real implementation, this would call the API to delete
    } catch (error) {
      console.error('Failed to delete medication:', error)
    }
  }

  const handleEdit = (id) => {
    // In a real implementation, this would open an edit modal
    alert('Edit functionality coming soon!')
  }

  // Calculate medication risk statistics
  const calculateRiskStats = (medications) => {
    const stats = {
      total: medications.length,
      safe: 0,
      needsAttention: 0,
      highRisk: 0
    }

    medications.forEach(med => {
      // For now, we'll simulate risk levels based on medication properties
      // In a real implementation, this would come from the API or be calculated based on interactions
      const riskLevel = med.risk_level || simulateRiskLevel(med)
      
      if (riskLevel === 'safe') {
        stats.safe++
      } else if (riskLevel === 'warning' || riskLevel === 'caution') {
        stats.needsAttention++
      } else if (riskLevel === 'high_risk' || riskLevel === 'dangerous') {
        stats.highRisk++
      } else {
        // Default to safe for unknown risk levels
        stats.safe++
      }
    })

    return stats
  }

  // Simulate risk level for medications (placeholder logic)
  const simulateRiskLevel = (medication) => {
    // This is placeholder logic - in a real implementation, 
    // risk would be determined by the backend based on interactions
    const drugName = medication.drug_name?.toLowerCase() || ''
    
    // Some common high-risk medications for demonstration
    const highRiskDrugs = ['warfarin', 'aspirin', 'ibuprofen', 'naproxen']
    const cautionDrugs = ['metformin', 'lisinopril', 'atorvastatin']
    
    if (highRiskDrugs.some(drug => drugName.includes(drug))) {
      return 'warning'
    } else if (cautionDrugs.some(drug => drugName.includes(drug))) {
      return 'caution'
    }
    
    return 'safe'
  }
  // Group medications by time periods
  const groupMedicationsByTime = (medications) => {
    const groups = {
      morning: [],
      afternoon: [],
      night: [],
      unspecified: []
    }

    medications.forEach(med => {
      // Determine time group based on frequency or time_of_day field
      const timeOfDay = med.time_of_day?.toLowerCase()
      const frequency = med.frequency?.toLowerCase() || ''

      if (timeOfDay === 'morning' || frequency.includes('morning') || frequency.includes('am')) {
        groups.morning.push(med)
      } else if (timeOfDay === 'afternoon' || frequency.includes('afternoon') || frequency.includes('noon')) {
        groups.afternoon.push(med)
      } else if (timeOfDay === 'night' || timeOfDay === 'evening' || frequency.includes('night') || frequency.includes('evening') || frequency.includes('pm')) {
        groups.night.push(med)
      } else {
        groups.unspecified.push(med)
      }
    })

    return groups
  }

  const medicationGroups = groupMedicationsByTime(medications)
  const riskStats = calculateRiskStats(medications)

  // Create timeline data for timeline view
  const createTimelineData = (medications) => {
    const timeSlots = [
      { time: '6:00 AM', label: 'Early Morning', medications: [] },
      { time: '8:00 AM', label: 'Morning', medications: [] },
      { time: '12:00 PM', label: 'Noon', medications: [] },
      { time: '2:00 PM', label: 'Afternoon', medications: [] },
      { time: '6:00 PM', label: 'Evening', medications: [] },
      { time: '9:00 PM', label: 'Night', medications: [] },
    ]

    medications.forEach(med => {
      const timeOfDay = med.time_of_day?.toLowerCase()
      const frequency = med.frequency?.toLowerCase() || ''

      // Assign medications to appropriate time slots
      if (timeOfDay === 'morning' || frequency.includes('morning') || frequency.includes('am')) {
        timeSlots[1].medications.push(med) // 8:00 AM
      } else if (timeOfDay === 'afternoon' || frequency.includes('afternoon') || frequency.includes('noon')) {
        timeSlots[3].medications.push(med) // 2:00 PM
      } else if (timeOfDay === 'night' || timeOfDay === 'evening' || frequency.includes('night') || frequency.includes('evening') || frequency.includes('pm')) {
        timeSlots[5].medications.push(med) // 9:00 PM
      } else {
        // For unspecified times, distribute across morning, afternoon, and evening
        const slotIndex = Math.floor(Math.random() * 3) * 2 + 1 // 1, 3, or 5
        timeSlots[slotIndex].medications.push(med)
      }
    })

    return timeSlots.filter(slot => slot.medications.length > 0)
  }

  const timelineData = createTimelineData(medications)

  const timeGroupConfig = [
    {
      key: 'morning',
      title: 'Morning',
      icon: Sun,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-200'
    },
    {
      key: 'afternoon', 
      title: 'Afternoon',
      icon: Sunset,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      borderColor: 'border-orange-200'
    },
    {
      key: 'night',
      title: 'Night',
      icon: Moon,
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-50',
      borderColor: 'border-indigo-200'
    }
  ]

  // Timeline View Component
  const TimelineView = () => (
    <div className="space-y-6">
      {timelineData.map((timeSlot, index) => (
        <motion.div
          key={timeSlot.time}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="relative"
        >
          {/* Timeline connector line */}
          {index < timelineData.length - 1 && (
            <div className="absolute left-6 top-16 w-0.5 h-full bg-neutral-200 -z-10" />
          )}
          
          <div className="flex items-start space-x-4">
            {/* Time indicator */}
            <div className="flex-shrink-0 w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center border-4 border-white shadow-sm">
              <Clock className="w-5 h-5 text-primary-600" />
            </div>
            
            {/* Time slot content */}
            <div className="flex-1 min-w-0">
              <div className="mb-3">
                <h3 className="text-lg font-semibold text-neutral-900">
                  {timeSlot.time}
                </h3>
                <p className="text-sm text-neutral-600">
                  {timeSlot.label} • {timeSlot.medications.length} {timeSlot.medications.length === 1 ? 'medication' : 'medications'}
                </p>
              </div>
              
              {/* Medications for this time slot */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <AnimatePresence>
                  {timeSlot.medications.map((med, medIndex) => (
                    <motion.div
                      key={med.id}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ delay: medIndex * 0.05 }}
                    >
                      <MedicationCard
                        medication={{
                          name: med.drug_name,
                          dosage: `${med.dosage_amount} ${med.dosage_unit}`,
                          frequency: med.frequency,
                          timeOfDay: timeSlot.label.toLowerCase()
                        }}
                        variant="dashboard"
                        onEdit={() => handleEdit(med.id)}
                        onRemove={() => handleDelete(med.id)}
                      />
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  )

  if (loading) {
    return (
      <div className="min-h-screen bg-neutral-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <LoadingSpinner message="Loading your medications..." />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-neutral-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Card>
            <div className="text-center py-12">
              <p className="text-red-600 mb-4">{error}</p>
              <Button onClick={fetchMedications}>Try Again</Button>
            </div>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen-mobile bg-neutral-50 prevent-horizontal-scroll">
      <div className="container-responsive py-4 sm:py-6 md:py-8">
        
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="text-center-mobile lg:text-left-desktop">
              <h1 className="text-responsive-3xl font-bold text-neutral-900 mb-2">
                My Medications
              </h1>
              <p className="text-responsive-base text-neutral-600">
                Manage your medication schedule and track your prescriptions
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* View Toggle - Only show when there are medications */}
              {medications.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.1 }}
                  className="flex items-center bg-white rounded-lg border border-neutral-200 p-1"
                >
                  <button
                    onClick={() => setViewMode('list')}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
                      viewMode === 'list'
                        ? 'bg-primary text-white shadow-sm'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-50'
                    }`}
                  >
                    <List className="w-4 h-4" />
                    <span>List</span>
                  </button>
                  <button
                    onClick={() => setViewMode('timeline')}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
                      viewMode === 'timeline'
                        ? 'bg-primary text-white shadow-sm'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-50'
                    }`}
                  >
                    <BarChart3 className="w-4 h-4" />
                    <span>Timeline</span>
                  </button>
                </motion.div>
              )}

              {/* Prominent Add Medication Button */}
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 }}
              >
                <Button
                  onClick={() => navigate('/check/medication')}
                  className="btn-responsive bg-primary hover:bg-primary-600 text-white rounded-lg font-medium shadow-soft hover:shadow-soft-lg transition-all touch-manipulation"
                  size="lg"
                >
                  <Plus className="w-5 h-5" />
                  <span>Add Medication</span>
                </Button>
              </motion.div>
            </div>
          </div>

          {/* Medication Count Summary */}
          {medications.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mt-6 flex items-center space-x-4 text-sm text-neutral-600"
            >
              <div className="flex items-center space-x-2">
                <Pill className="w-4 h-4 text-primary" />
                <span>
                  {medications.length} {medications.length === 1 ? 'medication' : 'medications'} total
                </span>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Summary Cards */}
        {medications.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mb-8"
          >
            <div className="grid-responsive-4">
              {/* Total Medications Card */}
              <Card className="card-responsive">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-neutral-600 mb-1">
                      Total Medications
                    </p>
                    <p className="text-3xl font-bold text-neutral-900">
                      {riskStats.total}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                    <Pill className="w-6 h-6 text-primary-600" />
                  </div>
                </div>
              </Card>

              {/* Safe Count Card */}
              <Card className="card-responsive">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-neutral-600 mb-1">
                      Safe
                    </p>
                    <p className="text-3xl font-bold text-success-600">
                      {riskStats.safe}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-success-100 rounded-lg flex items-center justify-center">
                    <Shield className="w-6 h-6 text-success-600" />
                  </div>
                </div>
              </Card>

              {/* Needs Attention Card */}
              <Card className="card-responsive">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-neutral-600 mb-1">
                      Needs Attention
                    </p>
                    <p className="text-3xl font-bold text-warning-600">
                      {riskStats.needsAttention}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-warning-100 rounded-lg flex items-center justify-center">
                    <AlertTriangle className="w-6 h-6 text-warning-600" />
                  </div>
                </div>
              </Card>

              {/* High Risk Card */}
              <Card className="card-responsive">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-neutral-600 mb-1">
                      High Risk
                    </p>
                    <p className="text-3xl font-bold text-danger-600">
                      {riskStats.highRisk}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-danger-100 rounded-lg flex items-center justify-center">
                    <XCircle className="w-6 h-6 text-danger-600" />
                  </div>
                </div>
              </Card>
            </div>
          </motion.div>
        )}

        {/* Empty State */}
        {medications.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <EmptyState
              icon={Pill}
              title="No medications yet"
              description="Start by adding your first medication to track your prescriptions and check for interactions."
              action={
                <Button
                  onClick={() => navigate('/check/medication')}
                  className="inline-flex items-center space-x-2"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add Your First Medication</span>
                </Button>
              }
            />
          </motion.div>
        ) : (
          /* Medication Display - List or Timeline View */
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            {viewMode === 'timeline' ? (
              /* Timeline View */
              timelineData.length > 0 ? (
                <Card className="p-6">
                  <div className="mb-6">
                    <h2 className="text-xl font-semibold text-neutral-900 mb-2">
                      Daily Schedule
                    </h2>
                    <p className="text-neutral-600">
                      Your medications organized by time of day
                    </p>
                  </div>
                  <TimelineView />
                </Card>
              ) : (
                <Card className="p-6">
                  <div className="text-center py-8">
                    <Clock className="w-12 h-12 text-neutral-300 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                      No scheduled medications
                    </h3>
                    <p className="text-neutral-600">
                      Add time information to your medications to see them in timeline view
                    </p>
                  </div>
                </Card>
              )
            ) : (
              /* List View - Existing grouped view */
              <div className="space-y-8">
                {timeGroupConfig.map((group, groupIndex) => {
                  const groupMedications = medicationGroups[group.key]
                  
                  if (groupMedications.length === 0) return null

                  return (
                    <motion.div
                      key={group.key}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 + groupIndex * 0.1 }}
                    >
                      <Card className="overflow-hidden">
                        {/* Group Header */}
                        <div className={`${group.bgColor} ${group.borderColor} border-b px-6 py-4`}>
                          <div className="flex items-center space-x-3">
                            <div className={`w-10 h-10 ${group.bgColor} rounded-lg flex items-center justify-center border ${group.borderColor}`}>
                              <group.icon className={`w-5 h-5 ${group.color}`} />
                            </div>
                            <div>
                              <h2 className="text-lg font-semibold text-neutral-900">
                                {group.title}
                              </h2>
                              <p className="text-sm text-neutral-600">
                                {groupMedications.length} {groupMedications.length === 1 ? 'medication' : 'medications'}
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* Medications Grid */}
                        <div className="p-4 sm:p-6">
                          <div className="grid-responsive-3">
                            <AnimatePresence>
                              {groupMedications.map((med, index) => (
                                <motion.div
                                  key={med.id}
                                  initial={{ opacity: 0, scale: 0.95 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  exit={{ opacity: 0, scale: 0.95 }}
                                  transition={{ delay: index * 0.05 }}
                                >
                                  <MedicationCard
                                    medication={{
                                      name: med.drug_name,
                                      dosage: `${med.dosage_amount} ${med.dosage_unit}`,
                                      frequency: med.frequency,
                                      timeOfDay: group.key
                                    }}
                                    variant="dashboard"
                                    onEdit={() => handleEdit(med.id)}
                                    onRemove={() => handleDelete(med.id)}
                                  />
                                </motion.div>
                              ))}
                            </AnimatePresence>
                          </div>
                        </div>
                      </Card>
                    </motion.div>
                  )
                })}

                {/* Unspecified Time Medications */}
                {medicationGroups.unspecified.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7 }}
                  >
                    <Card>
                      <div className="px-6 py-4 border-b border-neutral-200">
                        <div className="flex items-center space-x-3">
                          <div className="w-10 h-10 bg-neutral-100 rounded-lg flex items-center justify-center">
                            <Clock className="w-5 h-5 text-neutral-600" />
                          </div>
                          <div>
                            <h2 className="text-lg font-semibold text-neutral-900">
                              Other Medications
                            </h2>
                            <p className="text-sm text-neutral-600">
                              {medicationGroups.unspecified.length} {medicationGroups.unspecified.length === 1 ? 'medication' : 'medications'}
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className="p-4 sm:p-6">
                        <div className="grid-responsive-3">
                          <AnimatePresence>
                            {medicationGroups.unspecified.map((med, index) => (
                              <motion.div
                                key={med.id}
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                transition={{ delay: index * 0.05 }}
                              >
                                <MedicationCard
                                  medication={{
                                    name: med.drug_name,
                                    dosage: `${med.dosage_amount} ${med.dosage_unit}`,
                                    frequency: med.frequency,
                                    timeOfDay: 'unspecified'
                                  }}
                                  variant="dashboard"
                                  onEdit={() => handleEdit(med.id)}
                                  onRemove={() => handleDelete(med.id)}
                                />
                              </motion.div>
                            ))}
                          </AnimatePresence>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default MyMedPage