import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, ArrowRight, ArrowLeft } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import DrugSearch from '../components/medication/DrugSearch'
import StepIndicator from '../components/navigation/StepIndicator'
import Button from '../components/common/Button'
import Card from '../components/common/Card'
import { manageFocusTransition, announceToScreenReader } from '../utils/accessibility'

const MedicationStep = () => {
  const [selectedMedications, setSelectedMedications] = useState([])
  const navigate = useNavigate()

  // Announce page load to screen readers
  useEffect(() => {
    manageFocusTransition(1, 'forward')
    announceToScreenReader('Medication selection step loaded. Add medications to check for interactions.')
  }, [])

  const handleAddMedication = (drugName) => {
    if (!selectedMedications.includes(drugName)) {
      setSelectedMedications([...selectedMedications, drugName])
      announceToScreenReader(`${drugName} added to medication list. ${selectedMedications.length + 1} medications selected.`)
    }
  }

  const handleRemoveMedication = (drugName) => {
    setSelectedMedications(selectedMedications.filter(med => med !== drugName))
    announceToScreenReader(`${drugName} removed from medication list. ${selectedMedications.length - 1} medications remaining.`)
  }

  const handleNext = () => {
    // Store medications in sessionStorage for the next step
    sessionStorage.setItem('selectedMedications', JSON.stringify(selectedMedications))
    announceToScreenReader('Proceeding to dosage information step')
    navigate('/check/dosage')
  }

  const canProceed = selectedMedications.length > 0

  return (
    <div className="min-h-screen bg-neutral-50 prevent-horizontal-scroll">
      {/* Step Indicator */}
      <div className="bg-white border-b border-neutral-200">
        <div className="container-responsive">
          <StepIndicator currentStep={1} completedSteps={[]} />
        </div>
      </div>

      {/* Main Content */}
      <div className="container-responsive py-4 sm:py-6 md:py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6 sm:space-y-8"
        >
          {/* Header */}
          <header className="text-center">
            <h1 
              className="text-3xl sm:text-4xl font-bold text-neutral-900 mb-3 sm:mb-4"
              tabIndex="-1"
              id="page-title"
            >
              Personalized Safety Check
            </h1>
            <p className="text-neutral-600 text-base sm:text-lg max-w-2xl mx-auto px-2">
              Start by adding the medications you want to check for interactions. 
              Search and select each medication from our comprehensive database.
            </p>
          </header>

          {/* Medication Search */}
          <Card className="p-4 sm:p-6" role="region" aria-labelledby="search-section">
            <div className="space-y-4">
              <div>
                <h2 id="search-section" className="sr-only">Medication Search</h2>
                <label htmlFor="medication-search" className="block text-sm font-medium text-neutral-900 mb-2">
                  Search for medications
                </label>
                <DrugSearch
                  id="medication-search"
                  onSelect={handleAddMedication}
                  placeholder="Type medication name (e.g., Lisinopril, Aspirin)..."
                  aria-describedby="search-help"
                />
              </div>
              
              <p id="search-help" className="text-sm text-neutral-500">
                💡 Tip: Start typing the medication name and select from the dropdown suggestions
              </p>
            </div>
          </Card>

          {/* Selected Medications */}
          <section aria-labelledby="selected-medications-heading">
            <h2 id="selected-medications-heading" className="text-lg sm:text-xl font-semibold text-neutral-900 px-2 sm:px-0 mb-4">
              Selected Medications ({selectedMedications.length})
            </h2>
            
            <div 
              role="region" 
              aria-live="polite" 
              aria-label={`${selectedMedications.length} medications selected`}
            >
              <AnimatePresence>
                {selectedMedications.length === 0 ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center py-8 sm:py-12 text-neutral-500"
                    role="status"
                    aria-label="No medications selected"
                  >
                    <div className="text-4xl sm:text-6xl mb-3 sm:mb-4" aria-hidden="true">💊</div>
                    <p className="text-base sm:text-lg">No medications selected yet</p>
                    <p className="text-sm">Use the search above to add medications</p>
                  </motion.div>
                ) : (
                  <div 
                    className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 px-2 sm:px-0"
                    role="list"
                    aria-label="Selected medications list"
                  >
                    {selectedMedications.map((medication, index) => (
                      <motion.div
                        key={medication}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        transition={{ delay: index * 0.1 }}
                        className="relative"
                        role="listitem"
                      >
                        <Card className="p-4 bg-gradient-to-br from-primary-50 to-secondary-50 border-primary-200 hover:shadow-lg transition-all">
                          <div className="flex items-center justify-between">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center space-x-2 mb-1">
                                <span className="text-lg" aria-hidden="true">💊</span>
                                <h3 className="font-semibold text-neutral-900 text-sm sm:text-base truncate">
                                  {medication}
                                </h3>
                              </div>
                              <p className="text-xs text-neutral-600">
                                Ready for dosage information
                              </p>
                            </div>
                            <button
                              onClick={() => handleRemoveMedication(medication)}
                              className="p-2 text-neutral-400 hover:text-danger-500 transition-colors min-h-[44px] min-w-[44px] flex-shrink-0 ml-2 focus:outline-none focus:ring-2 focus:ring-danger-500 focus:ring-offset-2 rounded-full"
                              aria-label={`Remove ${medication} from medication list`}
                            >
                              <X className="w-4 h-4" aria-hidden="true" />
                            </button>
                          </div>
                        </Card>
                      </motion.div>
                    ))}
                  </div>
                )}
              </AnimatePresence>
            </div>
          </section>

          {/* Helper Text */}
          {selectedMedications.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-primary-50 border border-primary-200 rounded-lg p-4 mx-2 sm:mx-0"
              role="status"
              aria-live="polite"
            >
              <div className="flex items-start space-x-3">
                <div className="text-primary-500 text-lg flex-shrink-0" aria-hidden="true">ℹ️</div>
                <div className="min-w-0">
                  <h4 className="font-medium text-primary-900 mb-1">
                    Great! You've added {selectedMedications.length} medication{selectedMedications.length > 1 ? 's' : ''}
                  </h4>
                  <p className="text-sm text-primary-700">
                    Next, you'll provide dosage information for each medication to get accurate safety analysis.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Navigation */}
          <nav className="flex flex-col sm:flex-row justify-between items-center gap-4 pt-6 border-t border-neutral-200 mx-2 sm:mx-0" aria-label="Step navigation">
            <Button
              onClick={() => navigate('/')}
              variant="secondary"
              className="w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-3 rounded-lg font-medium flex items-center justify-center space-x-2 transition-all duration-300 min-h-[44px] bg-white border border-neutral-300 text-neutral-700 hover:bg-neutral-50"
            >
              <ArrowLeft className="w-5 h-5" aria-hidden="true" />
              <span>Back to Home</span>
            </Button>
            
            <Button
              onClick={handleNext}
              disabled={!canProceed}
              className={`w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-3 rounded-lg font-medium flex items-center justify-center space-x-2 transition-all duration-300 min-h-[44px] ${
                canProceed
                  ? 'bg-primary-500 hover:bg-primary-600 text-white shadow-md hover:shadow-lg transform hover:-translate-y-0.5'
                  : '!bg-neutral-200 !border-neutral-200 !text-neutral-500 cursor-not-allowed'
              }`}
              aria-label={canProceed ? 'Proceed to dosage information step' : 'Add at least one medication to proceed'}
              aria-describedby={!canProceed ? 'next-button-help' : undefined}
            >
              <span>Next: Dosage Information</span>
              <motion.div
                animate={{ x: canProceed ? [0, 5, 0] : 0 }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <ArrowRight className="w-5 h-5" aria-hidden="true" />
              </motion.div>
            </Button>
            
            {!canProceed && (
              <div id="next-button-help" className="sr-only">
                You must add at least one medication before proceeding to the next step.
              </div>
            )}
          </nav>
        </motion.div>
      </div>
    </div>
  )
}

export default MedicationStep
