import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Calendar, AlertCircle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import StepIndicator from '../components/navigation/StepIndicator'
import Button from '../components/common/Button'
import Card from '../components/common/Card'
import { manageFocusTransition, announceToScreenReader, generateId } from '../utils/accessibility'

const DosageStep = () => {
  const [medications, setMedications] = useState([])
  const [dosageData, setDosageData] = useState({})
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    // Get medications from previous step
    const selectedMedications = JSON.parse(sessionStorage.getItem('selectedMedications') || '[]')
    if (selectedMedications.length === 0) {
      // Redirect back to step 1 if no medications selected
      navigate('/check/medication')
      return
    }
    
    setMedications(selectedMedications)
    
    // Initialize dosage data for each medication
    const initialDosageData = {}
    selectedMedications.forEach(med => {
      initialDosageData[med] = {
        dosage_amount: '',
        dosage_unit: 'mg',
        frequency: '',
        start_date: '',
        end_date: ''
      }
    })
    setDosageData(initialDosageData)
    
    // Announce page load and focus management
    manageFocusTransition(2, 'forward')
    announceToScreenReader(`Dosage information step loaded. Please provide dosage details for ${selectedMedications.length} medications.`)
  }, [navigate])

  const handleDosageChange = (medication, field, value) => {
    setDosageData(prev => ({
      ...prev,
      [medication]: {
        ...prev[medication],
        [field]: value
      }
    }))
  }

  const validateForm = () => {
    const errors = {}
    medications.forEach(med => {
      const data = dosageData[med]
      if (!data || !data.dosage_amount || !data.frequency) {
        errors[med] = 'Dosage amount and frequency are required'
      }
      if (data && data.end_date && data.start_date && new Date(data.end_date) <= new Date(data.start_date)) {
        errors[med] = 'End date must be after start date'
      }
    })
    return errors
  }

  const handleBack = () => {
    announceToScreenReader('Returning to medication selection step')
    navigate('/check/medication')
  }

  const handleNext = async () => {
    const errors = validateForm()
    if (Object.keys(errors).length === 0) {
      setIsLoading(true)
      announceToScreenReader('Processing dosage information and proceeding to safety analysis')
      
      // Store dosage data for the next step
      sessionStorage.setItem('dosageData', JSON.stringify(dosageData))
      
      // Add a small delay to show loading state
      setTimeout(() => {
        navigate('/check/analysis')
      }, 500)
    } else {
      announceToScreenReader('Please correct the form errors before proceeding')
    }
  }

  // Check if all required fields are filled
  const canProceed = medications.every(med => {
    const data = dosageData[med]
    return data && data.dosage_amount && data.frequency
  })

  const formErrors = validateForm()

  return (
    <div className="min-h-screen bg-neutral-50 prevent-horizontal-scroll">
      {/* Step Indicator */}
      <div className="bg-white border-b border-neutral-200">
        <div className="container-responsive">
          <StepIndicator currentStep={2} completedSteps={[1]} />
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
              Dosage Information
            </h1>
            <p className="text-neutral-600 text-base sm:text-lg max-w-2xl mx-auto px-2">
              Provide dosage details for each medication to ensure accurate safety analysis. 
              All fields marked with * are required.
            </p>
          </header>

          {/* Dosage Forms */}
          <section aria-labelledby="dosage-forms-heading">
            <h2 id="dosage-forms-heading" className="sr-only">Dosage Information Forms</h2>
            <div className="space-y-4 sm:space-y-6">
              {medications.map((medication, index) => {
                const medicationId = generateId(`medication-${index}`)
                const errorId = generateId(`error-${index}`)
                
                return (
                  <motion.div
                    key={medication}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Card 
                      className="p-4 sm:p-6 mx-2 sm:mx-0"
                      role="region"
                      aria-labelledby={`${medicationId}-title`}
                    >
                      <div className="mb-4">
                        <h3 
                          id={`${medicationId}-title`}
                          className="text-lg sm:text-xl font-semibold text-neutral-900 flex items-center space-x-2"
                        >
                          <span className="text-xl sm:text-2xl" aria-hidden="true">💊</span>
                          <span className="truncate">{medication}</span>
                        </h3>
                      </div>

                      {/* Form validation error display */}
                      {formErrors[medication] && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          className="mb-4 p-3 bg-danger-50 border border-danger-200 rounded-lg flex items-start space-x-2"
                          role="alert"
                          id={errorId}
                          aria-live="polite"
                        >
                          <AlertCircle className="w-5 h-5 text-danger-500 mt-0.5 flex-shrink-0" aria-hidden="true" />
                          <p className="text-sm text-danger-700">{formErrors[medication]}</p>
                        </motion.div>
                      )}

                      <fieldset className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
                        <legend className="sr-only">Dosage information for {medication}</legend>
                        
                        {/* Dosage Amount */}
                        <div className="sm:col-span-1">
                          <label htmlFor={`${medicationId}-dosage-amount`} className="block text-sm font-medium text-neutral-900 mb-2">
                            Dosage Amount *
                          </label>
                          <input
                            id={`${medicationId}-dosage-amount`}
                            type="number"
                            min="0"
                            step="0.1"
                            value={dosageData[medication]?.dosage_amount || ''}
                            onChange={(e) => handleDosageChange(medication, 'dosage_amount', e.target.value)}
                            placeholder="e.g., 10"
                            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all min-h-[44px] medical-input-enhanced ${
                              formErrors[medication] ? 'border-danger-300' : 'border-neutral-300'
                            }`}
                            required
                            aria-describedby={formErrors[medication] ? errorId : `${medicationId}-dosage-help`}
                            aria-invalid={formErrors[medication] ? 'true' : 'false'}
                          />
                          <div id={`${medicationId}-dosage-help`} className="sr-only">
                            Enter the amount of medication taken each time
                          </div>
                        </div>

                        {/* Unit */}
                        <div className="sm:col-span-1">
                          <label htmlFor={`${medicationId}-unit`} className="block text-sm font-medium text-neutral-900 mb-2">
                            Unit *
                          </label>
                          <select
                            id={`${medicationId}-unit`}
                            value={dosageData[medication]?.dosage_unit || 'mg'}
                            onChange={(e) => handleDosageChange(medication, 'dosage_unit', e.target.value)}
                            className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all min-h-[44px] medical-input-enhanced"
                          >
                            <option value="mg">mg (milligrams)</option>
                            <option value="g">g (grams)</option>
                            <option value="ml">ml (milliliters)</option>
                            <option value="mcg">mcg (micrograms)</option>
                            <option value="IU">IU (International Units)</option>
                            <option value="tablets">tablets</option>
                            <option value="capsules">capsules</option>
                          </select>
                        </div>

                        {/* Frequency */}
                        <div className="sm:col-span-2 lg:col-span-1">
                          <label htmlFor={`${medicationId}-frequency`} className="block text-sm font-medium text-neutral-900 mb-2">
                            Frequency *
                          </label>
                          <select
                            id={`${medicationId}-frequency`}
                            value={dosageData[medication]?.frequency || ''}
                            onChange={(e) => handleDosageChange(medication, 'frequency', e.target.value)}
                            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all min-h-[44px] medical-input-enhanced ${
                              formErrors[medication] ? 'border-danger-300' : 'border-neutral-300'
                            }`}
                            required
                            aria-describedby={formErrors[medication] ? errorId : undefined}
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
                      </fieldset>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {/* Start Date */}
                        <div>
                          <label htmlFor={`${medicationId}-start-date`} className="block text-sm font-medium text-neutral-900 mb-2 flex items-center space-x-2">
                            <Calendar className="w-4 h-4" />
                            <span>Start Date</span>
                          </label>
                          <input
                            id={`${medicationId}-start-date`}
                            type="date"
                            value={dosageData[medication]?.start_date || ''}
                            onChange={(e) => handleDosageChange(medication, 'start_date', e.target.value)}
                            max={new Date().toISOString().split('T')[0]}
                            className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all min-h-[44px] medical-input-enhanced"
                          />
                          <p className="text-xs text-neutral-500 mt-1">
                            When did you start taking this medication?
                          </p>
                        </div>

                        {/* End Date */}
                        <div>
                          <label htmlFor={`${medicationId}-end-date`} className="block text-sm font-medium text-neutral-900 mb-2 flex items-center space-x-2">
                            <Calendar className="w-4 h-4" />
                            <span>End Date (Optional)</span>
                          </label>
                          <input
                            id={`${medicationId}-end-date`}
                            type="date"
                            value={dosageData[medication]?.end_date || ''}
                            onChange={(e) => handleDosageChange(medication, 'end_date', e.target.value)}
                            min={dosageData[medication]?.start_date || undefined}
                            className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all min-h-[44px] medical-input-enhanced"
                          />
                          <p className="text-xs text-neutral-500 mt-1">
                            Leave empty for ongoing medication
                          </p>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                )
              })}
            </div>
          </section>

          {/* Progress Indicator */}
          {medications.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-primary-50 border border-primary-200 rounded-lg p-4 mx-2 sm:mx-0"
            >
              <div className="flex items-start space-x-3">
                <div className="text-primary-500 text-lg flex-shrink-0">📋</div>
                <div className="min-w-0">
                  <h4 className="font-medium text-primary-900 mb-1">
                    Progress: {medications.filter(med => {
                      const data = dosageData[med]
                      return data && data.dosage_amount && data.frequency
                    }).length} of {medications.length} medications completed
                  </h4>
                  <p className="text-sm text-primary-700">
                    Fill in the required dosage information for all medications to proceed to safety analysis.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Navigation */}
          <nav className="flex flex-col sm:flex-row justify-between items-center gap-4 pt-6 border-t border-neutral-200 mx-2 sm:mx-0" aria-label="Step navigation">
            <Button
              onClick={handleBack}
              variant="secondary"
              className="w-full sm:w-auto px-6 sm:px-8 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 min-h-[44px] bg-white border border-neutral-300 text-neutral-700 hover:bg-neutral-50"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back: Medications</span>
            </Button>
            
            <Button
              onClick={handleNext}
              disabled={!canProceed || isLoading}
              variant="primary"
              className={`w-full sm:w-auto px-6 sm:px-8 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 min-h-[44px] transition-all duration-300 ${
                canProceed && !isLoading
                  ? 'bg-primary-500 hover:bg-primary-600 text-white shadow-md hover:shadow-lg transform hover:-translate-y-0.5'
                  : 'bg-neutral-200 text-neutral-400 cursor-not-allowed'
              }`}
              aria-label={canProceed ? 'Proceed to safety analysis step' : 'Complete all required fields to proceed'}
            >
              {isLoading ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                  />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span>Check Safety</span>
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </Button>
          </nav>
        </motion.div>
      </div>
    </div>
  )
}

export default DosageStep