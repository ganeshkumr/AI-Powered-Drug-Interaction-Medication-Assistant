import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, Save, MessageCircle, CheckCircle, AlertCircle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import StepIndicator from '../components/navigation/StepIndicator'
import RiskBadge from '../components/risk/RiskBadge'
import Button from '../components/common/Button'
import Card from '../components/common/Card'
import AnalysisLoader from '../components/common/AnalysisLoader'
import WarningModal from '../components/common/WarningModal'
import { medicationAPI } from '../services/api'
import { manageFocusTransition, announceToScreenReader } from '../utils/accessibility'

const AnalysisStep = () => {
  const [medications, setMedications] = useState([])
  const [dosageData, setDosageData] = useState({})
  const [analysisResult, setAnalysisResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [showWarningModal, setShowWarningModal] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    // Get data from previous steps
    const selectedMedications = JSON.parse(sessionStorage.getItem('selectedMedications') || '[]')
    const storedDosageData = JSON.parse(sessionStorage.getItem('dosageData') || '{}')
    
    if (selectedMedications.length === 0 || Object.keys(storedDosageData).length === 0) {
      // Redirect back to step 1 if no data
      navigate('/check/medication')
      return
    }
    
    setMedications(selectedMedications)
    setDosageData(storedDosageData)
    
    // Announce page load and focus management
    manageFocusTransition(3, 'forward')
    announceToScreenReader('Safety analysis step loaded. Analyzing your medications for interactions.')
    
    // Perform safety analysis
    performSafetyAnalysis(selectedMedications, storedDosageData)
  }, [navigate])

  const performSafetyAnalysis = async (meds, dosages) => {
    setLoading(true)
    try {
      // For multiple medications, we'll check each one and combine results
      const results = []
      
      for (const med of meds) {
        const dosage = dosages[med]
        if (dosage && dosage.dosage_amount && dosage.frequency) {
          try {
            const response = await medicationAPI.checkBeforeAdding({
              drug_name: med,
              dosage_amount: dosage.dosage_amount,
              dosage_unit: dosage.dosage_unit,
              frequency: dosage.frequency
            })
            results.push({
              medication: med,
              ...response.data
            })
          } catch (error) {
            console.error(`Analysis failed for ${med}:`, error)
            results.push({
              medication: med,
              verdict: 'ERROR',
              ai_response: `Failed to analyze ${med}. Please try again.`,
              gnn_risk: 0
            })
          }
        }
      }
      
      // Combine results into overall analysis
      const overallRisk = Math.max(...results.map(r => r.gnn_risk || 0))
      const hasUnsafe = results.some(r => r.verdict?.includes('UNSAFE'))
      const hasError = results.some(r => r.verdict === 'ERROR')
      
      let overallVerdict = 'SAFE'
      if (hasError) {
        overallVerdict = 'ERROR'
      } else if (hasUnsafe || overallRisk > 70) {
        overallVerdict = 'HIGH RISK'
      } else if (overallRisk > 40) {
        overallVerdict = 'CAUTION'
      }
      
      const analysisData = {
        overallRisk,
        overallVerdict,
        results,
        canSave: !hasError && !hasUnsafe
      }
      
      setAnalysisResult(analysisData)
      
      // Show warning modal for unsafe combinations (non-blocking)
      if (hasUnsafe || overallRisk > 70) {
        setShowWarningModal(true)
        announceToScreenReader(`High risk medication combination detected. Warning modal opened. Risk score: ${overallRisk}%.`)
      }
      
      // Announce results to screen reader
      announceToScreenReader(`Analysis complete. Overall risk assessment: ${overallVerdict}. Risk score: ${overallRisk}%.`)
      
    } catch (error) {
      console.error('Safety analysis failed:', error)
      setAnalysisResult({
        overallRisk: 0,
        overallVerdict: 'ERROR',
        results: [],
        canSave: false,
        error: 'Failed to perform safety analysis. Please try again.'
      })
      announceToScreenReader('Analysis failed. Please try again or contact support.')
    } finally {
      setLoading(false)
    }
  }

  const handleBack = () => {
    announceToScreenReader('Returning to dosage information step')
    navigate('/check/dosage')
  }

  const handleSaveMedications = async () => {
    if (!analysisResult?.canSave) return
    
    setSaving(true)
    announceToScreenReader('Saving medications to your profile')
    
    try {
      // Save each medication
      for (const med of medications) {
        const dosage = dosageData[med]
        if (dosage && dosage.dosage_amount && dosage.frequency) {
          await medicationAPI.addMedication({
            drug_name: med,
            dosage_amount: dosage.dosage_amount,
            dosage_unit: dosage.dosage_unit,
            frequency: dosage.frequency,
            start_date: dosage.start_date,
            end_date: dosage.end_date
          })
        }
      }
      
      // Clear session storage
      sessionStorage.removeItem('selectedMedications')
      sessionStorage.removeItem('dosageData')
      
      announceToScreenReader('Medications saved successfully. Redirecting to My Medications.')
      
      // Navigate to My Medications
      navigate('/my-med')
    } catch (error) {
      console.error('Failed to save medications:', error)
      announceToScreenReader('Failed to save medications. Please try again.')
      alert('Failed to save medications. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  const getRiskLevel = (risk) => {
    if (risk > 70) return 'high-risk'
    if (risk > 40) return 'caution'
    return 'safe'
  }

  const getRiskColor = (risk) => {
    if (risk > 70) return 'danger'
    if (risk > 40) return 'warning'
    return 'success'
  }

  if (loading) {
    return (
      <AnalysisLoader
        message="Analyzing Your Medications"
        submessage="Please wait while we check for interactions and safety concerns..."
        medicationCount={medications.length}
      />
    )
  }

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Step Indicator */}
      <div className="bg-white border-b border-neutral-200">
        <div className="container-responsive">
          <StepIndicator currentStep={3} completedSteps={[1, 2]} />
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
              Safety Analysis Results
            </h1>
            <p className="text-neutral-600 text-base sm:text-lg max-w-2xl mx-auto px-2">
              Here's your personalized medication safety analysis based on the information you provided.
            </p>
          </header>

          {analysisResult && (
            <>
              {/* Overall Risk Assessment */}
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="text-center"
              >
                <Card className="p-6 sm:p-8 bg-white shadow-lg border border-neutral-200">
                  <div className="mb-6">
                    <RiskBadge 
                      riskLevel={getRiskLevel(analysisResult.overallRisk)} 
                      size="large"
                      className="mx-auto"
                    />
                  </div>
                  
                  <h2 className="text-2xl sm:text-3xl font-bold text-neutral-900 mb-2">
                    {analysisResult.overallVerdict}
                  </h2>
                  
                  <p className="text-lg text-neutral-600 mb-4">
                    Overall Safety Assessment
                  </p>
                  
                  <div className="flex flex-wrap items-center justify-center gap-4 text-sm text-neutral-500">
                    <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full bg-${getRiskColor(analysisResult.overallRisk)}-500`}></div>
                      <span>Risk Score: {analysisResult.overallRisk}%</span>
                    </div>
                    <span className="hidden sm:inline">•</span>
                    <span>{medications.length} Medication{medications.length > 1 ? 's' : ''} Analyzed</span>
                  </div>
                </Card>
              </motion.div>

              {/* AI Analysis Results */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Card className="p-6 bg-gradient-to-r from-primary-50 to-secondary-50 border-l-4 border-primary-500">
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0">
                      <div className="w-12 h-12 bg-primary-500 rounded-full flex items-center justify-center">
                        <span className="text-2xl text-white">🤖</span>
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
                        <h3 className="text-xl font-semibold text-neutral-900">
                          AI Safety Analysis
                        </h3>
                        <span className="text-sm font-medium text-primary-600 bg-primary-100 px-3 py-1 rounded-full mt-2 sm:mt-0">
                          Personalized Results
                        </span>
                      </div>
                      
                      {analysisResult.error ? (
                        <div className="bg-danger-50 border border-danger-200 rounded-lg p-4">
                          <div className="flex items-center mb-2">
                            <AlertCircle className="w-5 h-5 text-danger-600 mr-2" />
                            <p className="text-danger-800 font-medium">Analysis Error</p>
                          </div>
                          <p className="text-danger-700">{analysisResult.error}</p>
                        </div>
                      ) : (
                        <div className="space-y-6">
                          {analysisResult.results.map((result, index) => (
                            <motion.div 
                              key={result.medication}
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: 0.1 * index }}
                              className="bg-white rounded-lg p-4 border-l-4 border-primary-500 shadow-sm"
                            >
                              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-3">
                                <h4 className="font-semibold text-lg text-neutral-900 flex items-center space-x-2">
                                  <span className="text-xl">💊</span>
                                  <span>{result.medication}</span>
                                </h4>
                                {result.gnn_risk !== undefined && (
                                  <div className="flex items-center space-x-2 mt-2 sm:mt-0">
                                    <div className={`w-2 h-2 rounded-full bg-${getRiskColor(result.gnn_risk)}-500`}></div>
                                    <span className="text-sm font-medium px-3 py-1 bg-neutral-100 rounded-full text-neutral-700">
                                      Risk Score: {result.gnn_risk}%
                                    </span>
                                  </div>
                                )}
                              </div>
                              <div 
                                className="text-neutral-700 leading-relaxed prose prose-sm max-w-none"
                                dangerouslySetInnerHTML={{ 
                                  __html: result.ai_response
                                    ?.replace(/\*\*/g, '')
                                    ?.replace(/\n/g, '<br>')
                                }}
                              />
                            </motion.div>
                          ))}
                          
                          {analysisResult.results.length > 1 && (
                            <motion.div 
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.6 }}
                              className="bg-primary-50 border border-primary-200 rounded-lg p-4"
                            >
                              <div className="flex items-center mb-2">
                                <CheckCircle className="w-5 h-5 text-primary-600 mr-2" />
                                <h4 className="font-semibold text-primary-900">Combined Analysis Summary</h4>
                              </div>
                              <p className="text-primary-800">
                                We've analyzed all {analysisResult.results.length} medications together to check for interactions and combined effects. 
                                The overall risk assessment considers both individual medication safety and potential interactions between them.
                              </p>
                            </motion.div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              </motion.div>

              {/* Medication Summary */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                <Card className="p-6">
                  <h3 className="text-xl font-semibold text-neutral-900 mb-4 flex items-center space-x-2">
                    <span className="text-2xl">📋</span>
                    <span>Medication Summary</span>
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {medications.map((med) => {
                      const dosage = dosageData[med]
                      return (
                        <div key={med} className="bg-neutral-50 rounded-lg p-4 border border-neutral-200">
                          <h4 className="font-semibold text-neutral-900 mb-1">{med}</h4>
                          <p className="text-sm text-neutral-600 mb-1">
                            {dosage.dosage_amount} {dosage.dosage_unit} - {dosage.frequency}
                          </p>
                          {dosage.start_date && (
                            <p className="text-xs text-neutral-500">
                              Started: {new Date(dosage.start_date).toLocaleDateString()}
                            </p>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </Card>
              </motion.div>

              {/* Action Buttons */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 }}
                className="flex flex-col sm:flex-row gap-4 justify-center"
              >
                {analysisResult.canSave && (
                  <Button
                    onClick={handleSaveMedications}
                    disabled={saving}
                    className={`px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 min-h-[44px] ${
                      saving 
                        ? 'bg-neutral-200 text-neutral-400 cursor-not-allowed'
                        : 'bg-success-600 hover:bg-success-700 text-white shadow-md hover:shadow-lg'
                    }`}
                    aria-label={saving ? 'Saving medications...' : 'Save medications to your profile'}
                  >
                    {saving ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Saving...</span>
                      </>
                    ) : (
                      <>
                        <Save className="w-5 h-5" />
                        <span>Save to My Medications</span>
                      </>
                    )}
                  </Button>
                )}
                
                <Button
                  onClick={() => {
                    // TODO: Integrate with existing chatbot functionality
                    announceToScreenReader('Opening AI assistant for follow-up questions')
                    console.log('Opening chatbot for follow-up questions')
                  }}
                  variant="primary"
                  className="px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 min-h-[44px] shadow-sm hover:shadow-md"
                >
                  <MessageCircle className="w-5 h-5" />
                  <span>Ask AI Assistant</span>
                </Button>
              </motion.div>
            </>
          )}

          {/* Navigation */}
          <motion.nav
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.0 }}
            className="flex flex-col sm:flex-row justify-between items-center gap-4 pt-6 border-t border-neutral-200"
            aria-label="Step navigation"
          >
            <Button
              onClick={handleBack}
              variant="secondary"
              className="w-full sm:w-auto px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 min-h-[44px]"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back: Dosage</span>
            </Button>
            
            <Button
              onClick={() => {
                announceToScreenReader('Starting new medication safety check')
                navigate('/check/medication')
              }}
              variant="primary"
              className="w-full sm:w-auto px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 min-h-[44px]"
            >
              <span>Start New Check</span>
            </Button>
          </motion.nav>
        </motion.div>
      </div>

      {/* Warning Modal for Unsafe Combinations */}
      <WarningModal
        isOpen={showWarningModal}
        onClose={() => setShowWarningModal(false)}
        onProceed={() => {
          // User acknowledges the warning and can continue
          announceToScreenReader('User acknowledged safety warning. Analysis results remain available.')
        }}
        riskData={analysisResult}
        medications={medications}
        title="Medication Safety Warning"
        showProceedButton={true}
        proceedButtonText="I Understand, Continue"
      />
    </div>
  )
}

export default AnalysisStep
