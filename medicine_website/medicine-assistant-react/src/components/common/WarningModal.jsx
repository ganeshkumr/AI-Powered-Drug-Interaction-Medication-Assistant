import React from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, X, Shield, Info, ChevronRight } from 'lucide-react'
import Modal from './Modal'
import Button from './Button'
import RiskBadge from '../risk/RiskBadge'
import { announceToScreenReader } from '../../utils/accessibility'

const WarningModal = ({
  isOpen,
  onClose,
  onProceed,
  riskData,
  medications = [],
  title = "Medication Safety Warning",
  showProceedButton = true,
  proceedButtonText = "I Understand, Continue",
  className = ""
}) => {
  const handleClose = () => {
    announceToScreenReader('Safety warning dismissed', 'polite')
    onClose()
  }

  const handleProceed = () => {
    announceToScreenReader('User acknowledged safety warning and chose to proceed', 'polite')
    if (onProceed) {
      onProceed()
    }
    onClose()
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

  const formatAIResponse = (response) => {
    if (!response) return ''
    
    // Convert newlines to HTML breaks and format for better readability
    return response
      .replace(/\n/g, '<br>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
  }

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      size="lg"
      closeOnBackdropClick={false}
      closeOnEscape={true}
      className={className}
      aria-labelledby="warning-modal-title"
      aria-describedby="warning-modal-content"
    >
      <div className="space-y-6">
        {/* Header Section */}
        <div className="text-center">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.3 }}
            className="mx-auto w-16 h-16 bg-warning-100 rounded-full flex items-center justify-center mb-4"
          >
            <AlertTriangle className="w-8 h-8 text-warning-600" aria-hidden="true" />
          </motion.div>
          
          <h2 
            id="warning-modal-title"
            className="text-2xl font-bold text-neutral-900 mb-2"
          >
            {title}
          </h2>
          
          <p className="text-neutral-600 text-base">
            Our AI analysis has identified potential safety concerns with your medication combination.
          </p>
        </div>

        {/* Risk Assessment Section */}
        {riskData && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-gradient-to-r from-warning-50 to-danger-50 border border-warning-200 rounded-lg p-6"
          >
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
              <div className="flex items-center space-x-3 mb-3 sm:mb-0">
                <Shield className="w-6 h-6 text-warning-600" aria-hidden="true" />
                <h3 className="text-lg font-semibold text-neutral-900">
                  Risk Assessment
                </h3>
              </div>
              
              {riskData.overallRisk !== undefined && (
                <RiskBadge 
                  riskLevel={getRiskLevel(riskData.overallRisk)} 
                  size="medium"
                />
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              {riskData.overallRisk !== undefined && (
                <div className="bg-white rounded-lg p-4 border border-warning-200">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className={`w-3 h-3 rounded-full bg-${getRiskColor(riskData.overallRisk)}-500`}></div>
                    <span className="text-sm font-medium text-neutral-700">Risk Score</span>
                  </div>
                  <p className="text-2xl font-bold text-neutral-900">{riskData.overallRisk}%</p>
                </div>
              )}
              
              {riskData.overallVerdict && (
                <div className="bg-white rounded-lg p-4 border border-warning-200">
                  <div className="flex items-center space-x-2 mb-2">
                    <Info className="w-4 h-4 text-neutral-500" aria-hidden="true" />
                    <span className="text-sm font-medium text-neutral-700">Assessment</span>
                  </div>
                  <p className="text-lg font-semibold text-neutral-900">{riskData.overallVerdict}</p>
                </div>
              )}
            </div>

            {medications.length > 0 && (
              <div className="bg-white rounded-lg p-4 border border-warning-200">
                <h4 className="text-sm font-medium text-neutral-700 mb-2">Medications Analyzed:</h4>
                <div className="flex flex-wrap gap-2">
                  {medications.map((med, index) => (
                    <span 
                      key={index}
                      className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-neutral-100 text-neutral-700"
                    >
                      {med}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* AI Analysis Section */}
        {riskData?.results && riskData.results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            id="warning-modal-content"
          >
            <div className="bg-primary-50 border border-primary-200 rounded-lg p-6">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-primary-500 rounded-full flex items-center justify-center">
                  <span className="text-xl text-white">🤖</span>
                </div>
                <h3 className="text-lg font-semibold text-neutral-900">
                  AI Safety Analysis
                </h3>
              </div>

              <div className="space-y-4 max-h-64 overflow-y-auto">
                {riskData.results.map((result, index) => (
                  <div 
                    key={result.medication || index}
                    className="bg-white rounded-lg p-4 border border-primary-200"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-neutral-900 flex items-center space-x-2">
                        <span className="text-lg">💊</span>
                        <span>{result.medication}</span>
                      </h4>
                      {result.gnn_risk !== undefined && (
                        <div className="flex items-center space-x-2">
                          <div className={`w-2 h-2 rounded-full bg-${getRiskColor(result.gnn_risk)}-500`}></div>
                          <span className="text-sm font-medium text-neutral-600">
                            {result.gnn_risk}%
                          </span>
                        </div>
                      )}
                    </div>
                    
                    {result.ai_response && (
                      <div 
                        className="text-sm text-neutral-700 leading-relaxed prose prose-sm max-w-none"
                        dangerouslySetInnerHTML={{ 
                          __html: formatAIResponse(result.ai_response)
                        }}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* Important Notice */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-neutral-50 border border-neutral-200 rounded-lg p-4"
        >
          <div className="flex items-start space-x-3">
            <Info className="w-5 h-5 text-neutral-500 mt-0.5 flex-shrink-0" aria-hidden="true" />
            <div className="text-sm text-neutral-700">
              <p className="font-medium mb-1">Important Notice:</p>
              <p>
                This analysis is for informational purposes only and should not replace professional medical advice. 
                Please consult with your healthcare provider or pharmacist before making any changes to your medication regimen.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="flex flex-col sm:flex-row gap-3 pt-4 border-t border-neutral-200"
        >
          <Button
            onClick={handleClose}
            variant="secondary"
            className="flex-1 sm:flex-none px-6 py-3 min-h-[44px]"
          >
            Review Information
          </Button>
          
          {showProceedButton && (
            <Button
              onClick={handleProceed}
              variant="warning"
              className="flex-1 sm:flex-none px-6 py-3 min-h-[44px] flex items-center justify-center space-x-2"
            >
              <span>{proceedButtonText}</span>
              <ChevronRight className="w-4 h-4" aria-hidden="true" />
            </Button>
          )}
        </motion.div>
      </div>
    </Modal>
  )
}

export default WarningModal