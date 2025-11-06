import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronUp, Info, BookOpen, ExternalLink } from 'lucide-react'
import Card from '../common/Card'

const ExplainPanel = ({ explanation, technicalDetails, sources, interactions }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState('plain') // 'plain' or 'technical'

  return (
    <Card shadow="soft-lg" className="overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-6 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-primary-50 rounded-lg flex items-center justify-center">
            <Info className="w-5 h-5 text-primary" />
          </div>
          <div className="text-left">
            <h3 className="text-lg font-heading font-bold text-neutral-text">
              Why this risk level?
            </h3>
            <p className="text-sm text-gray-500">
              Click to see detailed explanation
            </p>
          </div>
        </div>
        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.3 }}
        >
          <ChevronDown className="w-5 h-5 text-gray-400" />
        </motion.div>
      </button>

      {/* Expandable Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="border-t border-gray-100"
          >
            {/* Tabs */}
            <div className="flex border-b border-gray-100">
              <button
                onClick={() => setActiveTab('plain')}
                className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
                  activeTab === 'plain'
                    ? 'text-primary border-b-2 border-primary bg-primary-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-center space-x-2">
                  <Info className="w-4 h-4" />
                  <span>Plain Language</span>
                </div>
              </button>
              <button
                onClick={() => setActiveTab('technical')}
                className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
                  activeTab === 'technical'
                    ? 'text-primary border-b-2 border-primary bg-primary-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-center space-x-2">
                  <BookOpen className="w-4 h-4" />
                  <span>Technical Details</span>
                </div>
              </button>
            </div>

            {/* Tab Content */}
            <div className="p-6">
              <AnimatePresence mode="wait">
                {activeTab === 'plain' ? (
                  <motion.div
                    key="plain"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.2 }}
                  >
                    <div className="prose prose-sm max-w-none">
                      <p className="text-gray-700 leading-relaxed whitespace-pre-line">
                        {explanation}
                      </p>
                    </div>

                    {/* Interaction Details */}
                    {interactions && interactions.length > 0 && (
                      <div className="mt-6">
                        <h4 className="text-sm font-semibold text-neutral-text mb-3">
                          Specific Interactions Found:
                        </h4>
                        <div className="space-y-2">
                          {interactions.map((interaction, index) => (
                            <div
                              key={index}
                              className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                            >
                              <div className="flex items-center space-x-2">
                                <span className="text-sm font-medium text-gray-700">
                                  {interaction.drug1} + {interaction.drug2}
                                </span>
                              </div>
                              <span className={`text-sm font-bold ${
                                interaction.risk < 30 ? 'text-success' :
                                interaction.risk < 70 ? 'text-warning' :
                                'text-danger'
                              }`}>
                                {interaction.risk}% risk
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    key="technical"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.2 }}
                  >
                    <div className="space-y-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="text-sm font-semibold text-neutral-text mb-2">
                          Analysis Method
                        </h4>
                        <p className="text-sm text-gray-600">
                          {technicalDetails || 
                            'This analysis uses a Graph Neural Network (GNN) model trained on drug interaction databases. The model predicts interaction probability based on molecular structure and known pharmacological properties.'}
                        </p>
                      </div>

                      {sources && sources.length > 0 && (
                        <div>
                          <h4 className="text-sm font-semibold text-neutral-text mb-3">
                            Evidence Sources
                          </h4>
                          <div className="space-y-2">
                            {sources.map((source, index) => (
                              <a
                                key={index}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors group"
                              >
                                <span className="text-sm text-gray-700 group-hover:text-primary">
                                  {source.title}
                                </span>
                                <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-primary" />
                              </a>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                        <p className="text-xs text-amber-800">
                          <strong>Disclaimer:</strong> This analysis is for informational purposes only and should not replace professional medical advice. Always consult your healthcare provider before making medication decisions.
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  )
}

export default ExplainPanel
