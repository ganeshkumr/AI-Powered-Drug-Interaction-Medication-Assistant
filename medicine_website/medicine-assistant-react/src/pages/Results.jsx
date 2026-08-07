import { motion } from 'framer-motion'
import { useLocation, useNavigate } from 'react-router-dom'
import { ArrowLeft, Save, X, Activity } from 'lucide-react'
import { useState } from 'react'
import Layout from '../components/layout/Layout'
import RiskGauge from '../components/risk/RiskGauge'
import RiskBadge from '../components/risk/RiskBadge'
import ExplainPanel from '../components/risk/ExplainPanel'
import Button from '../components/common/Button'
import Card from '../components/common/Card'
import MedicationChip from '../components/medication/MedicationChip'
import { BASE_URL } from '../services/api'

const Results = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const { result, drugs } = location.state || {}
  const [saving, setSaving] = useState(false)

  if (!result || !drugs) {
    return (
      <Layout>
        <div className="text-center py-20">
          <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h2 className="text-2xl font-heading font-bold text-neutral-text mb-2">
            No Results Found
          </h2>
          <p className="text-gray-600 mb-6">
            Please perform a drug interaction check first.
          </p>
          <Button variant="primary" onClick={() => navigate('/')}>
            Go to Home
          </Button>
        </div>
      </Layout>
    )
  }

  const handleSave = async () => {
    const token = localStorage.getItem('token')
    if (!token) {
      navigate('/login', { state: { from: '/results', result, drugs } })
      return
    }

    setSaving(true)
    try {
      // Save medications to wallet
      for (const drug of drugs) {
        await fetch(`${BASE_URL}/api/medications`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            drug_name: drug,
            dosage_amount: 0,
            dosage_unit: 'mg',
            frequency: 'as needed'
          })
        })
      }
      navigate('/my-med')
    } catch (error) {
      console.error('Save error:', error)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        {/* Back Button */}
        <motion.button
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={() => navigate(-1)}
          className="flex items-center space-x-2 text-gray-600 hover:text-primary transition-colors mb-6"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back</span>
        </motion.button>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-heading font-bold text-neutral-text mb-3">
            Interaction Analysis Results
          </h1>
          <p className="text-gray-600">
            AI-powered analysis of your medication combination
          </p>
        </motion.div>

        {/* Medications Checked */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8"
        >
          <Card>
            <div className="p-6">
              <h3 className="text-sm font-semibold text-gray-600 mb-3">
                Medications Checked:
              </h3>
              <div className="flex flex-wrap gap-2">
                {drugs.map((drug) => (
                  <MedicationChip key={drug} drug={drug} />
                ))}
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Main Results Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Left Column - Risk Visualization */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card shadow="soft-lg">
              <div className="p-8 text-center">
                <h2 className="text-xl font-heading font-bold text-neutral-text mb-6">
                  Risk Assessment
                </h2>
                
                {/* Risk Gauge */}
                <div className="mb-6">
                  <RiskGauge risk={result.gnn_risk} size="xl" />
                </div>

                {/* Risk Badge */}
                <div className="flex justify-center mb-6">
                  <RiskBadge verdict={result.verdict} size="lg" />
                </div>

                {/* GNN Prediction Info */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">
                    <span className="font-semibold">Model Confidence:</span>{' '}
                    {Math.round((1 - Math.abs(result.gnn_risk - 0.5) * 2) * 100)}%
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Based on Graph Neural Network analysis of drug-drug interactions
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Right Column - Detailed Analysis */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card shadow="soft-lg">
              <div className="p-8">
                <h2 className="text-xl font-heading font-bold text-neutral-text mb-6">
                  Clinical Details
                </h2>
                
                <ExplainPanel explanation={result.explanation} />
              </div>
            </Card>
          </motion.div>
        </div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="flex flex-col sm:flex-row justify-center gap-4"
        >
          <Button
            variant="primary"
            icon={<Save className="w-5 h-5" />}
            onClick={handleSave}
            loading={saving}
          >
            Save to My Medication Wallet
          </Button>

          <Button
            variant="secondary"
            onClick={() => navigate('/')}
          >
            Check Another Combination
          </Button>
        </motion.div>
      </div>
    </Layout>
  )
}

export default Results
