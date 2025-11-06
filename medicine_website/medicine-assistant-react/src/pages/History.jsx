import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Clock, Filter, Search, Calendar, Eye, TrendingUp, AlertCircle } from 'lucide-react'
import Layout from '../components/layout/Layout'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import RiskBadge from '../components/risk/RiskBadge'

const History = () => {
  const navigate = useNavigate()
  const [checks, setChecks] = useState([])
  const [loading, setLoading] = useState(true)
  const [filterRisk, setFilterRisk] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [dateFilter, setDateFilter] = useState('all')

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    try {
      // Mock data for now - in real app, fetch from API
      const mockHistory = [
        {
          id: 1,
          date: new Date('2024-11-03'),
          drugs: ['Aspirin', 'Ibuprofen'],
          risk: 45,
          verdict: 'CAUTION ADVISED',
          summary: 'Moderate interaction risk detected between NSAIDs'
        },
        {
          id: 2,
          date: new Date('2024-11-02'),
          drugs: ['Lisinopril', 'Metformin'],
          risk: 15,
          verdict: 'SAFE TO ADD',
          summary: 'No significant interactions found'
        },
        {
          id: 3,
          date: new Date('2024-11-01'),
          drugs: ['Warfarin', 'Aspirin', 'Vitamin K'],
          risk: 85,
          verdict: 'DO NOT ADD',
          summary: 'High risk of bleeding complications'
        },
        {
          id: 4,
          date: new Date('2024-10-30'),
          drugs: ['Metformin', 'Atorvastatin'],
          risk: 20,
          verdict: 'SAFE TO ADD',
          summary: 'Commonly prescribed together safely'
        }
      ]
      setChecks(mockHistory)
    } catch (error) {
      console.error('Failed to fetch history:', error)
    } finally {
      setLoading(false)
    }
  }

  const filteredChecks = checks.filter(check => {
    // Risk filter
    if (filterRisk !== 'all') {
      if (filterRisk === 'safe' && check.risk >= 30) return false
      if (filterRisk === 'caution' && (check.risk < 30 || check.risk >= 70)) return false
      if (filterRisk === 'danger' && check.risk < 70) return false
    }

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      const drugsMatch = check.drugs.some(drug => drug.toLowerCase().includes(query))
      if (!drugsMatch) return false
    }

    // Date filter
    if (dateFilter !== 'all') {
      const checkDate = new Date(check.date)
      const now = new Date()
      const daysDiff = Math.floor((now - checkDate) / (1000 * 60 * 60 * 24))
      
      if (dateFilter === 'today' && daysDiff > 0) return false
      if (dateFilter === 'week' && daysDiff > 7) return false
      if (dateFilter === 'month' && daysDiff > 30) return false
    }

    return true
  })

  const getRiskColor = (risk) => {
    if (risk < 30) return 'text-success'
    if (risk < 70) return 'text-warning'
    return 'text-danger'
  }

  return (
    <Layout>
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-heading font-bold text-neutral-text mb-3">
            Check History
          </h1>
          <p className="text-gray-600">
            View and manage your past medication interaction checks
          </p>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"
        >
          <Card>
            <div className="p-4 flex items-center space-x-4">
              <div className="w-12 h-12 bg-primary-50 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold font-heading text-neutral-text">
                  {checks.length}
                </p>
                <p className="text-sm text-gray-600">Total Checks</p>
              </div>
            </div>
          </Card>

          <Card>
            <div className="p-4 flex items-center space-x-4">
              <div className="w-12 h-12 bg-success-50 rounded-lg flex items-center justify-center">
                <Clock className="w-6 h-6 text-success" />
              </div>
              <div>
                <p className="text-2xl font-bold font-heading text-neutral-text">
                  {checks.filter(c => c.risk < 30).length}
                </p>
                <p className="text-sm text-gray-600">Safe Results</p>
              </div>
            </div>
          </Card>

          <Card>
            <div className="p-4 flex items-center space-x-4">
              <div className="w-12 h-12 bg-danger-50 rounded-lg flex items-center justify-center">
                <AlertCircle className="w-6 h-6 text-danger" />
              </div>
              <div>
                <p className="text-2xl font-bold font-heading text-neutral-text">
                  {checks.filter(c => c.risk >= 70).length}
                </p>
                <p className="text-sm text-gray-600">High Risk</p>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="mb-6">
            <div className="p-4">
              <div className="flex flex-col md:flex-row md:items-center md:space-x-4 space-y-4 md:space-y-0">
                {/* Search */}
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search by drug name..."
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                  />
                </div>

                {/* Risk Filter */}
                <div className="flex items-center space-x-2">
                  <Filter className="w-5 h-5 text-gray-400" />
                  <select
                    value={filterRisk}
                    onChange={(e) => setFilterRisk(e.target.value)}
                    className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                  >
                    <option value="all">All Risk Levels</option>
                    <option value="safe">Safe</option>
                    <option value="caution">Caution</option>
                    <option value="danger">Dangerous</option>
                  </select>
                </div>

                {/* Date Filter */}
                <div className="flex items-center space-x-2">
                  <Calendar className="w-5 h-5 text-gray-400" />
                  <select
                    value={dateFilter}
                    onChange={(e) => setDateFilter(e.target.value)}
                    className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                  >
                    <option value="all">All Time</option>
                    <option value="today">Today</option>
                    <option value="week">This Week</option>
                    <option value="month">This Month</option>
                  </select>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* History Timeline */}
        {loading ? (
          <Card>
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              <p className="ml-3 text-gray-600">Loading history...</p>
            </div>
          </Card>
        ) : filteredChecks.length === 0 ? (
          <Card>
            <div className="text-center py-12">
              <Clock className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-xl font-heading font-bold text-neutral-text mb-2">
                No checks found
              </h3>
              <p className="text-gray-600 mb-6">
                {searchQuery || filterRisk !== 'all' || dateFilter !== 'all'
                  ? 'Try adjusting your filters'
                  : 'Start checking medications to see your history here'}
              </p>
              <Button variant="primary" onClick={() => navigate('/')}>
                Check Medications
              </Button>
            </div>
          </Card>
        ) : (
          <div className="space-y-4">
            <AnimatePresence>
              {filteredChecks.map((check, index) => (
                <motion.div
                  key={check.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className="hover:shadow-soft-lg transition-shadow">
                    <div className="p-6">
                      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
                        {/* Left: Date and Drugs */}
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-3">
                            <div className="w-10 h-10 bg-primary-50 rounded-lg flex items-center justify-center">
                              <Clock className="w-5 h-5 text-primary" />
                            </div>
                            <div>
                              <p className="text-sm text-gray-500">
                                {check.date.toLocaleDateString('en-US', {
                                  weekday: 'short',
                                  year: 'numeric',
                                  month: 'short',
                                  day: 'numeric'
                                })}
                              </p>
                              <p className="text-xs text-gray-400">
                                {check.date.toLocaleTimeString('en-US', {
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </p>
                            </div>
                          </div>

                          <div className="mb-2">
                            <p className="text-sm font-medium text-gray-600 mb-1">
                              Medications checked:
                            </p>
                            <div className="flex flex-wrap gap-2">
                              {check.drugs.map((drug) => (
                                <span
                                  key={drug}
                                  className="inline-flex items-center px-3 py-1 bg-primary-50 text-primary text-sm font-medium rounded-full"
                                >
                                  {drug}
                                </span>
                              ))}
                            </div>
                          </div>

                          <p className="text-sm text-gray-600">{check.summary}</p>
                        </div>

                        {/* Right: Risk and Actions */}
                        <div className="flex flex-col items-end space-y-3">
                          <div className="text-center">
                            <p className={`text-3xl font-bold font-heading ${getRiskColor(check.risk)}`}>
                              {check.risk}%
                            </p>
                            <p className="text-xs text-gray-500">Risk Score</p>
                          </div>

                          <RiskBadge verdict={check.verdict} size="sm" />

                          <Button
                            variant="secondary"
                            size="sm"
                            icon={<Eye className="w-4 h-4" />}
                            onClick={() => navigate('/results', {
                              state: {
                                result: {
                                  gnn_risk: check.risk,
                                  verdict: check.verdict,
                                  ai_response: check.summary,
                                  can_add: check.risk < 70
                                },
                                drugs: check.drugs
                              }
                            })}
                          >
                            View Full Report
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </Layout>
  )
}

export default History
