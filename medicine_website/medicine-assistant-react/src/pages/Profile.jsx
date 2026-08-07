import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { User, Save, Loader, ChevronDown, ChevronUp, X, Plus } from 'lucide-react'
import api from '../services/api'
import Button from '../components/common/Button'

const MEDICAL_CONDITION_OPTIONS = [
  'Diabetes',
  'Hypertension',
  'Asthma',
  'Heart Disease',
  'Kidney Disease',
  'Liver Disease',
  'Thyroid Disorder',
  'Arthritis',
  'Depression',
  'Anxiety',
  'Migraine',
  'Epilepsy',
  'COPD',
  'GERD',
  'Anemia',
  'High Cholesterol',
  'PCOS'
]

const Profile = () => {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [conditionInput, setConditionInput] = useState('')
  const [conditionsList, setConditionsList] = useState([])
  const [profile, setProfile] = useState({
    name: '',
    email: '',
    dob: '',
    gender: '',
    weight_kg: '',
    height_cm: '',
    emergency_contact: '',
    conditions: '',
    drug_allergies: '',
    food_allergies: '',
    other_allergies: '',
    is_smoker: '',
    alcohol_consumption: ''
  })

  useEffect(() => {
    fetchProfile()
  }, [])

  const fetchProfile = async () => {
    try {
      const response = await api.get('/api/profile')
      const profileData = response?.data?.profile || {}
      setProfile(prev => ({ ...prev, ...profileData }))
      
      // Parse conditions into chips
      if (profileData.conditions) {
        setConditionsList(profileData.conditions.split(',').map(c => c.trim()).filter(c => c))
      }
    } catch (error) {
      console.error('Failed to fetch profile:', error)
    }
  }

  const handleChange = (e) => {
    setProfile({ ...profile, [e.target.name]: e.target.value })
  }

  const handleAddCondition = () => {
    if (conditionInput.trim() && !conditionsList.includes(conditionInput.trim())) {
      const newList = [...conditionsList, conditionInput.trim()]
      setConditionsList(newList)
      setProfile({ ...profile, conditions: newList.join(', ') })
      setConditionInput('')
    }
  }

  const handleRemoveCondition = (condition) => {
    const newList = conditionsList.filter(c => c !== condition)
    setConditionsList(newList)
    setProfile({ ...profile, conditions: newList.join(', ') })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      await api.post('/api/profile', profile)
      alert('Profile updated successfully!')
      navigate('/my-med')
    } catch (error) {
      console.error('Failed to update profile:', error)
      alert('Failed to update profile')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="container mx-auto max-w-4xl px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-8"
        >
          <div className="flex items-center mb-6">
            <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mr-4">
              <User className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-800 dark:text-white">
                Complete Your Profile
              </h1>
              <p className="text-slate-600 dark:text-slate-400">
                Help us provide personalized health recommendations
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Information */}
            <div className="border-b border-slate-200 dark:border-slate-700 pb-6">
              <h2 className="text-xl font-semibold mb-4">Basic Information</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Full Name</label>
                  <input
                    type="text"
                    name="name"
                    value={profile.name}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Email</label>
                  <input
                    type="email"
                    name="email"
                    value={profile.email}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
                    disabled
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Date of Birth</label>
                  <input
                    type="date"
                    name="dob"
                    value={profile.dob}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Gender</label>
                  <select
                    name="gender"
                    value={profile.gender}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
                    required
                  >
                    <option value="">Select Gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Weight (kg)</label>
                  <input
                    type="number"
                    name="weight_kg"
                    value={profile.weight_kg}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Height (cm)</label>
                  <input
                    type="number"
                    name="height_cm"
                    value={profile.height_cm}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
                  />
                </div>
              </div>
            </div>

            {/* Medical Information */}
            <div className="border-b border-slate-200 dark:border-slate-700 pb-6">
              <h2 className="text-xl font-semibold mb-4 text-neutral-text">Medical Information</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-text mb-2">
                    Medical Conditions
                  </label>
                  <div className="flex flex-col sm:flex-row gap-2 mb-3">
                    <select
                      value={conditionInput}
                      onChange={(e) => setConditionInput(e.target.value)}
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                    >
                      <option value="">Select medical condition</option>
                      {MEDICAL_CONDITION_OPTIONS.map((condition) => (
                        <option key={condition} value={condition}>
                          {condition}
                        </option>
                      ))}
                    </select>
                    <input
                      type="text"
                      value={conditionInput}
                      onChange={(e) => setConditionInput(e.target.value)}
                      placeholder="Or type custom condition"
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                    />
                    <Button
                      type="button"
                      variant="secondary"
                      onClick={handleAddCondition}
                      icon={<Plus className="w-4 h-4" />}
                    >
                      Add
                    </Button>
                  </div>
                  
                  {/* Condition Chips */}
                  {conditionsList.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      <AnimatePresence>
                        {conditionsList.map((condition) => (
                          <motion.div
                            key={condition}
                            initial={{ scale: 0, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0, opacity: 0 }}
                            className="inline-flex items-center space-x-2 bg-primary-50 border border-primary-200 rounded-full px-3 py-1"
                          >
                            <span className="text-sm font-medium text-primary">{condition}</span>
                            <button
                              type="button"
                              onClick={() => handleRemoveCondition(condition)}
                              className="p-0.5 hover:bg-primary-200 rounded-full transition-colors"
                            >
                              <X className="w-3 h-3 text-primary-600" />
                            </button>
                          </motion.div>
                        ))}
                      </AnimatePresence>
                    </div>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-text mb-2">
                    Drug Allergies
                  </label>
                  <textarea
                    name="drug_allergies"
                    value={profile.drug_allergies}
                    onChange={handleChange}
                    placeholder="e.g., Penicillin, Aspirin"
                    rows="2"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Separate multiple allergies with commas
                  </p>
                </div>

                {/* Progressive Disclosure for Additional Allergies */}
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center space-x-2 text-primary hover:text-primary-600 transition-colors"
                >
                  {showAdvanced ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                  <span className="text-sm font-medium">
                    {showAdvanced ? 'Hide' : 'Show'} additional allergy information
                  </span>
                </button>

                <AnimatePresence>
                  {showAdvanced && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="space-y-4"
                    >
                      <div>
                        <label className="block text-sm font-medium text-neutral-text mb-2">
                          Food Allergies
                        </label>
                        <textarea
                          name="food_allergies"
                          value={profile.food_allergies}
                          onChange={handleChange}
                          placeholder="e.g., Peanuts, Shellfish"
                          rows="2"
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-neutral-text mb-2">
                          Other Allergies
                        </label>
                        <textarea
                          name="other_allergies"
                          value={profile.other_allergies}
                          onChange={handleChange}
                          placeholder="e.g., Latex, Pollen"
                          rows="2"
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Lifestyle */}
            <div className="pb-6">
              <h2 className="text-xl font-semibold mb-4 text-neutral-text">Lifestyle</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Smoking Status</label>
                  <select
                    name="is_smoker"
                    value={profile.is_smoker}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
                  >
                    <option value="">Select</option>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                    <option value="Former">Former Smoker</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Alcohol Consumption</label>
                  <select
                    name="alcohol_consumption"
                    value={profile.alcohol_consumption}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
                  >
                    <option value="">Select</option>
                    <option value="None">None</option>
                    <option value="Occasional">Occasional</option>
                    <option value="Regular">Regular</option>
                  </select>
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium mb-1">Emergency Contact</label>
                  <input
                    type="text"
                    name="emergency_contact"
                    value={profile.emergency_contact}
                    onChange={handleChange}
                    placeholder="Name and phone number"
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-slate-700"
                  />
                </div>
              </div>
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-primary text-white rounded-lg font-medium hover:bg-primary-600 disabled:opacity-50 flex items-center justify-center space-x-2 shadow-soft hover:shadow-soft-lg transition-all"
            >
              {loading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  <span>Saving...</span>
                </>
              ) : (
                <>
                  <Save className="w-5 h-5" />
                  <span>Save Profile</span>
                </>
              )}
            </motion.button>
          </form>
        </motion.div>
      </div>
    </div>
  )
}

export default Profile

