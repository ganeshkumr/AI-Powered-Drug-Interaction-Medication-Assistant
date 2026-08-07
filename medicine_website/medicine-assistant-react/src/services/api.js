import axios from 'axios'

// In production (Vercel), VITE_API_URL points to the Render backend.
// Falls back to the deployed Render backend URL so production builds always connect.
export const BASE_URL = import.meta.env.VITE_API_URL || 'https://ai-powered-drug-interaction-medication.onrender.com'

const api = axios.create({
  baseURL: BASE_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const healthAPI = {
  getHealthData: () => api.get('/api/health-data'),
  getHealthAlerts: () => api.get('/api/health-alerts'),
}

export const medicationAPI = {
  checkBeforeAdding: (data) => api.post('/check_before_adding', data),
  addMedication: (data) => api.post('/add_medication', data),
  getMedications: () => api.get('/api/medications'),
  deleteMedication: (id) => api.delete(`/api/medications/${id}`),
}

export const emergencyAPI = {
  checkInteraction: (drug1, drug2) =>
    api.post('/emergency-check', { drug1, drug2 }),
}

export const chatbotAPI = {
  askAssistant: (question) =>
    api.post('/ask_assistant', { question }),
}

export default api
