import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:5000',
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