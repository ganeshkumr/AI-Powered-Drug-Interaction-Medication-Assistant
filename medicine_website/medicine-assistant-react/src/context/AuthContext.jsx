import { createContext, useContext, useState, useEffect } from 'react'
import api from '../services/api'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used within AuthProvider')
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    checkAuth()
  }, [])

  const checkAuth = async () => {
    try {
      const response = await api.get('/api/check-auth')
      if (response.data.authenticated) {
        setUser(response.data.user)
        setIsAuthenticated(true)
      }
    } catch (error) {
      console.error('Auth check failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const login = async (email, password) => {
    try {
      const response = await api.post('/api/login', { email, password })
      if (response.data.success) {
        setUser(response.data.user)
        setIsAuthenticated(true)
        return { success: true }
      }
      return { success: false, error: response.data.error }
    } catch (error) {
      console.error('Login error:', error)
      return { success: false, error: error.response?.data?.error || 'Login failed' }
    }
  }

  const register = async (name, email, password) => {
    try {
      const response = await api.post('/api/register', { name, email, password })
      if (response.data.success) {
        setUser(response.data.user)
        setIsAuthenticated(true)
        return { success: true }
      }
      return { success: false, error: response.data.error }
    } catch (error) {
      console.error('Registration error:', error)
      return { success: false, error: error.response?.data?.error || 'Registration failed' }
    }
  }

  const logout = async () => {
    await api.get('/api/logout')
    setUser(null)
    setIsAuthenticated(false)
  }

  return (
    <AuthContext.Provider value={{
      user,
      isAuthenticated,
      loading,
      login,
      register,
      logout
    }}>
      {children}
    </AuthContext.Provider>
  )
}