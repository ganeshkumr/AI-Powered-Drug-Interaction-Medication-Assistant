import { motion } from 'framer-motion'
import { Moon, Sun, User, LogOut, Activity } from 'lucide-react'
import { useTheme } from '../../context/ThemeContext'
import { useAuth } from '../../context/AuthContext'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

const Header = () => {
  const { theme, toggleTheme } = useTheme()
  const { user, logout } = useAuth()
  const [showDropdown, setShowDropdown] = useState(false)
  const navigate = useNavigate()

  const handleLogout = async () => {
    await logout()
    navigate('/login')
  }

  return (
    <motion.header
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="bg-white dark:bg-slate-800 shadow-soft sticky top-0 z-40 border-b border-gray-100"
    >
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center space-x-3 cursor-pointer"
            onClick={() => navigate('/dashboard')}
          >
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-primary-600 rounded-xl flex items-center justify-center shadow-soft">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-heading font-bold gradient-text">
                AI-HealthMate
              </h1>
              <p className="text-xs text-gray-500">Medicine Safety Assistant</p>
            </div>
          </motion.div>

          <div className="flex items-center space-x-4">
            {/* Live Status */}
            <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 bg-green-50 rounded-full">
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 2 }}
                className="w-2 h-2 bg-status-safe rounded-full"
              />
              <span className="text-sm font-medium text-status-safe">
                Live
              </span>
            </div>

            {/* Theme Toggle */}
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={toggleTheme}
              className="p-2 rounded-lg hover:bg-primary-50 text-gray-600 hover:text-primary transition-colors"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5" />
              ) : (
                <Sun className="w-5 h-5" />
              )}
            </motion.button>

            {/* User Dropdown */}
            <div className="relative">
              <motion.button
                whileHover={{ scale: 1.05 }}
                onClick={() => setShowDropdown(!showDropdown)}
                className="flex items-center space-x-2 p-2 rounded-lg hover:bg-primary-50 transition-colors"
                aria-label="User menu"
              >
                <div className="w-9 h-9 bg-gradient-to-br from-primary to-primary-600 rounded-full flex items-center justify-center text-white font-semibold shadow-soft">
                  {user?.name?.[0]?.toUpperCase() || 'U'}
                </div>
                <span className="text-sm font-medium hidden sm:block text-neutral-text">
                  {user?.name}
                </span>
              </motion.button>

              {showDropdown && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute right-0 mt-2 w-48 bg-white dark:bg-slate-800 rounded-card shadow-soft-lg border border-gray-100 dark:border-slate-700 overflow-hidden"
                >
                  <button
                    onClick={() => {
                      navigate('/profile')
                      setShowDropdown(false)
                    }}
                    className="w-full flex items-center space-x-3 px-4 py-3 hover:bg-primary-50 transition-colors text-left"
                  >
                    <User className="w-4 h-4 text-primary" />
                    <span className="text-sm font-medium">Edit Profile</span>
                  </button>
                  <div className="border-t border-gray-100"></div>
                  <button
                    onClick={handleLogout}
                    className="w-full flex items-center space-x-3 px-4 py-3 hover:bg-red-50 transition-colors text-left text-status-danger"
                  >
                    <LogOut className="w-4 h-4" />
                    <span className="text-sm font-medium">Logout</span>
                  </button>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>
    </motion.header>
  )
}

export default Header