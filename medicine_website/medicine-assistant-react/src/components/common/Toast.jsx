import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, AlertCircle, Info, X } from 'lucide-react'
import { createContext, useContext, useState } from 'react'

const ToastContext = createContext()

export const useToast = () => {
  const context = useContext(ToastContext)
  if (!context) throw new Error('useToast must be used within ToastProvider')
  return context
}

export const ToastProvider = ({ children }) => {
  const [toasts, setToasts] = useState([])

  const addToast = (message, type = 'info') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => removeToast(id), 5000)
  }

  const removeToast = (id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id))
  }

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 space-y-2">
        <AnimatePresence>
          {toasts.map(toast => (
            <Toast key={toast.id} {...toast} onClose={() => removeToast(toast.id)} />
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  )
}

const Toast = ({ message, type, onClose }) => {
  const icons = {
    success: <CheckCircle className="w-5 h-5 text-status-safe" />,
    error: <AlertCircle className="w-5 h-5 text-status-danger" />,
    warning: <AlertCircle className="w-5 h-5 text-status-warning" />,
    info: <Info className="w-5 h-5 text-primary" />,
  }

  const backgrounds = {
    success: 'bg-green-50 border-status-safe',
    error: 'bg-red-50 border-status-danger',
    warning: 'bg-yellow-50 border-status-warning',
    info: 'bg-primary-50 border-primary',
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 100 }}
      className={`flex items-center space-x-3 p-4 rounded-card shadow-soft-lg border-l-4 ${backgrounds[type]} min-w-[300px]`}
    >
      {icons[type]}
      <p className="flex-1 text-sm font-medium text-neutral-text">{message}</p>
      <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
        <X className="w-4 h-4" />
      </button>
    </motion.div>
  )
}

export default Toast
