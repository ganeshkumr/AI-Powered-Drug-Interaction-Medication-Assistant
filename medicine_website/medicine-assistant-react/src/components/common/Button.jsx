import { motion } from 'framer-motion'
import { Loader } from 'lucide-react'

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  icon, 
  loading = false, 
  disabled = false,
  className = '',
  ...props 
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed'
  
  const variants = {
    primary: 'bg-primary text-white hover:bg-primary-600 focus:ring-primary-500 shadow-soft hover:shadow-soft-lg',
    secondary: 'bg-white text-primary border-2 border-primary hover:bg-primary-50 focus:ring-primary-500',
    danger: 'bg-status-danger text-white hover:bg-red-600 focus:ring-red-500 shadow-soft hover:shadow-soft-lg',
    accent: 'bg-accent text-white hover:bg-accent-600 focus:ring-accent-500 shadow-soft hover:shadow-soft-lg',
    ghost: 'bg-transparent text-primary hover:bg-primary-50 focus:ring-primary-500',
  }
  
  const sizes = {
    sm: 'px-3 py-2 text-sm min-h-[36px]',
    md: 'px-4 py-2.5 text-base min-h-[44px]',
    lg: 'px-6 py-3 text-lg min-h-[52px]',
  }
  
  return (
    <motion.button
      whileHover={{ scale: disabled || loading ? 1 : 1.02 }}
      whileTap={{ scale: disabled || loading ? 1 : 0.98 }}
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <>
          <Loader className="w-5 h-5 mr-2 animate-spin" />
          <span>Loading...</span>
        </>
      ) : (
        <>
          {icon && <span className="mr-2">{icon}</span>}
          {children}
        </>
      )}
    </motion.button>
  )
}

export default Button
