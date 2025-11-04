import { motion } from 'framer-motion'

const Card = ({ 
  children, 
  shadow = 'soft', 
  padding = 'md', 
  rounded = 'card',
  className = '',
  hover = false,
  ...props 
}) => {
  const shadows = {
    soft: 'shadow-soft',
    'soft-lg': 'shadow-soft-lg',
    none: 'shadow-none',
  }
  
  const paddings = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  }
  
  const roundeds = {
    card: 'rounded-card',
    'card-lg': 'rounded-card-lg',
  }
  
  const baseStyles = 'bg-white dark:bg-slate-800 border border-gray-100 dark:border-slate-700'
  const hoverStyles = hover ? 'hover:shadow-soft-lg transition-shadow duration-200' : ''
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${baseStyles} ${shadows[shadow]} ${paddings[padding]} ${roundeds[rounded]} ${hoverStyles} ${className}`}
      {...props}
    >
      {children}
    </motion.div>
  )
}

export default Card
