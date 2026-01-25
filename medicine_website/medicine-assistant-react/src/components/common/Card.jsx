import { motion } from 'framer-motion'

/**
 * Medical-Grade Card Component
 * 
 * Implements medical-appropriate styling with:
 * - Subtle shadows for professional appearance
 * - Consistent padding and border radius
 * - Smooth hover interactions
 * - Accessibility-compliant focus states
 */
const Card = ({ 
  children, 
  shadow = 'card', 
  padding = 'md', 
  rounded = 'card',
  className = '',
  hover = false,
  interactive = false,
  ...props 
}) => {
  const shadows = {
    none: 'shadow-none',
    xs: 'shadow-xs',
    sm: 'shadow-sm',
    card: 'shadow-card',
    base: 'shadow-base',
    md: 'shadow-md',
    lg: 'shadow-lg',
    xl: 'shadow-xl',
    '2xl': 'shadow-2xl',
  }
  
  const paddings = {
    none: 'p-0',
    sm: 'p-3',
    md: 'p-4 sm:p-6',
    lg: 'p-6 sm:p-8',
    xl: 'p-8 sm:p-10',
  }
  
  const roundeds = {
    none: 'rounded-none',
    sm: 'rounded-sm',
    base: 'rounded-base',
    card: 'rounded-card',
    md: 'rounded-md',
    lg: 'rounded-lg',
    xl: 'rounded-xl',
  }
  
  const baseStyles = `
    bg-white border border-neutral-200
    transition-all duration-base ease-out
  `
  
  const hoverStyles = hover || interactive ? `
    hover:shadow-card-hover hover:-translate-y-1 medical-card-interactive
  ` : ''
  
  const interactiveStyles = interactive ? `
    cursor-pointer select-none medical-focus-ring
    focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
    active:translate-y-0 active:shadow-card
  ` : ''
  
  const Component = interactive ? motion.button : motion.div
  
  return (
    <Component
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        duration: 0.3, 
        ease: [0, 0, 0.2, 1] 
      }}
      whileHover={hover || interactive ? { 
        y: -4,
        scale: 1.01,
        transition: { duration: 0.2, ease: [0, 0, 0.2, 1] }
      } : undefined}
      whileTap={interactive ? { 
        y: 0,
        scale: 0.99,
        transition: { duration: 0.1, ease: [0, 0, 0.2, 1] }
      } : undefined}
      className={`
        ${baseStyles} 
        ${shadows[shadow]} 
        ${paddings[padding]} 
        ${roundeds[rounded]} 
        ${hoverStyles}
        ${interactiveStyles}
        ${className}
      `}
      {...props}
    >
      {children}
    </Component>
  )
}

export default Card
