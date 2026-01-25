import { motion } from 'framer-motion'
import { Pill, X } from 'lucide-react'

const MedicationChip = ({ drug, dosage, onRemove }) => {
  return (
    <motion.div
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0, opacity: 0 }}
      whileHover={{ 
        scale: 1.05,
        y: -2,
        transition: { duration: 0.2, ease: [0, 0, 0.2, 1] }
      }}
      whileTap={{ 
        scale: 0.98,
        transition: { duration: 0.1 }
      }}
      className="inline-flex items-center space-x-2 bg-primary-50 border border-primary-200 rounded-full px-4 py-2 group hover:bg-primary-100 transition-all duration-200 medical-chip-interactive"
    >
      <motion.div
        initial={{ rotate: 0 }}
        whileHover={{ rotate: 360 }}
        transition={{ duration: 0.5, ease: "easeInOut" }}
      >
        <Pill className="w-4 h-4 text-primary" />
      </motion.div>
      <div className="flex flex-col">
        <span className="text-sm font-medium text-neutral-text">{drug}</span>
        {dosage && (
          <span className="text-xs text-gray-500">{dosage}</span>
        )}
      </div>
      <motion.button
        onClick={onRemove}
        whileHover={{ 
          scale: 1.2,
          rotate: 90,
          backgroundColor: "rgb(239 68 68 / 0.1)",
          transition: { duration: 0.2 }
        }}
        whileTap={{ 
          scale: 0.9,
          transition: { duration: 0.1 }
        }}
        className="ml-2 p-1 hover:bg-red-100 rounded-full transition-all duration-200 medical-focus-ring"
        aria-label={`Remove ${drug}`}
      >
        <X className="w-3 h-3 text-primary-600 group-hover:text-red-600 transition-colors duration-200" />
      </motion.button>
    </motion.div>
  )
}

export default MedicationChip
