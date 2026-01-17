import { motion } from 'framer-motion'
import { Pill, X } from 'lucide-react'

const MedicationChip = ({ drug, dosage, onRemove }) => {
  return (
    <motion.div
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0, opacity: 0 }}
      whileHover={{ scale: 1.05 }}
      className="inline-flex items-center space-x-2 bg-primary-50 border border-primary-200 rounded-full px-4 py-2 group hover:bg-primary-100 transition-colors"
    >
      <Pill className="w-4 h-4 text-primary" />
      <div className="flex flex-col">
        <span className="text-sm font-medium text-neutral-text">{drug}</span>
        {dosage && (
          <span className="text-xs text-gray-500">{dosage}</span>
        )}
      </div>
      <button
        onClick={onRemove}
        className="ml-2 p-1 hover:bg-primary-200 rounded-full transition-colors"
        aria-label={`Remove ${drug}`}
      >
        <X className="w-3 h-3 text-primary-600" />
      </button>
    </motion.div>
  )
}

export default MedicationChip
