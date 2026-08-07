import { motion, AnimatePresence } from 'framer-motion'
import { X } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { createAccessibleModal, announceToScreenReader } from '../../utils/accessibility'

const Modal = ({ 
  isOpen, 
  onClose, 
  title, 
  children, 
  size = 'md',
  closeOnEscape = true,
  closeOnBackdropClick = true,
  initialFocus,
  'aria-labelledby': ariaLabelledby,
  'aria-describedby': ariaDescribedby,
}) => {
  const modalRef = useRef(null)
  const titleId = `modal-title-${Math.random().toString(36).substr(2, 9)}`
  const contentId = `modal-content-${Math.random().toString(36).substr(2, 9)}`
  
  const sizes = {
    sm: 'max-w-md',
    md: 'max-w-2xl',
    lg: 'max-w-4xl',
    xl: 'max-w-6xl',
  }

  useEffect(() => {
    if (!modalRef.current) return

    const modalManager = createAccessibleModal(modalRef.current, {
      onClose,
      closeOnEscape,
      closeOnBackdropClick,
      initialFocus,
    })

    if (isOpen) {
      modalManager.open()
    }

    return () => {
      modalManager.cleanup()
    }
  }, [isOpen, onClose, closeOnEscape, closeOnBackdropClick, initialFocus])

  const handleBackdropClick = (e) => {
    if (closeOnBackdropClick && e.target === e.currentTarget) {
      onClose()
    }
  }

  const handleCloseClick = () => {
    announceToScreenReader('Modal closed', 'polite')
    onClose()
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleBackdropClick}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            aria-hidden="true"
          />
          
          {/* Modal */}
          <div 
            className="fixed inset-0 z-50 flex items-start sm:items-center justify-center p-4 overflow-y-auto"
            role="dialog"
            aria-modal="true"
            aria-labelledby={ariaLabelledby || titleId}
            aria-describedby={ariaDescribedby || contentId}
          >
            <motion.div
              ref={modalRef}
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className={`bg-white dark:bg-slate-800 rounded-card-lg shadow-soft-lg w-full ${sizes[size]} max-h-[90vh] overflow-hidden flex flex-col my-6`}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-gray-100">
                <h2 
                  id={titleId}
                  className="text-xl font-heading font-bold text-neutral-text"
                >
                  {title}
                </h2>
                <button
                  onClick={handleCloseClick}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-colors min-h-[44px] min-w-[44px] focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
                  aria-label="Close modal"
                  type="button"
                >
                  <X className="w-5 h-5" aria-hidden="true" />
                </button>
              </div>
              
              {/* Content */}
              <div 
                id={contentId}
                className="flex-1 overflow-y-auto p-6"
                tabIndex="-1"
              >
                {children}
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  )
}

export default Modal
