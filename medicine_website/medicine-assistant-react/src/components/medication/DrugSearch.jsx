import { useState, useEffect, useRef } from 'react'
import { Search, Loader2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  handleAdvancedKeyboardNavigation, 
  announceToScreenReader, 
  createAccessibleFormField,
  generateId 
} from '../../utils/accessibility'

const DrugSearch = ({ 
  onSelect, 
  placeholder = "Search for a medication...",
  id,
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedby,
  disabled = false,
}) => {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [loading, setLoading] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const [announceCount, setAnnounceCount] = useState(0)
  
  const inputRef = useRef(null)
  const dropdownRef = useRef(null)
  const listboxId = generateId('drugSearch-listbox')
  const fieldId = id || generateId('drugSearch')

  // Create accessible form field props
  const { fieldProps, labelProps } = createAccessibleFormField({
    id: fieldId,
    label: ariaLabel || 'Search for medications',
    description: 'Type to search for medications in our database',
  })

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target) &&
        !inputRef.current.contains(event.target)
      ) {
        setShowDropdown(false)
        setSelectedIndex(-1)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    const searchDrugs = async () => {
      if (query.length < 2) {
        setSuggestions([])
        setShowDropdown(false)
        setSelectedIndex(-1)
        return
      }

      setLoading(true)
      try {
        const response = await fetch(`http://localhost:5000/api/search-drugs?q=${encodeURIComponent(query)}`)
        if (response.ok) {
          const data = await response.json()
          const drugs = data.drugs || []
          setSuggestions(drugs)
          setShowDropdown(true)
          setSelectedIndex(-1)
          
          // Announce search results to screen readers
          const count = drugs.length
          setAnnounceCount(count)
          if (count === 0) {
            announceToScreenReader('No medications found', 'polite')
          } else {
            announceToScreenReader(`${count} medication${count === 1 ? '' : 's'} found`, 'polite')
          }
        }
      } catch (error) {
        console.error('Drug search error:', error)
        setSuggestions([])
        announceToScreenReader('Search failed. Please try again.', 'assertive')
      } finally {
        setLoading(false)
      }
    }

    const debounce = setTimeout(searchDrugs, 300)
    return () => clearTimeout(debounce)
  }, [query])

  const handleSelect = (drug, index = -1) => {
    onSelect(drug)
    setQuery('')
    setSuggestions([])
    setShowDropdown(false)
    setSelectedIndex(-1)
    inputRef.current?.focus()
    
    announceToScreenReader(`${drug} selected`, 'polite')
  }

  const handleKeyDown = (event) => {
    handleAdvancedKeyboardNavigation(event, {
      onArrowDown: (e) => {
        e.preventDefault()
        if (!showDropdown || suggestions.length === 0) return
        
        const newIndex = selectedIndex < suggestions.length - 1 ? selectedIndex + 1 : 0
        setSelectedIndex(newIndex)
        announceToScreenReader(`${suggestions[newIndex]} option ${newIndex + 1} of ${suggestions.length}`, 'assertive')
      },
      onArrowUp: (e) => {
        e.preventDefault()
        if (!showDropdown || suggestions.length === 0) return
        
        const newIndex = selectedIndex > 0 ? selectedIndex - 1 : suggestions.length - 1
        setSelectedIndex(newIndex)
        announceToScreenReader(`${suggestions[newIndex]} option ${newIndex + 1} of ${suggestions.length}`, 'assertive')
      },
      onEnter: (e) => {
        e.preventDefault()
        if (selectedIndex >= 0 && suggestions[selectedIndex]) {
          handleSelect(suggestions[selectedIndex], selectedIndex)
        }
      },
      onEscape: (e) => {
        e.preventDefault()
        setShowDropdown(false)
        setSelectedIndex(-1)
        announceToScreenReader('Search suggestions closed', 'polite')
      },
      onTab: () => {
        setShowDropdown(false)
        setSelectedIndex(-1)
      },
    })
  }

  const handleInputChange = (e) => {
    const value = e.target.value
    setQuery(value)
    if (value.length < 2) {
      setShowDropdown(false)
      setSelectedIndex(-1)
    }
  }

  const handleInputFocus = () => {
    if (query.length >= 2 && suggestions.length > 0) {
      setShowDropdown(true)
    }
  }

  return (
    <div className="relative">
      {/* Hidden label for screen readers */}
      <label {...labelProps} className="sr-only">
        {ariaLabel || 'Search for medications'}
      </label>
      
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" aria-hidden="true" />
        <input
          {...fieldProps}
          ref={inputRef}
          type="text"
          value={query}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className="w-full pl-10 pr-10 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all min-h-[44px] disabled:opacity-50 disabled:cursor-not-allowed medical-input-enhanced"
          aria-label={ariaLabel}
          aria-describedby={ariaDescribedby}
          aria-expanded={showDropdown}
          aria-haspopup="listbox"
          aria-owns={showDropdown ? listboxId : undefined}
          aria-activedescendant={selectedIndex >= 0 ? `${listboxId}-option-${selectedIndex}` : undefined}
          aria-autocomplete="list"
          role="combobox"
        />
        {loading && (
          <Loader2 
            className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-primary animate-spin" 
            aria-hidden="true"
          />
        )}
      </div>

      {/* Live region for announcements */}
      <div 
        aria-live="polite" 
        aria-atomic="true" 
        className="sr-only"
        role="status"
      >
        {loading && 'Searching medications...'}
        {!loading && showDropdown && announceCount > 0 && 
          `${announceCount} medication${announceCount === 1 ? '' : 's'} available. Use arrow keys to navigate.`
        }
      </div>

      <AnimatePresence>
        {showDropdown && suggestions.length > 0 && (
          <motion.div
            ref={dropdownRef}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute z-50 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-soft-lg max-h-60 overflow-y-auto"
            role="listbox"
            id={listboxId}
            aria-label="Medication suggestions"
          >
            {suggestions.map((drug, index) => (
              <motion.button
                key={`${drug}-${index}`}
                id={`${listboxId}-option-${index}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05, duration: 0.2, ease: [0, 0, 0.2, 1] }}
                whileHover={{ 
                  x: 4,
                  backgroundColor: selectedIndex === index ? 'rgb(239 246 255)' : 'rgb(239 246 255)',
                  transition: { duration: 0.15 }
                }}
                whileTap={{ 
                  scale: 0.98,
                  transition: { duration: 0.1 }
                }}
                onClick={() => handleSelect(drug, index)}
                onMouseEnter={() => setSelectedIndex(index)}
                className={`w-full px-4 py-3 text-left transition-all duration-200 border-b border-gray-100 last:border-b-0 min-h-[44px] focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-inset medical-hover-scale ${
                  selectedIndex === index 
                    ? 'bg-primary-50 text-primary-700 border-l-4 border-l-primary-500' 
                    : 'hover:bg-primary-50 text-neutral-700 hover:border-l-4 hover:border-l-primary-300'
                }`}
                role="option"
                aria-selected={selectedIndex === index}
                type="button"
              >
                <div className="font-medium">{drug}</div>
              </motion.button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {showDropdown && query.length >= 2 && suggestions.length === 0 && !loading && (
        <div 
          className="absolute z-50 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-soft-lg p-4 text-center text-gray-500"
          role="status"
          aria-live="polite"
        >
          No medications found. Try a different search term.
        </div>
      )}
    </div>
  )
}

export default DrugSearch
