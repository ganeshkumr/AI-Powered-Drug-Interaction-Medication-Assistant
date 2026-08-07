/**
 * Accessibility Utilities
 * 
 * Provides utility functions for managing focus, announcements, and other accessibility features.
 * Requirements: 7.6 (Accessibility)
 */

/**
 * Manages focus during step transitions
 * @param {number} newStep - The step number being navigated to
 * @param {string} direction - 'forward' or 'backward'
 */
export const manageFocusTransition = (newStep, direction = 'forward') => {
  // Small delay to allow DOM updates
  setTimeout(() => {
    // Try to focus on the main heading of the new step
    const mainHeading = document.querySelector('h1');
    if (mainHeading) {
      mainHeading.focus();
      mainHeading.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      // Fallback to main content area
      const mainContent = document.getElementById('main-content');
      if (mainContent) {
        mainContent.focus();
      }
    }
  }, 100);
};

/**
 * Announces content changes to screen readers
 * @param {string} message - The message to announce
 * @param {string} priority - 'polite' or 'assertive'
 */
export const announceToScreenReader = (message, priority = 'polite') => {
  const announcement = document.createElement('div');
  announcement.setAttribute('aria-live', priority);
  announcement.setAttribute('aria-atomic', 'true');
  announcement.className = 'sr-only';
  announcement.textContent = message;
  
  document.body.appendChild(announcement);
  
  // Remove the announcement after it's been read
  setTimeout(() => {
    document.body.removeChild(announcement);
  }, 1000);
};

/**
 * Traps focus within a modal or dropdown
 * @param {HTMLElement} container - The container element to trap focus within
 */
export const trapFocus = (container) => {
  const focusableElements = container.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];
  
  const handleTabKey = (e) => {
    if (e.key === 'Tab') {
      if (e.shiftKey) {
        // Shift + Tab
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        // Tab
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    }
    
    if (e.key === 'Escape') {
      // Allow escape to close modals
      const closeButton = container.querySelector('[data-close-modal]');
      if (closeButton) {
        closeButton.click();
      }
    }
  };
  
  container.addEventListener('keydown', handleTabKey);
  
  // Focus the first element
  if (firstElement) {
    firstElement.focus();
  }
  
  // Return cleanup function
  return () => {
    container.removeEventListener('keydown', handleTabKey);
  };
};

/**
 * Generates unique IDs for form elements and their labels
 * @param {string} prefix - Prefix for the ID
 */
export const generateId = (prefix = 'element') => {
  return `${prefix}-${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Validates color contrast for accessibility compliance
 * @param {string} foreground - Foreground color (hex)
 * @param {string} background - Background color (hex)
 * @param {boolean} isLargeText - Whether the text is large (18pt+ or 14pt+ bold)
 * @returns {Object} - Contrast ratio and compliance status
 */
export const validateColorContrast = (foreground, background, isLargeText = false) => {
  // Convert hex to RGB
  const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  };
  
  // Calculate relative luminance
  const getLuminance = (rgb) => {
    const { r, g, b } = rgb;
    const [rs, gs, bs] = [r, g, b].map(c => {
      c = c / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
  };
  
  const fg = hexToRgb(foreground);
  const bg = hexToRgb(background);
  
  if (!fg || !bg) return { ratio: 0, passesAA: false, passesAAA: false };
  
  const fgLum = getLuminance(fg);
  const bgLum = getLuminance(bg);
  
  const contrast = (Math.max(fgLum, bgLum) + 0.05) / (Math.min(fgLum, bgLum) + 0.05);
  
  // WCAG standards
  const aaThreshold = isLargeText ? 3.0 : 4.5;
  const aaaThreshold = isLargeText ? 4.5 : 7.0;
  
  return {
    ratio: Math.round(contrast * 100) / 100,
    passesAA: contrast >= aaThreshold,
    passesAAA: contrast >= aaaThreshold
  };
};

/**
 * Keyboard navigation handler for custom components
 * @param {KeyboardEvent} event - The keyboard event
 * @param {Object} handlers - Object containing key handlers
 */
export const handleKeyboardNavigation = (event, handlers = {}) => {
  const { key } = event;
  
  const keyMap = {
    'Enter': handlers.onEnter,
    ' ': handlers.onSpace,
    'ArrowUp': handlers.onArrowUp,
    'ArrowDown': handlers.onArrowDown,
    'ArrowLeft': handlers.onArrowLeft,
    'ArrowRight': handlers.onArrowRight,
    'Home': handlers.onHome,
    'End': handlers.onEnd,
    'Escape': handlers.onEscape,
    'Tab': handlers.onTab
  };
  
  const handler = keyMap[key];
  if (handler) {
    handler(event);
  }
};

/**
 * Ensures minimum touch target size for mobile accessibility
 * @param {HTMLElement} element - The element to check
 * @returns {boolean} - Whether the element meets minimum size requirements
 */
export const validateTouchTargetSize = (element) => {
  const rect = element.getBoundingClientRect();
  const minSize = 44; // 44px minimum as per WCAG guidelines
  
  return rect.width >= minSize && rect.height >= minSize;
};

/**
 * Creates accessible loading announcements
 * @param {boolean} isLoading - Whether content is loading
 * @param {string} loadingMessage - Message to announce when loading starts
 * @param {string} completeMessage - Message to announce when loading completes
 */
export const manageLoadingAnnouncements = (isLoading, loadingMessage = 'Loading', completeMessage = 'Content loaded') => {
  if (isLoading) {
    announceToScreenReader(loadingMessage, 'assertive');
  } else {
    announceToScreenReader(completeMessage, 'polite');
  }
};

/**
 * Creates accessible form field with proper labeling and error handling
 * @param {Object} options - Configuration options
 * @returns {Object} - Accessibility props for form field
 */
export const createAccessibleFormField = (options = {}) => {
  const {
    id,
    label,
    error,
    description,
    required = false,
    invalid = false
  } = options;

  const fieldId = id || generateId('field');
  const labelId = `${fieldId}-label`;
  const errorId = error ? `${fieldId}-error` : undefined;
  const descriptionId = description ? `${fieldId}-description` : undefined;

  const describedBy = [descriptionId, errorId].filter(Boolean).join(' ');

  return {
    fieldProps: {
      id: fieldId,
      'aria-labelledby': labelId,
      'aria-describedby': describedBy || undefined,
      'aria-invalid': invalid,
      'aria-required': required,
    },
    labelProps: {
      id: labelId,
      htmlFor: fieldId,
    },
    errorProps: error ? {
      id: errorId,
      role: 'alert',
      'aria-live': 'polite',
    } : {},
    descriptionProps: description ? {
      id: descriptionId,
    } : {},
  };
};

/**
 * Enhanced keyboard navigation handler with comprehensive key support
 * @param {KeyboardEvent} event - The keyboard event
 * @param {Object} handlers - Object containing key handlers
 */
export const handleAdvancedKeyboardNavigation = (event, handlers = {}) => {
  const { key, ctrlKey, metaKey, shiftKey, altKey } = event;
  
  // Handle modifier combinations
  const modifiers = {
    ctrl: ctrlKey,
    meta: metaKey,
    shift: shiftKey,
    alt: altKey,
  };

  const keyMap = {
    'Enter': handlers.onEnter,
    ' ': handlers.onSpace,
    'ArrowUp': handlers.onArrowUp,
    'ArrowDown': handlers.onArrowDown,
    'ArrowLeft': handlers.onArrowLeft,
    'ArrowRight': handlers.onArrowRight,
    'Home': handlers.onHome,
    'End': handlers.onEnd,
    'PageUp': handlers.onPageUp,
    'PageDown': handlers.onPageDown,
    'Escape': handlers.onEscape,
    'Tab': handlers.onTab,
    'F1': handlers.onF1,
    'F2': handlers.onF2,
    'F3': handlers.onF3,
    'F4': handlers.onF4,
    'F5': handlers.onF5,
    'F6': handlers.onF6,
    'F7': handlers.onF7,
    'F8': handlers.onF8,
    'F9': handlers.onF9,
    'F10': handlers.onF10,
    'F11': handlers.onF11,
    'F12': handlers.onF12,
  };
  
  const handler = keyMap[key];
  if (handler) {
    handler(event, modifiers);
  }

  // Handle character keys for search/filter
  if (handlers.onCharacter && key.length === 1 && !ctrlKey && !metaKey && !altKey) {
    handlers.onCharacter(event, key);
  }
};

/**
 * Creates accessible modal/dialog with proper focus management
 * @param {HTMLElement} modalElement - The modal container element
 * @param {Object} options - Configuration options
 * @returns {Object} - Modal management functions
 */
export const createAccessibleModal = (modalElement, options = {}) => {
  const {
    onClose,
    closeOnEscape = true,
    closeOnBackdropClick = true,
    returnFocusTo,
    initialFocus,
  } = options;

  let previousActiveElement = null;
  let focusTrapCleanup = null;
  let previousBodyOverflow = '';

  const open = () => {
    // Store the previously focused element
    previousActiveElement = document.activeElement;
    
    // Prevent body scroll
    previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    
    // Set up focus trap
    focusTrapCleanup = trapFocus(modalElement);
    
    // Focus initial element or first focusable element
    setTimeout(() => {
      if (initialFocus && typeof initialFocus === 'function') {
        initialFocus();
      } else if (initialFocus) {
        initialFocus.focus();
      } else {
        const firstFocusable = modalElement.querySelector(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (firstFocusable) {
          firstFocusable.focus();
        }
      }
    }, 100);

    // Announce modal opening
    announceToScreenReader('Dialog opened', 'assertive');
  };

  const close = () => {
    // Restore body scroll
    document.body.style.overflow = previousBodyOverflow;
    
    // Clean up focus trap
    if (focusTrapCleanup) {
      focusTrapCleanup();
      focusTrapCleanup = null;
    }
    
    // Return focus to previous element or specified element
    const focusTarget = returnFocusTo || previousActiveElement;
    if (focusTarget && focusTarget.focus) {
      focusTarget.focus();
    }
    
    // Announce modal closing
    announceToScreenReader('Dialog closed', 'polite');
    
    // Call onClose callback
    if (onClose) {
      onClose();
    }
  };

  const handleKeyDown = (event) => {
    if (closeOnEscape && event.key === 'Escape') {
      event.preventDefault();
      close();
    }
  };

  const handleBackdropClick = (event) => {
    if (closeOnBackdropClick && event.target === modalElement) {
      close();
    }
  };

  // Set up event listeners
  modalElement.addEventListener('keydown', handleKeyDown);
  if (closeOnBackdropClick) {
    modalElement.addEventListener('click', handleBackdropClick);
  }

  return {
    open,
    close,
    cleanup: () => {
      modalElement.removeEventListener('keydown', handleKeyDown);
      modalElement.removeEventListener('click', handleBackdropClick);
      document.body.style.overflow = previousBodyOverflow;
      if (focusTrapCleanup) {
        focusTrapCleanup();
        focusTrapCleanup = null;
      }
    },
  };
};

/**
 * Validates and reports accessibility issues for an element
 * @param {HTMLElement} element - The element to validate
 * @returns {Array} - Array of accessibility issues found
 */
export const validateAccessibility = (element) => {
  const issues = [];

  // Check for missing alt text on images
  const images = element.querySelectorAll('img');
  images.forEach((img, index) => {
    if (!img.alt && !img.getAttribute('aria-label') && !img.getAttribute('aria-labelledby')) {
      issues.push({
        type: 'missing-alt-text',
        element: img,
        message: `Image ${index + 1} is missing alt text`,
        severity: 'error',
      });
    }
  });

  // Check for missing form labels
  const inputs = element.querySelectorAll('input, textarea, select');
  inputs.forEach((input, index) => {
    const hasLabel = input.labels && input.labels.length > 0;
    const hasAriaLabel = input.getAttribute('aria-label');
    const hasAriaLabelledby = input.getAttribute('aria-labelledby');
    
    if (!hasLabel && !hasAriaLabel && !hasAriaLabelledby) {
      issues.push({
        type: 'missing-form-label',
        element: input,
        message: `Form field ${index + 1} is missing a label`,
        severity: 'error',
      });
    }
  });

  // Check for insufficient color contrast
  const textElements = element.querySelectorAll('p, span, div, h1, h2, h3, h4, h5, h6, a, button');
  textElements.forEach((el, index) => {
    const styles = window.getComputedStyle(el);
    const color = styles.color;
    const backgroundColor = styles.backgroundColor;
    
    if (color && backgroundColor && color !== 'rgba(0, 0, 0, 0)' && backgroundColor !== 'rgba(0, 0, 0, 0)') {
      // Convert RGB to hex for validation (simplified)
      const rgbToHex = (rgb) => {
        const match = rgb.match(/\d+/g);
        if (match && match.length >= 3) {
          return '#' + match.slice(0, 3).map(x => parseInt(x).toString(16).padStart(2, '0')).join('');
        }
        return null;
      };
      
      const fgHex = rgbToHex(color);
      const bgHex = rgbToHex(backgroundColor);
      
      if (fgHex && bgHex) {
        const fontSize = parseFloat(styles.fontSize);
        const fontWeight = styles.fontWeight;
        const isLargeText = fontSize >= 18 || (fontSize >= 14 && (fontWeight === 'bold' || parseInt(fontWeight) >= 700));
        
        const contrastResult = validateColorContrast(fgHex, bgHex, isLargeText);
        
        if (!contrastResult.passesAA) {
          issues.push({
            type: 'insufficient-contrast',
            element: el,
            message: `Text element ${index + 1} has insufficient color contrast (${contrastResult.ratio}:1)`,
            severity: 'warning',
            details: contrastResult,
          });
        }
      }
    }
  });

  // Check for missing heading hierarchy
  const headings = element.querySelectorAll('h1, h2, h3, h4, h5, h6');
  let previousLevel = 0;
  headings.forEach((heading, index) => {
    const currentLevel = parseInt(heading.tagName.charAt(1));
    if (currentLevel > previousLevel + 1) {
      issues.push({
        type: 'heading-hierarchy-skip',
        element: heading,
        message: `Heading ${index + 1} skips levels (from h${previousLevel} to h${currentLevel})`,
        severity: 'warning',
      });
    }
    previousLevel = currentLevel;
  });

  // Check for touch target size
  const interactiveElements = element.querySelectorAll('button, a, input[type="checkbox"], input[type="radio"], [role="button"]');
  interactiveElements.forEach((el, index) => {
    if (!validateTouchTargetSize(el)) {
      issues.push({
        type: 'insufficient-touch-target',
        element: el,
        message: `Interactive element ${index + 1} is smaller than minimum touch target size (44px)`,
        severity: 'warning',
      });
    }
  });

  return issues;
};
