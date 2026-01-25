/**
 * Skip to Main Content Link
 * 
 * Provides keyboard users with a way to skip navigation and go directly to main content.
 * This is a WCAG requirement for accessibility compliance.
 */
const SkipToMain = () => {
  const handleClick = (e) => {
    e.preventDefault()
    const mainContent = document.getElementById('main-content')
    if (mainContent) {
      mainContent.focus()
      mainContent.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <a
      href="#main-content"
      onClick={handleClick}
      className="skip-to-main focus:left-0 focus:top-0"
      onFocus={(e) => {
        // Ensure the link is visible when focused
        e.target.style.left = '0'
        e.target.style.top = '0'
      }}
      onBlur={(e) => {
        // Hide the link when focus is lost
        e.target.style.left = '-9999px'
      }}
    >
      Skip to main content
    </a>
  )
}

export default SkipToMain
