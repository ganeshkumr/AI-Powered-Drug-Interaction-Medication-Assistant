import { motion } from 'framer-motion'
import GlobalNavigation from '../navigation/GlobalNavigation'
import Footer from './Footer'
import SkipToMain from '../common/SkipToMain'
import ChatbotButton from '../chatbot/ChatbotButton'
import { useAuth } from '../../context/AuthContext'

const Layout = ({ children }) => {
  const { user } = useAuth()

  // Placeholder function for chatbot toggle in navigation
  // The actual chatbot is handled by the ChatbotButton component
  const handleChatbotToggle = () => {
    // This could be used to trigger the chatbot from the navigation
    // For now, it's a placeholder since ChatbotButton manages its own state
    console.log('Chatbot toggle from navigation')
  }

  return (
    <div className="min-h-screen bg-neutral-bg dark:bg-slate-900 flex flex-col">
      <SkipToMain />
      <GlobalNavigation 
        user={user}
        onChatbotToggle={handleChatbotToggle}
      />
      <motion.main
        id="main-content"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{
          duration: 0.3,
          ease: [0.4, 0, 0.2, 1]
        }}
        className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 flex-1"
        role="main"
        aria-label="Main content"
        tabIndex="-1"
        style={{
          willChange: 'transform, opacity',
          backfaceVisibility: 'hidden'
        }}
      >
        {children}
      </motion.main>
      <Footer />
      
      {/* Chatbot Integration */}
      <ChatbotButton />
    </div>
  )
}

export default Layout
