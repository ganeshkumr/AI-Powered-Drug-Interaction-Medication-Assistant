import { motion } from 'framer-motion'
import Header from './Header'
import Footer from './Footer'

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen bg-neutral-bg dark:bg-slate-900 flex flex-col">
      <Header />
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 flex-1"
      >
        {children}
      </motion.main>
      <Footer />
    </div>
  )
}

export default Layout