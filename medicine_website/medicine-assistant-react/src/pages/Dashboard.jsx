import { motion } from 'framer-motion'
import MedicationForm from '../components/dashboard/MedicationForm'
import MedicationList from '../components/dashboard/MedicationList'

const Dashboard = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5 }
    }
  }

  return (
    <div>
      {/* Welcome Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-heading font-bold text-neutral-text mb-2">
          Welcome to Your Health Dashboard
        </h1>
        <p className="text-gray-600">
          Monitor your health, check medication safety, and manage your prescriptions all in one place.
        </p>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-6"
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div variants={itemVariants}>
            <MedicationForm />
          </motion.div>

          <motion.div variants={itemVariants}>
            <MedicationList />
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}

export default Dashboard