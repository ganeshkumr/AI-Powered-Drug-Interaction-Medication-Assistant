import { motion } from 'framer-motion'
import HealthMonitoring from '../components/dashboard/HealthMonitoring'
import EmergencyCheck from '../components/dashboard/EmergencyCheck'
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
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8"
    >
      <motion.div variants={itemVariants}>
        <HealthMonitoring />
      </motion.div>

      <motion.div variants={itemVariants}>
        <EmergencyCheck />
      </motion.div>

      <motion.div variants={itemVariants}>
        <MedicationForm />
      </motion.div>

      <motion.div variants={itemVariants}>
        <MedicationList />
      </motion.div>
    </motion.div>
  )
}

export default Dashboard