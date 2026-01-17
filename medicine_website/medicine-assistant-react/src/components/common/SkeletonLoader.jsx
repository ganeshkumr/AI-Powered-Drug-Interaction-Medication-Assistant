import { motion } from 'framer-motion'

const SkeletonLoader = ({ type = 'card', count = 1 }) => {
  const skeletons = Array.from({ length: count }, (_, i) => i)

  const CardSkeleton = () => (
    <div className="bg-white rounded-card-lg border border-gray-200 p-6 space-y-4">
      <div className="flex items-start justify-between">
        <div className="flex-1 space-y-3">
          <motion.div
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="h-6 bg-gray-200 rounded w-3/4"
          />
          <motion.div
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
            className="h-4 bg-gray-200 rounded w-1/2"
          />
        </div>
        <motion.div
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity, delay: 0.1 }}
          className="w-12 h-12 bg-gray-200 rounded-full"
        />
      </div>
      <motion.div
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity, delay: 0.3 }}
        className="h-4 bg-gray-200 rounded w-full"
      />
      <motion.div
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }}
        className="h-4 bg-gray-200 rounded w-5/6"
      />
    </div>
  )

  const ListSkeleton = () => (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <motion.div
          key={i}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.1 }}
          className="h-16 bg-gray-200 rounded-lg"
        />
      ))}
    </div>
  )

  const TextSkeleton = () => (
    <div className="space-y-2">
      <motion.div
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity }}
        className="h-4 bg-gray-200 rounded w-full"
      />
      <motion.div
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity, delay: 0.1 }}
        className="h-4 bg-gray-200 rounded w-5/6"
      />
      <motion.div
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
        className="h-4 bg-gray-200 rounded w-4/6"
      />
    </div>
  )

  const renderSkeleton = () => {
    switch (type) {
      case 'card':
        return <CardSkeleton />
      case 'list':
        return <ListSkeleton />
      case 'text':
        return <TextSkeleton />
      default:
        return <CardSkeleton />
    }
  }

  return (
    <div className="space-y-4">
      {skeletons.map((i) => (
        <div key={i}>{renderSkeleton()}</div>
      ))}
    </div>
  )
}

export default SkeletonLoader
