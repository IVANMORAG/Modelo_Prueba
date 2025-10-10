import { motion } from 'framer-motion';
import { Loader2, BarChart3 } from 'lucide-react';

/**
 * LoadingSpinner - Spinner elegante con variantes (SOLID: OCP)
 */
const LoadingSpinner = ({ 
  variant = 'default', 
  size = 'md', 
  message = 'Cargando...',
  fullScreen = false 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16',
  };

  // Variant: Default spinner
  if (variant === 'default') {
    const content = (
      <div className="flex flex-col items-center justify-center gap-4">
        <Loader2 className={`${sizeClasses[size]} text-blue-500 animate-spin`} />
        {message && (
          <p className="text-sm text-gray-600 dark:text-gray-400 animate-pulse">
            {message}
          </p>
        )}
      </div>
    );

    if (fullScreen) {
      return (
        <div className="fixed inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-50">
          {content}
        </div>
      );
    }

    return content;
  }

  // Variant: Dots
  if (variant === 'dots') {
    return (
      <div className="flex items-center gap-2">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="w-3 h-3 bg-blue-500 rounded-full"
            animate={{
              scale: [1, 1.5, 1],
              opacity: [1, 0.5, 1],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              delay: i * 0.2,
            }}
          />
        ))}
      </div>
    );
  }

  // Variant: Pulse (para skeleton)
  if (variant === 'pulse') {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
      </div>
    );
  }

  // Variant: Chart (para cuando cargan gráficas)
  if (variant === 'chart') {
    const content = (
      <div className="flex flex-col items-center justify-center gap-4 p-8">
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 5, -5, 0],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
          }}
        >
          <BarChart3 className={`${sizeClasses[size]} text-blue-500`} />
        </motion.div>
        <div className="flex gap-1">
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={i}
              className="w-2 bg-blue-500 rounded-t"
              animate={{
                height: ['20px', '40px', '20px'],
              }}
              transition={{
                duration: 1,
                repeat: Infinity,
                delay: i * 0.1,
              }}
            />
          ))}
        </div>
        {message && (
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {message}
          </p>
        )}
      </div>
    );

    if (fullScreen) {
      return (
        <div className="fixed inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-50">
          {content}
        </div>
      );
    }

    return content;
  }

  // Variant: Progress bar
  if (variant === 'progress') {
    return (
      <div className="w-full space-y-2">
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
          <span>{message}</span>
          <motion.span
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            ...
          </motion.span>
        </div>
        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-600"
            animate={{
              x: ['-100%', '100%'],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'linear',
            }}
          />
        </div>
      </div>
    );
  }

  return <LoadingSpinner variant="default" size={size} message={message} />;
};

export default LoadingSpinner;