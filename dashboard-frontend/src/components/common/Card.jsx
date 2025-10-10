import { motion } from 'framer-motion';

/**
 * Card - Componente de tarjeta base (SOLID: OCP)
 */
const Card = ({ 
  children, 
  className = '', 
  hover = true,
  padding = 'p-6',
  ...props 
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`
        bg-white dark:bg-gray-800 
        rounded-xl shadow-lg
        ${hover ? 'hover:shadow-xl transition-shadow duration-300' : ''}
        ${padding}
        ${className}
      `}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export default Card;