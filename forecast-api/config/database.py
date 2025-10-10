"""
Gestión de Conexiones a Base de Datos usando OLEDB (como Excel)
"""
from contextlib import contextmanager
from typing import Generator
import pandas as pd
from config.settings import settings
from utils.logger import logger
import win32com.client

class DatabaseManager:
    """Gestor de conexiones a la base de datos SQL Server usando OLEDB"""
    
    def __init__(self):
        self.connection_string = settings.oledb_connection_string
        logger.info(" Motor de base de datos inicializado correctamente")
    
    def _get_connection(self):
        """Obtiene una nueva conexión OLEDB"""
        conn = win32com.client.Dispatch("ADODB.Connection")
        conn.Open(self.connection_string)
        return conn
    
    @contextmanager
    def get_connection(self):
        """Context manager para conexiones de base de datos"""
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        except Exception as e:
            logger.error(f" Error en conexión de base de datos: {e}")
            raise
        finally:
            if conn:
                conn.Close()
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Ejecuta una consulta SQL y retorna un DataFrame de pandas
        
        Args:
            query: Consulta SQL a ejecutar
        
        Returns:
            DataFrame con los resultados
        """
        try:
            with self.get_connection() as conn:
                recordset = win32com.client.Dispatch("ADODB.Recordset")
                recordset.Open(query, conn)
                
                # Obtener nombres de columnas
                columns = [field.Name for field in recordset.Fields]
                
                # Obtener datos
                data = []
                while not recordset.EOF:
                    row = [recordset.Fields(i).Value for i in range(len(columns))]
                    data.append(row)
                    recordset.MoveNext()
                
                recordset.Close()
                
                df = pd.DataFrame(data, columns=columns)
                logger.info(f" Consulta ejecutada: {len(df)} registros obtenidos")
                return df
                
        except Exception as e:
            logger.error(f" Error al ejecutar consulta: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Prueba la conexión a la base de datos"""
        try:
            with self.get_connection() as conn:
                recordset = win32com.client.Dispatch("ADODB.Recordset")
                recordset.Open("SELECT 1", conn)
                recordset.Close()
                logger.info(" Conexión a base de datos exitosa")
                return True
        except Exception as e:
            logger.error(f" Error al conectar a la base de datos: {e}")
            return False

# Singleton del gestor de base de datos
db_manager = DatabaseManager()

def get_db():
    """Dependency para FastAPI - Provee conexión de base de datos"""
    with db_manager.get_connection() as conn:
        yield conn