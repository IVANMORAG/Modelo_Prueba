"""
Repositorio de Datos
Implementa Dependency Inversion Principle (DIP) - Abstracción de acceso a datos
"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

from config.database import db_manager
from utils.logger import logger
from utils.exceptions import DatabaseConnectionException


class DataRepository:
    """Repositorio para acceso a datos de compras"""
    
    def __init__(self):
        self.db = db_manager
    
    def get_compras_data(
        self,
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None,
        origen: Optional[str] = None,
        empresa: Optional[str] = None,
        filtros_adicionales: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Obtiene datos de compras con filtros opcionales
        
        Args:
            fecha_inicio: Fecha inicio (formato 'YYYY-MM-DD')
            fecha_fin: Fecha fin (formato 'YYYY-MM-DD')
            origen: Filtrar por ORIGEN
            empresa: Filtrar por EMPRESA
            filtros_adicionales: Filtros adicionales
            
        Returns:
            DataFrame con datos de compras
        """
        logger.info("📥 Obteniendo datos de compras desde base de datos...")
        
        try:
            # Construir query base
            query = self._build_base_query()
            
            # Agregar filtros
            where_clauses = []
            params = {}
            
            if fecha_inicio and fecha_fin:
                where_clauses.append(
                    "(A.FECHAPEDIDO BETWEEN :fecha_inicio AND :fecha_fin)"
                )
                params['fecha_inicio'] = fecha_inicio
                params['fecha_fin'] = fecha_fin
            
            if origen:
                where_clauses.append("A.ORIGEN = :origen")
                params['origen'] = origen
            
            if empresa:
                where_clauses.append("A.EMPRESA = :empresa")
                params['empresa'] = empresa
            
            # Aplicar filtros adicionales
            if filtros_adicionales:
                for key, value in filtros_adicionales.items():
                    where_clauses.append(f"A.{key} = :{key}")
                    params[key] = value
            
            # Agregar WHERE si hay filtros
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            # Agregar GROUP BY
            query += self._get_group_by_clause()
            
            # Ejecutar query
            df = self.db.execute_query(query, params)
            
            logger.info(f" Datos obtenidos: {len(df):,} registros")
            
            return df
            
        except Exception as e:
            logger.error(f" Error al obtener datos: {e}")
            raise DatabaseConnectionException(str(e))
    
    def get_compras_año(self, anio: int) -> pd.DataFrame:
        """
        Obtiene todos los datos de un año específico
        
        Args:
            anio: Año a consultar
            
        Returns:
            DataFrame con datos del año
        """
        fecha_inicio = f"{anio}-01-01"
        fecha_fin = f"{anio}-12-31"
        
        return self.get_compras_data(
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin
        )
    
    def get_compras_rango_años(
        self,
        anio_inicio: int,
        anio_fin: int
    ) -> pd.DataFrame:
        """
        Obtiene datos de un rango de años
        
        Args:
            anio_inicio: Año inicial
            anio_fin: Año final
            
        Returns:
            DataFrame con datos del rango
        """
        fecha_inicio = f"{anio_inicio}-01-01"
        fecha_fin = f"{anio_fin}-12-31"
        
        return self.get_compras_data(
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin
        )
    
    def _build_base_query(self) -> str:
        """Construye query base SQL"""
        return """
            SELECT 
                A.ORIGEN,
                A.SISTEMA,
                A.EMPRESA,
                D.UN, 
                A.FECHASOLICITUD, 
                A.FECHAPEDIDO,
                A.CENTROCOSTO,
                D.DESTINO,
                D.NIVEL6, 
                D.NIVEL7, 
                D.NIVEL8, 
                D.NATURALEZA,
                A.UNEGOCIO,
                SUM(A.TOTALPESOS) AS TOTALPESOS,
                A.DETALLEFINAL, 
                A.SOLICITANTE,
                A.NEGOCIADORCOMPRADOR AS PREPARADOR,
                A.SOLICITUD, 
                A.POSICION, 
                CONCAT(A.SOLICITUD, A.POSICION) AS CONCATENADO, 
                A.CATEGORIA, 
                A.CLASE, 
                A.FAMILIA, 
                A.[SOCIEDAD COMPRADORA]
            FROM NSC_Temporal_SPPI_2025 AS A 
            LEFT JOIN CAT_EKT_TVA_TPE D 
                ON D.IdCC = A.CENTROCOSTO 
                AND D.UN = A.EMPRESA
        """
    
    def _get_group_by_clause(self) -> str:
        """Retorna cláusula GROUP BY"""
        return """
            GROUP BY 
                A.ORIGEN, 
                A.SISTEMA,
                D.UN,
                A.EMPRESA, 
                A.FECHASOLICITUD, 
                A.FECHAPEDIDO,
                A.CENTROCOSTO,
                D.DESTINO,
                D.NIVEL6, 
                D.NIVEL7, 
                D.NIVEL8, 
                D.NATURALEZA,
                A.UNEGOCIO,
                A.DETALLEFINAL, 
                A.SOLICITANTE,
                A.NEGOCIADORCOMPRADOR,
                A.SOLICITUD, 
                A.POSICION, 
                A.CATEGORIA, 
                A.CLASE, 
                A.FAMILIA, 
                A.[SOCIEDAD COMPRADORA]
        """
    
    def test_connection(self) -> bool:
        """Prueba la conexión a la base de datos"""
        return self.db.test_connection()


# Singleton del repositorio
data_repository = DataRepository()