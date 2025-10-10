"""
Motor de Recomendaciones Estratégicas
Implementa Single Responsibility Principle (SRP)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from models.schemas import (
    RecommendationResponse, Recomendacion, TendenciaDimension
)
from models.enums import NivelSeveridad, DimensionAnalisis
from utils.logger import logger


class RecommendationEngine:
    """Motor para generar recomendaciones estratégicas basadas en análisis de datos"""
    
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        dimensiones_analisis: List[DimensionAnalisis],
        incluir_tendencias: bool = True
    ) -> RecommendationResponse:
        """
        Genera recomendaciones estratégicas
        
        Args:
            df: DataFrame con datos históricos
            dimensiones_analisis: Dimensiones a analizar
            incluir_tendencias: Si incluir análisis de tendencias
            
        Returns:
            RecommendationResponse con recomendaciones
        """
        logger.info(" Generando recomendaciones estratégicas...")
        
        # Análisis por ORIGEN
        analisis_origen = self._analyze_origen(df)
        
        # Análisis por UN
        analisis_un = self._analyze_un(df)
        
        # Generar recomendaciones principales
        recomendaciones = self._generate_main_recommendations(
            df, analisis_origen, analisis_un
        )
        
        # Análisis de tendencias
        tendencias = []
        if incluir_tendencias:
            tendencias = self._analyze_trends(df, dimensiones_analisis)
        
        # Identificar oportunidades de ahorro
        oportunidades_ahorro = self._identify_saving_opportunities(
            df, analisis_origen, analisis_un
        )
        
        # Generar resumen ejecutivo
        resumen_ejecutivo = self._generate_executive_summary(
            df, recomendaciones, oportunidades_ahorro
        )
        
        response = RecommendationResponse(
            recomendaciones=recomendaciones,
            analisis_origen=analisis_origen,
            analisis_un=analisis_un,
            tendencias=tendencias,
            oportunidades_ahorro=oportunidades_ahorro,
            resumen_ejecutivo=resumen_ejecutivo
        )
        
        logger.info(f" {len(recomendaciones)} recomendaciones generadas")
        
        return response
    
    def _analyze_origen(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analiza gastos por ORIGEN"""
        if 'ORIGEN' not in df.columns:
            return {}
        
        analisis = {}
        
        for origen in df['ORIGEN'].unique():
            df_origen = df[df['ORIGEN'] == origen]
            
            # Calcular cuartiles para clasificar gasto
            q25 = df['TOTALPESOS'].quantile(0.25)
            q75 = df['TOTALPESOS'].quantile(0.75)
            
            gasto_promedio = float(df_origen['TOTALPESOS'].mean())
            gasto_total = float(df_origen['TOTALPESOS'].sum())
            total_pedidos = len(df_origen)
            
            # Clasificar gasto
            if gasto_promedio > q75:
                clasificacion = "alto"
            elif gasto_promedio < q25:
                clasificacion = "bajo"
            else:
                clasificacion = "normal"
            
            analisis[origen] = {
                'gasto_total': gasto_total,
                'gasto_promedio': gasto_promedio,
                'total_pedidos': total_pedidos,
                'clasificacion_gasto': clasificacion,
                'top_categorias': df_origen.groupby('CATEGORIA')['TOTALPESOS'].sum().nlargest(3).to_dict()
                if 'CATEGORIA' in df_origen.columns else {}
            }
        
        return analisis
    
    def _analyze_un(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analiza gastos por Unidad de Negocio"""
        if 'UN' not in df.columns:
            return {}
        
        analisis = {}
        
        df_un = df[df['UN'].notna()]
        
        for un in df_un['UN'].unique():
            df_unidad = df_un[df_un['UN'] == un]
            
            gasto_promedio = float(df_unidad['TOTALPESOS'].mean())
            gasto_total = float(df_unidad['TOTALPESOS'].sum())
            total_pedidos = len(df_unidad)
            
            analisis[un] = {
                'gasto_total': gasto_total,
                'gasto_promedio': gasto_promedio,
                'total_pedidos': total_pedidos,
                'top_categorias': df_unidad.groupby('CATEGORIA')['TOTALPESOS'].sum().nlargest(3).to_dict()
                if 'CATEGORIA' in df_unidad.columns else {}
            }
        
        return analisis
    
    def _generate_main_recommendations(
        self,
        df: pd.DataFrame,
        analisis_origen: Dict[str, Dict[str, Any]],
        analisis_un: Dict[str, Dict[str, Any]]
    ) -> List[Recomendacion]:
        """Genera recomendaciones principales"""
        recomendaciones = []
        
        # Recomendación 1: ORIGEN con gasto alto
        gasto_promedio_global = df['TOTALPESOS'].mean()
        
        for origen, data in analisis_origen.items():
            if data['clasificacion_gasto'] == "alto":
                recomendaciones.append(Recomendacion(
                    titulo=f"Optimizar gastos en {origen}",
                    descripcion=f"El origen {origen} presenta un gasto promedio de "
                               f"${data['gasto_promedio']:,.2f}, superior al promedio global. "
                               f"Total acumulado: ${data['gasto_total']:,.2f}",
                    prioridad=NivelSeveridad.ALTO,
                    categoria="Optimización de Costos",
                    impacto_estimado=f"Ahorro potencial: ${data['gasto_total'] * 0.1:,.2f} (10%)",
                    acciones_sugeridas=[
                        "Revisar contratos con proveedores y negociar descuentos por volumen",
                        "Consolidar compras para obtener mejores precios",
                        "Implementar proceso de validación adicional para compras >$50k",
                        "Comparar precios con otros ORIGENes de menor costo"
                    ]
                ))
        
        # Recomendación 2: Alto volumen de pedidos
        for origen, data in analisis_origen.items():
            percentil_75_pedidos = df.groupby('ORIGEN').size().quantile(0.75)
            
            if data['total_pedidos'] > percentil_75_pedidos:
                recomendaciones.append(Recomendacion(
                    titulo=f"Consolidar pedidos en {origen}",
                    descripcion=f"{origen} tiene {data['total_pedidos']:,} pedidos, "
                               f"uno de los volúmenes más altos. La consolidación puede "
                               f"reducir costos operativos.",
                    prioridad=NivelSeveridad.MEDIO,
                    categoria="Eficiencia Operacional",
                    impacto_estimado="Reducción de costos administrativos del 15-20%",
                    acciones_sugeridas=[
                        "Implementar compras programadas mensuales en lugar de diarias",
                        "Establecer montos mínimos por pedido",
                        "Crear calendario de compras consolidadas",
                        "Automatizar pedidos recurrentes"
                    ]
                ))
        
        # Recomendación 3: UN con gasto elevado
        if analisis_un:
            for un, data in sorted(
                analisis_un.items(), 
                key=lambda x: x[1]['gasto_promedio'], 
                reverse=True
            )[:3]:  # Top 3 UNs
                
                if data['gasto_promedio'] > gasto_promedio_global * 1.5:
                    recomendaciones.append(Recomendacion(
                        titulo=f"Revisar procesos de aprobación en {un}",
                        descripcion=f"La unidad {un} tiene un gasto promedio 50% superior "
                                   f"al global (${data['gasto_promedio']:,.2f}). "
                                   f"Total: ${data['gasto_total']:,.2f}",
                        prioridad=NivelSeveridad.ALTO,
                        categoria="Gobernanza",
                        impacto_estimado=f"Reducción potencial: ${data['gasto_total'] * 0.15:,.2f}",
                        acciones_sugeridas=[
                            "Revisar niveles de autorización para aprobaciones",
                            "Implementar proceso de validación de necesidad",
                            "Buscar alternativas más económicas",
                            "Establecer presupuestos por categoría"
                        ]
                    ))
        
        # Recomendación 4: Categorías de alto impacto
        if 'CATEGORIA' in df.columns:
            top_categorias = df.groupby('CATEGORIA')['TOTALPESOS'].sum().nlargest(3)
            
            for categoria, gasto in top_categorias.items():
                recomendaciones.append(Recomendacion(
                    titulo=f"Estrategia de compra para {categoria}",
                    descripcion=f"{categoria} representa ${gasto:,.2f} del gasto total. "
                               f"Optimizar esta categoría tiene alto impacto.",
                    prioridad=NivelSeveridad.ALTO,
                    categoria="Categorización Estratégica",
                    impacto_estimado=f"Impacto potencial: ${gasto * 0.08:,.2f} (8% ahorro)",
                    acciones_sugeridas=[
                        f"Desarrollar estrategia específica para {categoria}",
                        "Identificar proveedores preferentes",
                        "Estandarizar especificaciones técnicas",
                        "Implementar contratos marco"
                    ]
                ))
        
        # Recomendación 5: Bajo volumen de compras
        for un, data in analisis_un.items():
            if data['total_pedidos'] < 100:
                recomendaciones.append(Recomendacion(
                    titulo=f"Consolidar compras de {un} con otras unidades",
                    descripcion=f"{un} tiene solo {data['total_pedidos']} pedidos. "
                               f"Consolidar con otras UNs puede mejorar poder de negociación.",
                    prioridad=NivelSeveridad.MEDIO,
                    categoria="Sinergia Organizacional",
                    impacto_estimado="Descuentos por volumen del 5-10%",
                    acciones_sugeridas=[
                        "Identificar necesidades comunes con otras UNs",
                        "Crear comité de compras inter-unidades",
                        "Negociar contratos corporativos",
                        "Compartir mejores prácticas"
                    ]
                ))
        
        # Limitar a top 10 recomendaciones
        return recomendaciones[:10]
    
    def _analyze_trends(
        self,
        df: pd.DataFrame,
        dimensiones: List[DimensionAnalisis]
    ) -> List[TendenciaDimension]:
        """Analiza tendencias por dimensión"""
        tendencias = []
        
        if 'FECHAPEDIDO' not in df.columns:
            return tendencias
        
        df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'])
        df['AÑO_MES'] = df['FECHAPEDIDO'].dt.to_period('M')
        
        for dimension in dimensiones:
            dim_col = dimension.value
            
            if dim_col not in df.columns:
                continue
            
            # Analizar últimos 6 meses vs 6 meses anteriores
            periodos = sorted(df['AÑO_MES'].unique())
            
            if len(periodos) < 12:
                continue
            
            periodo_actual = periodos[-6:]
            periodo_anterior = periodos[-12:-6]
            
            df_actual = df[df['AÑO_MES'].isin(periodo_actual)]
            df_anterior = df[df['AÑO_MES'].isin(periodo_anterior)]
            
            # Top 5 valores de la dimensión
            top_valores = df[dim_col].value_counts().head(5).index
            
            for valor in top_valores:
                gasto_actual = df_actual[df_actual[dim_col] == valor]['TOTALPESOS'].sum()
                gasto_anterior = df_anterior[df_anterior[dim_col] == valor]['TOTALPESOS'].sum()
                
                if gasto_anterior == 0:
                    continue
                
                variacion = ((gasto_actual - gasto_anterior) / gasto_anterior) * 100
                
                if abs(variacion) > 10:  # Solo tendencias significativas
                    tendencia_str = "creciente" if variacion > 0 else "decreciente"
                    
                    tendencias.append(TendenciaDimension(
                        dimension=str(dim_col),
                        valor=str(valor),
                        tendencia=tendencia_str,
                        variacion_porcentual=float(variacion),
                        gasto_actual=float(gasto_actual),
                        gasto_anterior=float(gasto_anterior)
                    ))
        
        return sorted(tendencias, key=lambda x: abs(x.variacion_porcentual), reverse=True)[:10]
    
    def _identify_saving_opportunities(
        self,
        df: pd.DataFrame,
        analisis_origen: Dict[str, Dict[str, Any]],
        analisis_un: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identifica oportunidades específicas de ahorro"""
        oportunidades = []
        
        # Oportunidad 1: Proveedores caros
        if 'ORIGEN' in df.columns and analisis_origen:
            gasto_promedio_global = df['TOTALPESOS'].mean()
            
            for origen, data in analisis_origen.items():
                if data['gasto_promedio'] > gasto_promedio_global * 1.3:
                    ahorro_estimado = data['gasto_total'] * 0.15
                    
                    oportunidades.append({
                        'tipo': 'Negociación de Proveedores',
                        'dimension': 'ORIGEN',
                        'valor': origen,
                        'descripcion': f"Renegociar precios con {origen}",
                        'gasto_actual': data['gasto_total'],
                        'ahorro_estimado': ahorro_estimado,
                        'porcentaje_ahorro': 15.0,
                        'complejidad': 'Media',
                        'plazo_implementacion': '3-6 meses'
                    })
        
        # Oportunidad 2: Consolidación de pedidos pequeños
        if 'TOTALPESOS' in df.columns:
            pedidos_pequeños = df[df['TOTALPESOS'] < df['TOTALPESOS'].quantile(0.25)]
            gasto_total_pequeños = pedidos_pequeños['TOTALPESOS'].sum()
            
            if len(pedidos_pequeños) > 1000:
                ahorro_estimado = gasto_total_pequeños * 0.10
                
                oportunidades.append({
                    'tipo': 'Consolidación de Compras',
                    'dimension': 'OPERACIONAL',
                    'valor': 'Pedidos pequeños',
                    'descripcion': f"Consolidar {len(pedidos_pequeños):,} pedidos pequeños",
                    'gasto_actual': gasto_total_pequeños,
                    'ahorro_estimado': ahorro_estimado,
                    'porcentaje_ahorro': 10.0,
                    'complejidad': 'Baja',
                    'plazo_implementacion': '1-3 meses'
                    })
        
        # Ordenar por ahorro estimado
        return sorted(oportunidades, key=lambda x: x['ahorro_estimado'], reverse=True)[:5]
    
    def _generate_executive_summary(
        self,
        df: pd.DataFrame,
        recomendaciones: List[Recomendacion],
        oportunidades: List[Dict[str, Any]]
    ) -> str:
        """Genera resumen ejecutivo"""
        gasto_total = df['TOTALPESOS'].sum()
        total_pedidos = len(df)
        gasto_promedio = df['TOTALPESOS'].mean()
        
        ahorro_total_estimado = sum(op['ahorro_estimado'] for op in oportunidades)
        
        resumen = f"""
 RESUMEN EJECUTIVO - ANÁLISIS DE COMPRAS

🔹 Situación Actual:
   • Gasto Total Analizado: ${gasto_total:,.2f}
   • Total de Pedidos: {total_pedidos:,}
   • Gasto Promedio por Pedido: ${gasto_promedio:,.2f}

 Oportunidades Identificadas:
   • {len(recomendaciones)} recomendaciones estratégicas generadas
   • {len(oportunidades)} oportunidades de ahorro cuantificadas
   • Ahorro Potencial Estimado: ${ahorro_total_estimado:,.2f} ({ahorro_total_estimado/gasto_total*100:.2f}% del gasto total)

 Acciones Prioritarias:
"""
        
        # Agregar top 3 recomendaciones críticas/altas
        top_recs = [r for r in recomendaciones if r.prioridad in [NivelSeveridad.CRITICO, NivelSeveridad.ALTO]][:3]
        
        for i, rec in enumerate(top_recs, 1):
            resumen += f"\n   {i}. {rec.titulo}"
        
        resumen += f"\n\n Próximos Pasos: Implementar las {len(top_recs)} acciones prioritarias identificadas para maximizar el impacto en los próximos 3-6 meses."
        
        return resumen.strip()


# Singleton del motor de recomendaciones
recommendation_engine = RecommendationEngine()