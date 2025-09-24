import pandas as pd
import numpy as np
from collections import defaultdict
import json

def analizar_patrones_faltantes(df, stats):
    """Analiza los patrones de datos faltantes en categoría, clase y familia."""
    print("="*60)
    print("ANÁLISIS DE PATRONES DE DATOS FALTANTES")
    print("="*60)
    
    cat_vacia = df['CATEGORIA'].isna() | (df['CATEGORIA'].str.strip() == '') | (df['CATEGORIA'] == 'Sin Categoría')
    clase_vacia = df['CLASE'].isna() | (df['CLASE'].str.strip() == '')
    familia_vacia = df['FAMILIA'].isna() | (df['FAMILIA'].str.strip() == '')
    
    total_registros = len(df)
    
    patrones = {
        'Todos completos': (~cat_vacia & ~clase_vacia & ~familia_vacia).sum(),
        'Solo CATEGORIA': (~cat_vacia & clase_vacia & familia_vacia).sum(),
        'Solo CLASE': (cat_vacia & ~clase_vacia & familia_vacia).sum(),
        'Solo FAMILIA': (cat_vacia & clase_vacia & ~familia_vacia).sum(),
        'CATEGORIA + CLASE': (~cat_vacia & ~clase_vacia & familia_vacia).sum(),
        'CATEGORIA + FAMILIA': (~cat_vacia & clase_vacia & ~familia_vacia).sum(),
        'CLASE + FAMILIA': (cat_vacia & ~clase_vacia & ~familia_vacia).sum(),
        'Todos vacíos': (cat_vacia & clase_vacia & familia_vacia).sum()
    }
    
    print(f"Total de registros: {total_registros:,}")
    print("\nDistribución de patrones:")
    for patron, cantidad in patrones.items():
        porcentaje = (cantidad / total_registros) * 100
        print(f"  {patron:<20}: {cantidad:>8,} ({porcentaje:>5.1f}%)")
    
    print("\nValores únicos:")
    print(f"  CATEGORIA: {df['CATEGORIA'].nunique():,} valores únicos")
    print(f"  CLASE: {df['CLASE'].nunique():,} valores únicos") 
    print(f"  FAMILIA: {df['FAMILIA'].nunique():,} valores únicos")
    
    stats['patrones'] = patrones
    stats['total_registros'] = total_registros
    
    return patrones, stats

def construir_diccionarios_mapeo(df):
    """Construye diccionarios de mapeo entre categoría, clase y familia."""
    print("\n" + "="*60)
    print("CONSTRUYENDO DICCIONARIOS DE MAPEO")
    print("="*60)
    
    df_completo = df.dropna(subset=['CATEGORIA', 'CLASE', 'FAMILIA'])
    df_completo = df_completo[
        (df_completo['CATEGORIA'].str.strip() != '') & 
        (df_completo['CLASE'].str.strip() != '') & 
        (df_completo['FAMILIA'].str.strip() != '') &
        (df_completo['CATEGORIA'] != 'Sin Categoría')
    ]
    
    print(f"Registros con datos completos para mapeo: {len(df_completo):,}")
    
    mapeo_categoria_clase = defaultdict(set)
    mapeo_categoria_familia = defaultdict(set)
    mapeo_clase_familia = defaultdict(set)
    mapeo_familia_categoria = defaultdict(set)
    mapeo_familia_clase = defaultdict(set)
    mapeo_clase_categoria = defaultdict(set)

    for _, row in df_completo.iterrows():
        cat = str(row['CATEGORIA']).strip()
        clase = str(row['CLASE']).strip()
        familia = str(row['FAMILIA']).strip()
        
        mapeo_categoria_clase[cat].add(clase)
        mapeo_categoria_familia[cat].add(familia)
        mapeo_clase_categoria[clase].add(cat)
        mapeo_clase_familia[clase].add(familia)
        mapeo_familia_categoria[familia].add(cat)
        mapeo_familia_clase[familia].add(clase)

    mapeos = {
        'mapeo_categoria_clase': {k: list(v) for k, v in mapeo_categoria_clase.items()},
        'mapeo_categoria_familia': {k: list(v) for k, v in mapeo_categoria_familia.items()},
        'mapeo_clase_categoria': {k: list(v) for k, v in mapeo_clase_categoria.items()},
        'mapeo_clase_familia': {k: list(v) for k, v in mapeo_clase_familia.items()},
        'mapeo_familia_categoria': {k: list(v) for k, v in mapeo_familia_categoria.items()},
        'mapeo_familia_clase': {k: list(v) for k, v in mapeo_familia_clase.items()}
    }
    
    print(f"Mapeos construidos:")
    print(f"  Categorías únicas en mapeo: {len(mapeos['mapeo_categoria_clase'])}")
    print(f"  Clases únicas en mapeo: {len(mapeos['mapeo_clase_categoria'])}")
    print(f"  Familias únicas en mapeo: {len(mapeos['mapeo_familia_categoria'])}")
    
    avg_clases_por_cat = np.mean([len(v) for v in mapeos['mapeo_categoria_clase'].values()])
    avg_familias_por_cat = np.mean([len(v) for v in mapeos['mapeo_categoria_familia'].values()])
    print("\nEstadísticas de relaciones:")
    print(f"  Promedio clases por categoría: {avg_clases_por_cat:.1f}")
    print(f"  Promedio familias por categoría: {avg_familias_por_cat:.1f}")
    
    return mapeos

def mostrar_mapeos_principales(mapeos, top_n=10):
    """Muestra los principales mapeos encontrados."""
    print("\n" + "="*60)
    print("PRINCIPALES MAPEOS ENCONTRADOS")
    print("="*60)
    
    print(f"\nTOP {top_n} CATEGORÍAS con más clases:")
    cat_clases = [(cat, len(clases)) for cat, clases in mapeos['mapeo_categoria_clase'].items()]
    for cat, num_clases in sorted(cat_clases, key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"  {cat}: {num_clases} clases")
    
    print(f"\nTOP {top_n} CATEGORÍAS con más familias:")
    cat_familias = [(cat, len(familias)) for cat, familias in mapeos['mapeo_categoria_familia'].items()]
    for cat, num_familias in sorted(cat_familias, key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"  {cat}: {num_familias} familias")

def rellenar_valores_faltantes(df, mapeos):
    """Rellena valores faltantes usando los mapeos construidos."""
    print("\n" + "="*60)
    print("RELLENANDO VALORES FALTANTES")
    print("="*60)
    
    df_limpio = df.copy()
    
    cambios = {
        'categoria_desde_clase': 0, 'categoria_desde_familia': 0,
        'clase_desde_categoria': 0, 'clase_desde_familia': 0,
        'familia_desde_categoria': 0, 'familia_desde_clase': 0
    }
    
    total_filas = len(df_limpio)
    
    for idx, row in df_limpio.iterrows():
        if idx % 100000 == 0:
            print(f"  Procesando: {idx:,}/{total_filas:,} ({idx/total_filas*100:.1f}%)")
        
        cat = str(row['CATEGORIA']).strip() if pd.notna(row['CATEGORIA']) else ''
        clase = str(row['CLASE']).strip() if pd.notna(row['CLASE']) else ''
        familia = str(row['FAMILIA']).strip() if pd.notna(row['FAMILIA']) else ''
        
        cat_vacia = cat == '' or cat == 'Sin Categoría' or pd.isna(row['CATEGORIA'])
        clase_vacia = clase == '' or pd.isna(row['CLASE'])
        familia_vacia = familia == '' or pd.isna(row['FAMILIA'])
        
        if cat_vacia and not clase_vacia and clase in mapeos['mapeo_clase_categoria']:
            posibles_cats = mapeos['mapeo_clase_categoria'][clase]
            if len(posibles_cats) == 1:
                df_limpio.at[idx, 'CATEGORIA'] = posibles_cats[0]
                cambios['categoria_desde_clase'] += 1
        
        if cat_vacia and not familia_vacia and familia in mapeos['mapeo_familia_categoria']:
            posibles_cats = mapeos['mapeo_familia_categoria'][familia]
            if len(posibles_cats) == 1:
                df_limpio.at[idx, 'CATEGORIA'] = posibles_cats[0]
                cambios['categoria_desde_familia'] += 1
        
        if clase_vacia and not cat_vacia and cat in mapeos['mapeo_categoria_clase']:
            posibles_clases = mapeos['mapeo_categoria_clase'][cat]
            if len(posibles_clases) == 1:
                df_limpio.at[idx, 'CLASE'] = posibles_clases[0]
                cambios['clase_desde_categoria'] += 1
        
        if clase_vacia and not familia_vacia and familia in mapeos['mapeo_familia_clase']:
            posibles_clases = mapeos['mapeo_familia_clase'][familia]
            if len(posibles_clases) == 1:
                df_limpio.at[idx, 'CLASE'] = posibles_clases[0]
                cambios['clase_desde_familia'] += 1
        
        if familia_vacia and not cat_vacia and cat in mapeos['mapeo_categoria_familia']:
            posibles_familias = mapeos['mapeo_categoria_familia'][cat]
            if len(posibles_familias) == 1:
                df_limpio.at[idx, 'FAMILIA'] = posibles_familias[0]
                cambios['familia_desde_categoria'] += 1
        
        if familia_vacia and not clase_vacia and clase in mapeos['mapeo_clase_familia']:
            posibles_familias = mapeos['mapeo_clase_familia'][clase]
            if len(posibles_familias) == 1:
                df_limpio.at[idx, 'FAMILIA'] = posibles_familias[0]
                cambios['familia_desde_clase'] += 1
    
    print("\nCambios realizados:")
    for tipo, cantidad in cambios.items():
        print(f"  {tipo.replace('_', ' ').title()}: {cantidad:,}")
    
    print(f"\nTotal de campos rellenados: {sum(cambios.values()):,}")
    
    return df_limpio, cambios

def identificar_registros_sin_clasificar(df_limpio):
    """Identifica registros que aún necesitan clasificación manual."""
    print("\n" + "="*60)
    print("IDENTIFICANDO REGISTROS SIN CLASIFICAR")
    print("="*60)
    
    cat_vacia = df_limpio['CATEGORIA'].isna() | (df_limpio['CATEGORIA'].str.strip() == '') | (df_limpio['CATEGORIA'] == 'Sin Categoría')
    clase_vacia = df_limpio['CLASE'].isna() | (df_limpio['CLASE'].str.strip() == '')
    familia_vacia = df_limpio['FAMILIA'].isna() | (df_limpio['FAMILIA'].str.strip() == '')
    
    sin_clasificar_mask = cat_vacia & clase_vacia & familia_vacia
    parcialmente_clasificados_mask = ~sin_clasificar_mask & (cat_vacia | clase_vacia | familia_vacia)
    completamente_clasificados_mask = ~cat_vacia & ~clase_vacia & ~familia_vacia
    
    total = len(df_limpio)
    
    print("Estado después de la limpieza:")
    print(f"  Completamente clasificados: {completamente_clasificados_mask.sum():,} ({completamente_clasificados_mask.sum()/total*100:.1f}%)")
    print(f"  Parcialmente clasificados: {parcialmente_clasificados_mask.sum():,} ({parcialmente_clasificados_mask.sum()/total*100:.1f}%)")
    print(f"  Sin clasificar: {sin_clasificar_mask.sum():,} ({sin_clasificar_mask.sum()/total*100:.1f}%)")
    
    if sin_clasificar_mask.sum() > 0:
        print("\nEjemplos de registros sin clasificar:")
        ejemplos = df_limpio[sin_clasificar_mask].head(5)
        for _, row in ejemplos.iterrows():
            print(f"  - {row['DETALLEFINAL'][:50]}... | Gasto: ${row['TOTALPESOS']:,.2f}")
    
    return df_limpio[sin_clasificar_mask], df_limpio[parcialmente_clasificados_mask]

def analizar_por_años(df_limpio):
    """Analiza la calidad de clasificación por años."""
    print("\n" + "="*60)
    print("ANÁLISIS POR AÑOS")
    print("="*60)
    
    df_años = df_limpio.copy()
    df_años['AÑO'] = pd.to_datetime(df_años['FECHAPEDIDO']).dt.year
    
    cat_vacia = df_años['CATEGORIA'].isna() | (df_años['CATEGORIA'].str.strip() == '') | (df_años['CATEGORIA'] == 'Sin Categoría')
    clase_vacia = df_años['CLASE'].isna() | (df_años['CLASE'].str.strip() == '')
    familia_vacia = df_años['FAMILIA'].isna() | (df_años['FAMILIA'].str.strip() == '')
    
    años_stats = []
    for año in sorted(df_años['AÑO'].dropna().unique()):
        df_año = df_años[df_años['AÑO'] == año]
        
        cat_v = cat_vacia[df_años['AÑO'] == año]
        clase_v = clase_vacia[df_años['AÑO'] == año] 
        familia_v = familia_vacia[df_años['AÑO'] == año]
        
        total = len(df_año)
        completos = (~cat_v & ~clase_v & ~familia_v).sum()
        sin_clasificar = (cat_v & clase_v & familia_v).sum()
        
        años_stats.append({
            'Año': int(año),
            'Total_Registros': total,
            'Completos': completos,
            'Sin_Clasificar': sin_clasificar,
            'Pct_Completos': (completos/total*100) if total > 0 else 0,
            'Pct_Sin_Clasificar': (sin_clasificar/total*100) if total > 0 else 0,
            'Categorias_Unicas': df_año[~cat_v]['CATEGORIA'].nunique(),
            'Gasto_Total': df_año['TOTALPESOS'].sum()
        })
    
    stats_df = pd.DataFrame(años_stats)
    
    print("Evolución por años:")
    print(f"{'Año':<6} {'Registros':<12} {'Completos':<10} {'% Completos':<12} {'% Sin Clasif':<12} {'Categorías':<12}")
    print("-" * 70)
    
    for _, row in stats_df.iterrows():
        print(f"{row['Año']:<6} {row['Total_Registros']:<12,} {row['Completos']:<10,} "
              f"{row['Pct_Completos']:<12.1f} {row['Pct_Sin_Clasificar']:<12.1f} {row['Categorias_Unicas']:<12}")
    
    print("\nTendencias:")
    mejores_años = stats_df.nlargest(3, 'Pct_Completos')
    peores_años = stats_df.nsmallest(3, 'Pct_Completos')
    
    print("Años con mejor clasificación:")
    for _, row in mejores_años.iterrows():
        print(f"  {row['Año']}: {row['Pct_Completos']:.1f}% completos")
        
    print("\nAños con peor clasificación:")
    for _, row in peores_años.iterrows():
        print(f"  {row['Año']}: {row['Pct_Completos']:.1f}% completos")
    
    return stats_df

def guardar_resultados(df_limpio, mapeos, stats, nombre_archivo="datos_limpios_categorias.csv"):
    """Guarda el dataset limpio y los diccionarios de mapeo."""
    print(f"\n" + "="*60)
    print("GUARDANDO RESULTADOS")
    print("="*60)
    
    df_limpio.to_csv(nombre_archivo, index=False)
    print(f"Dataset limpio guardado: {nombre_archivo}")
    
    mapeos_guardar = {
        'mapeos': mapeos,
        'estadisticas': stats
    }
    
    with open('diccionarios_mapeo_categorias.json', 'w', encoding='utf-8') as f:
        json.dump(mapeos_guardar, f, indent=2, ensure_ascii=False)
    
    print("Diccionarios de mapeo y estadísticas guardados: diccionarios_mapeo_categorias.json")
    
    return nombre_archivo

# --- Flujo de ejecución para Jupyter Notebook ---
# Este es el bloque que puedes ejecutar celda por celda

def ejecutar_limpieza_completa(ruta_archivo):
    print("LIMPIEZA Y MAPEO DE CATEGORÍAS - GRUPO SALINAS")
    print("="*60)
    
    # 1. Cargar datos
    print("Cargando datos...")
    df = pd.read_csv(ruta_archivo)
    print(f"Datos cargados: {len(df):,} registros")
    
    # 2. Inicializar diccionarios y estadísticas
    stats = {}
    
    # 3. Analizar patrones iniciales
    patrones, stats = analizar_patrones_faltantes(df, stats)
    
    # 4. Construir diccionarios de mapeo
    mapeos = construir_diccionarios_mapeo(df)
    
    # 5. Mostrar mapeos principales
    mostrar_mapeos_principales(mapeos)
    
    # 6. Rellenar valores faltantes
    df_limpio, cambios = rellenar_valores_faltantes(df, mapeos)
    
    # 7. Análisis por años
    stats_años = analizar_por_años(df_limpio)
    
    # 8. Identificar registros sin clasificar
    sin_clasificar, parcial = identificar_registros_sin_clasificar(df_limpio)
    
    # 9. Guardar resultados
    archivo_limpio = guardar_resultados(df_limpio, mapeos, stats)
    
    print("\n" + "="*60)
    print("LIMPIEZA COMPLETADA EXITOSAMENTE!")
    print("="*60)
    
    return df_limpio, cambios, sin_clasificar, parcial, stats_años

# Ejemplo de uso:
# Si tu archivo se llama 'datos.csv', solo necesitas ejecutar esta línea
# df_limpio, cambios, sin_clasificar, parcial, stats_años = ejecutar_limpieza_completa("datos.csv")