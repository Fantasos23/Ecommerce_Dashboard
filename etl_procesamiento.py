import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Directorios de Entrada y Salida
DATA_DIR = Path('Data') if Path('Data').exists() else Path('data')
OUTPUT_DIR = Path('datasets_procesados')

# Crear la carpeta de salida si no existe
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def limpiar_y_estandarizar_df(df):
    if df.empty:
        return df

    # 1. Eliminar filas completamente vacías
    df = df.dropna(how='all').copy()

    # 2. Limpiar espacios en blanco en los nombres de las columnas
    df.columns = [str(col).strip() for col in df.columns]

    # 3. Limpiar celdas de texto
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()

    # 4. Convertir columnas booleanas a enteros
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    # 5. Reemplazar 'nan' textual por vacíos
    df = df.replace(to_replace=['nan', 'NaN', 'null', 'NULL', 'None', 'none'], value='')

    return df


def unificar_carpeta(nombre_fuente, separador_default=';', id_duplicados=None, usar_api=True):
    """
    Busca todos los archivos en la carpeta correspondiente, los une y aplica limpieza estructural.
    Si nombre_fuente es 'VTEX' o 'Meta' y usar_api es True, intenta sincronizar directamente desde la API.
    """
    if nombre_fuente.upper() == 'VTEX' and usar_api:
        try:
            from vtex_api_sync import sincronizar_vtex_orders
            print("\n" + "="*50)
            print("SINCRONIZANDO VTEX DIRECTAMENTE DESDE LA API")
            print("="*50)
            res = sincronizar_vtex_orders()
            if res.get('status') == 'success':
                return
        except Exception as e:
            print(f"⚠️ No se pudo sincronizar VTEX vía API ({e}). Procediendo con archivos locales de Data/VTEX...")

    if nombre_fuente.upper() == 'META' and usar_api:
        try:
            from meta_api_sync import sincronizar_meta_ads
            print("\n" + "="*50)
            print("SINCRONIZANDO META ADS DIRECTAMENTE DESDE LA API")
            print("="*50)
            res = sincronizar_meta_ads()
            if res.get('status') == 'success':
                return
        except Exception as e:
            print(f"⚠️ No se pudo sincronizar Meta vía API ({e}). Procediendo con archivos locales de Data/Meta...")

    if nombre_fuente.upper() == 'GOOGLE' and usar_api:
        try:
            from google_ads_api_sync import sincronizar_google_ads
            print("\n" + "="*50)
            print("SINCRONIZANDO GOOGLE ADS DIRECTAMENTE DESDE LA API")
            print("="*50)
            res = sincronizar_google_ads()
            if res.get('status') == 'success':
                return
        except Exception as e:
            print(f"⚠️ No se pudo sincronizar Google Ads vía API ({e}). Procediendo con archivos locales de Data/Google...")

    path_carpeta = DATA_DIR / nombre_fuente
    archivos = list(path_carpeta.glob('*.csv')) + list(path_carpeta.glob('*.xlsx'))

    print("\n" + "="*50)
    print(f"PROCESANDO Y UNIFICANDO ARCHIVOS DE: {nombre_fuente.upper()}")
    print("="*50)

    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {path_carpeta}")
        return

    dfs = []
    print(f"📂 Encontrados {len(archivos)} archivos:")

    for archivo in archivos:
        print(f"  └─ Leyendo: {archivo.name}")
        try:
            if archivo.suffix == '.csv':
                # Intentamos primero con la autodetección de separador
                try:
                    df = pd.read_csv(archivo, sep=None, engine='python', encoding='utf-8-sig', low_memory=False)
                except Exception:
                    df = pd.read_csv(archivo, sep=separador_default, encoding='utf-8-sig', low_memory=False)
            else:
                df = pd.read_excel(archivo)

            print(f"     ├── Leídas {len(df):,} filas y {len(df.columns)} columnas.")
            dfs.append(df)
        except Exception as e:
            print(f"  ❌ Error leyendo {archivo.name}: {e}")

    if not dfs:
        print("⚠️ No se pudieron procesar datos.")
        return

    # Unificación masiva
    df_unificado = pd.concat(dfs, ignore_index=True)
    filas_totales = len(df_unificado)

    # Limpieza estructural
    df_unificado = limpiar_y_estandarizar_df(df_unificado)

    # ==========================================================
    # CORRECCIÓN DE DESDUPLICACIÓN:
    # Ignoramos la deduplicación por 'Order' para VTEX para no perder SKUs
    # ==========================================================
    if nombre_fuente.upper() == 'VTEX':
        # Para VTEX NUNCA desduplicamos por 'Order'. Solo borramos filas 100% idénticas.
        df_unificado = df_unificado.drop_duplicates(keep='first')
    elif id_duplicados and id_duplicados in df_unificado.columns:
        df_unificado = df_unificado.drop_duplicates(subset=[id_duplicados], keep='first')
    else:
        df_unificado = df_unificado.drop_duplicates(keep='first')

    filas_limpias = len(df_unificado)
    duplicados_eliminados = filas_totales - filas_limpias

    # Exportar archivo consolidado
    ruta_salida = OUTPUT_DIR / f'dataset_{nombre_fuente.lower()}_unificado.csv'
    df_unificado.to_csv(ruta_salida, index=False, encoding='utf-8-sig')

    print(f"\n✅ Finalizado con éxito:")
    print(f"  • Filas totales leídas: {filas_totales:,}")
    print(f"  • Filas duplicadas omitidas: {duplicados_eliminados:,}")
    print(f"  • Total filas guardadas: {filas_limpias:,}")
    print(f"📁 Guardado en: {ruta_salida}")


if __name__ == '__main__':
    unificar_carpeta('VTEX', ';', 'Order')
    unificar_carpeta('Meta', ',')
    unificar_carpeta('Google', ',')