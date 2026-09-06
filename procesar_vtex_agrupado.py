from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Rutas de entrada y salida
DATASETS_DIR = Path('datasets_procesados')
ENTRADA_VTEX = DATASETS_DIR / 'dataset_vtex_unificado.csv'

# Marcamos las dos salidas:
SALIDA_VTEX_AGRUPADO = DATASETS_DIR / 'dataset_vtex_agrupado_ordenes.csv'
SALIDA_VTEX_DETALLE_SKU = DATASETS_DIR / 'dataset_vtex_detalle_skus.csv'


def generar_dataset_vtex_por_orden():
    print("\n" + "="*60)
    print("PROCESANDO VTEX: AGRUPACIÓN Y DETALLE DE ÓRDENES Y SKUS")
    print("="*60)

    if not ENTRADA_VTEX.exists():
        print(f"❌ Error: No se encontró el archivo {ENTRADA_VTEX}.")
        print("   Asegúrate de haber ejecutado primero el ETL de unificación.")
        return

    print(f"📂 Leyendo: {ENTRADA_VTEX.name}...")
    df = pd.read_csv(ENTRADA_VTEX, low_memory=False)

    # 1. Mapeo de columnas requeridas
    col_mapping = {
        'Order': 'Order',
        'Creation Date': 'Creation Date',
        'Client Document': 'Client Document',
        'City': 'City',
        'Status': 'Status',
        'UtmMedium': 'UtmMedium',
        'UtmCampaign': 'UtmCampaign',
        'UtmSource': 'UtmSource',
        'Coupon': 'Coupon',
        'Payment System Name': 'Payment System Name',
        'Quantity_SKU': 'Quantity_SKU',
        'SKU Name': 'SKU Name',
        'Total Value': 'Total Value',
        'Discounts Names': 'Discounts Names'
    }

    df = df.rename(columns={c: col_mapping[c] for c in df.columns if c in col_mapping})

    columnas_deseadas = [
        'Order', 'Creation Date', 'Client Document', 'City', 'Status',
        'UtmMedium', 'UtmCampaign', 'UtmSource', 'Coupon',
        'Payment System Name', 'Quantity_SKU', 'SKU Name', 'Total Value', 'Discounts Names'
    ]

    cols_existentes = [col for col in columnas_deseadas if col in df.columns]
    
    if 'Order' not in cols_existentes:
        print("❌ Error: La columna clave 'Order' no existe en el archivo unificado.")
        return

    df_sub = df[cols_existentes].copy()

    # 2. Asegurar tipos de datos numéricos
    if 'Quantity_SKU' in df_sub.columns:
        df_sub['Quantity_SKU'] = pd.to_numeric(df_sub['Quantity_SKU'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)

    if 'Total Value' in df_sub.columns:
        df_sub['Total Value'] = pd.to_numeric(df_sub['Total Value'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)

    cols_texto = [c for c in cols_existentes if c not in ['Order', 'Quantity_SKU', 'Total Value']]
    for c in cols_texto:
        df_sub[c] = df_sub[c].fillna('').astype(str).str.strip()

    # Eliminar duplicados exactos
    filas_iniciales = len(df_sub)
    df_sub = df_sub.drop_duplicates().copy()
    duplicados_removidos = filas_iniciales - len(df_sub)

    if duplicados_removidos > 0:
        print(f"🧹 Se eliminaron {duplicados_removidos:,} filas exactamente duplicadas.")
    else:
        print("✨ No se encontraron filas duplicadas en el dataset.")

    # Filtrar órdenes canceladas
    if 'Status' in df_sub.columns:
        patron_cancelado = 'cancel'
        ordenes_canceladas = df_sub[
            df_sub['Status'].str.lower().str.contains(patron_cancelado, na=False)
        ]['Order'].unique()

        if len(ordenes_canceladas) > 0:
            print(f"🚫 Eliminando {len(ordenes_canceladas):,} órdenes canceladas...")
            df_sub = df_sub[~df_sub['Order'].isin(ordenes_canceladas)].copy()

    print(f"📊 Total de ítems/filas a procesar: {len(df_sub):,}")

    # ==========================================================
    # 3. GENERAR DATASET 1: DETALLE POR SKU (GRANULAR)
    # ==========================================================
    if 'SKU Name' in df_sub.columns:
        print("📦 Generando dataset de detalle por SKU...")
        
        df_sku_detalle = df_sub.groupby(['Order', 'SKU Name'], as_index=False).agg({
            'Quantity_SKU': 'sum',
            'Total Value': 'first',  # Tomamos el valor de la orden (sin sumar duplicados de filas)
            'Creation Date': 'first',
            'City': 'first',
            'Status': 'first'
        })

        # Prorrateamos el Total Value de la orden proporcionalmente a las unidades de este SKU frente al total de la orden
        total_unidades_orden = df_sku_detalle.groupby('Order')['Quantity_SKU'].transform('sum')
        df_sku_detalle['Total_Value_SKU'] = np.where(
            total_unidades_orden > 0,
            (df_sku_detalle['Total Value'] / total_unidades_orden) * df_sku_detalle['Quantity_SKU'],
            0
        )

        SALIDA_VTEX_DETALLE_SKU.parent.mkdir(parents=True, exist_ok=True)
        df_sku_detalle.to_csv(SALIDA_VTEX_DETALLE_SKU, index=False, encoding='utf-8-sig')
        print(f"  └─ Guardado detalle por producto en: {SALIDA_VTEX_DETALLE_SKU.name}")

    # ==========================================================
    # 4. GENERAR DATASET 2: CONSOLIDADO A NIVEL DE ORDEN
    # ==========================================================
    print("🔄 Agrupando por 'Order' a nivel de cabecera...")

    # Creamos un texto descriptivo temporal, ej: "Camiseta Roja (x2)"
    if 'SKU Name' in df_sub.columns and 'Quantity_SKU' in df_sub.columns:
        df_sub['SKU_con_Cantidad'] = df_sub['SKU Name'] + " (x" + df_sub['Quantity_SKU'].astype(int).astype(str) + ")"

    agg_rules = {}
    for col in cols_existentes:
        if col == 'Order':
            continue
        elif col == 'Quantity_SKU':
            # Sumamos las unidades totales de la orden
            agg_rules[col] = 'sum'
        elif col == 'Total Value':
            # ⚠️ CAMBIO CLAVE: Usamos 'first' para no multiplicar el valor total de la orden por el N° de filas
            agg_rules[col] = 'first'
        elif col == 'SKU Name':
            # Consolidamos los SKUs junto a sus cantidades
            agg_rules['SKU_con_Cantidad'] = lambda x: ' | '.join(unique_vals) if (unique_vals := [v for v in set(x) if v and not v.startswith(' (x')]) else ''
        elif col in ['Discounts Names', 'Payment System Name']:
            agg_rules[col] = lambda x: ' | '.join(unique_vals) if (unique_vals := [v for v in set(x) if v and v.lower() != 'nan']) else ''
        else:
            agg_rules[col] = 'first'

    df_agrupado = df_sub.groupby('Order', as_index=False).agg(agg_rules)

    # Renombrar columna para claridad en el CSV de órdenes
    if 'SKU_con_Cantidad' in df_agrupado.columns:
        df_agrupado = df_agrupado.rename(columns={'SKU_con_Cantidad': 'SKU Name'})

    # 5. Cliente Nuevo vs Recurrente
    if 'Client Document' in df_agrupado.columns and 'Creation Date' in df_agrupado.columns:
        df_agrupado['Creation Date_DT'] = pd.to_datetime(
            df_agrupado['Creation Date'], 
            format='mixed', 
            dayfirst=True, 
            errors='coerce'
        )
        
        mask_doc_valido = df_agrupado['Client Document'].astype(str).str.strip() != ''
        df_agrupado.loc[mask_doc_valido, 'First_Purchase_Date'] = df_agrupado[mask_doc_valido].groupby('Client Document')['Creation Date_DT'].transform('min')
        
        df_agrupado['Tipo_Cliente'] = np.where(
            df_agrupado['Creation Date_DT'] == df_agrupado['First_Purchase_Date'], 
            'Nuevo', 
            'Recurrente'
        )
        
        df_agrupado = df_agrupado.drop(columns=['Creation Date_DT', 'First_Purchase_Date'])

    # 6. Guardar dataset agrupado por orden
    SALIDA_VTEX_AGRUPADO.parent.mkdir(parents=True, exist_ok=True)
    df_agrupado.to_csv(SALIDA_VTEX_AGRUPADO, index=False, encoding='utf-8-sig')

    print(f"\n✅ Proceso completado exitosamente:")
    print(f"  • Filas procesadas (válidas): {len(df_sub):,}")
    print(f"  • Órdenes únicas finales: {len(df_agrupado):,}")
    print(f"📁 Archivos generados:")
    print(f"  1. Órdenes Consolidadas: {SALIDA_VTEX_AGRUPADO}")
    print(f"  2. Detalle SKUs:          {SALIDA_VTEX_DETALLE_SKU}")


if __name__ == '__main__':
    generar_dataset_vtex_por_orden()