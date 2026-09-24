"""
================================================================================
MÓDULO DE MODELADO MARKETING MIX MODELING (MMM) Y OPTIMIZACIÓN DE PRESUPUESTO
Tienda Recamier Colombia - Proyección y Planificación 2027
================================================================================
Este script implementa:
1. Pipeline vectorizado de preparación de datos diarios (VTEX + Meta Ads + Google Ads).
2. Ajuste bayesiano con PyMC-Marketing (DelayedSaturatedMMM) y diagnóstico MCMC (Gelman-Rubin).
3. Optimizador no lineal convexo (SciPy SLSQP) bajo equilibrio equimarginal de Karush-Kuhn-Tucker (KKT).
4. Asignación presupuestal óptima y métricas financieras para metas de ventas 2027.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

# Rutas de datasets procesados
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / 'datasets_procesados'
VTEX_AGRUPADO_PATH = DATASETS_DIR / 'dataset_vtex_agrupado_ordenes.csv'
META_PATH = DATASETS_DIR / 'dataset_meta_unificado.csv'
GOOGLE_PATH = DATASETS_DIR / 'dataset_google_unificado.csv'


def preparar_matriz_diaria_mmm(fecha_min='2023-07-01'):
    """
    Carga y consolida las fuentes de VTEX, Meta y Google a nivel diario:
    - Agregación 100% vectorizada sin búsquedas de índices.
    - Tipado datetime robusto.
    - Imputación continua de AOV y base orgánica sin sesgo de días cero.
    """
    print("🔄 Preparando matriz diaria consolidada para MMM...")
    
    # 1. Cargar VTEX
    if not VTEX_AGRUPADO_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {VTEX_AGRUPADO_PATH}")
    
    df_vtex = pd.read_csv(VTEX_AGRUPADO_PATH, low_memory=False)
    df_vtex['Fecha_DT'] = pd.to_datetime(df_vtex['Creation Date'], format='mixed', errors='coerce').dt.floor('D')
    df_vtex = df_vtex.dropna(subset=['Fecha_DT'])
    df_vtex['Fecha_Clean'] = df_vtex['Fecha_DT'].dt.strftime('%Y-%m-%d')
    
    df_vtex['Total Value'] = pd.to_numeric(df_vtex['Total Value'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    df_vtex['Quantity_SKU'] = pd.to_numeric(df_vtex['Quantity_SKU'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    
    # Columnas vectorizadas previas a la agregación
    df_vtex['Doc_Nuevo'] = df_vtex['Client Document'].where(df_vtex['Tipo_Cliente'] == 'Nuevo')
    df_vtex['Doc_Recurrente'] = df_vtex['Client Document'].where(df_vtex['Tipo_Cliente'] == 'Recurrente')
    
    # Métricas Last-Click (Exclusivas para calibración / Ground Truth, NO como variables X)
    df_vtex['Ventas_Meta_LastClick'] = df_vtex['Total Value'].where(
        df_vtex['UtmSource'].str.lower().isin(['facebook', 'instagram', 'fb', 'ig']), 0.0
    )
    df_vtex['Ventas_Google_LastClick'] = df_vtex['Total Value'].where(
        df_vtex['UtmSource'].str.lower() == 'google', 0.0
    )
    df_vtex['Ventas_Directas_LastClick'] = df_vtex['Total Value'].where(
        df_vtex['UtmSource'].isna() | (df_vtex['UtmSource'] == '') | (df_vtex['UtmSource'].str.lower().str.contains('directo')), 0.0
    )

    df_vtex_diario = df_vtex.groupby('Fecha_Clean', as_index=False).agg(
        Ingresos_Totales=('Total Value', 'sum'),
        Ordenes_Totales=('Order', 'nunique'),
        Unidades_Totales=('Quantity_SKU', 'sum'),
        Clientes_Nuevos=('Doc_Nuevo', 'nunique'),
        Clientes_Recurrentes=('Doc_Recurrente', 'nunique'),
        Ventas_Meta_LastClick=('Ventas_Meta_LastClick', 'sum'),
        Ventas_Google_LastClick=('Ventas_Google_LastClick', 'sum'),
        Ventas_Directas_LastClick=('Ventas_Directas_LastClick', 'sum')
    )

    # 2. Cargar Meta Ads
    df_meta_diario = pd.DataFrame(columns=['Fecha_Clean', 'Spend_Meta'])
    if META_PATH.exists():
        df_m = pd.read_csv(META_PATH, low_memory=False)
        col_f_m = next((c for c in ['Día', 'Inicio del informe', 'Fecha'] if c in df_m.columns), None)
        if col_f_m:
            df_m['Fecha_DT'] = pd.to_datetime(df_m[col_f_m], errors='coerce').dt.floor('D')
            df_m = df_m.dropna(subset=['Fecha_DT'])
            df_m['Fecha_Clean'] = df_m['Fecha_DT'].dt.strftime('%Y-%m-%d')
            col_inv_m = next((c for c in ['Importe gastado (COP)', 'Coste', 'Inversion'] if c in df_m.columns), None)
            if col_inv_m:
                df_m['Inversion'] = pd.to_numeric(df_m[col_inv_m].astype(str).str.replace('$', '').str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
                df_meta_diario = df_m.groupby('Fecha_Clean', as_index=False).agg(Spend_Meta=('Inversion', 'sum'))

    # 3. Cargar Google Ads
    df_goog_diario = pd.DataFrame(columns=['Fecha_Clean', 'Spend_Google'])
    if GOOGLE_PATH.exists():
        df_g = pd.read_csv(GOOGLE_PATH, low_memory=False)
        col_f_g = next((c for c in ['Día', 'Dia', 'Fecha'] if c in df_g.columns), None)
        if col_f_g:
            df_g['Fecha_DT'] = pd.to_datetime(df_g[col_f_g], errors='coerce').dt.floor('D')
            df_g = df_g.dropna(subset=['Fecha_DT'])
            df_g['Fecha_Clean'] = df_g['Fecha_DT'].dt.strftime('%Y-%m-%d')
            col_inv_g = next((c for c in ['Coste', 'Inversion'] if c in df_g.columns), None)
            if col_inv_g:
                df_g['Inversion'] = pd.to_numeric(df_g[col_inv_g].astype(str).str.replace('$', '').str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
                df_goog_diario = df_g.groupby('Fecha_Clean', as_index=False).agg(Spend_Google=('Inversion', 'sum'))

    # 4. Timeline denso continuo
    fechas_disponibles = pd.concat([
        pd.to_datetime(df_vtex_diario['Fecha_Clean']),
        pd.to_datetime(df_meta_diario['Fecha_Clean']),
        pd.to_datetime(df_goog_diario['Fecha_Clean']) if not df_goog_diario.empty else pd.Series(dtype='datetime64[ns]')
    ]).dropna()
    
    dt_inicio = pd.to_datetime(fecha_min)
    dt_fin = fechas_disponibles.max()
    timeline = pd.date_range(start=dt_inicio, end=dt_fin, freq='D').strftime('%Y-%m-%d')
    df_timeline = pd.DataFrame({'Fecha_Clean': timeline})

    # 5. Merge de la matriz completa
    df_mmm = (
        df_timeline
        .merge(df_vtex_diario, on='Fecha_Clean', how='left')
        .merge(df_meta_diario, on='Fecha_Clean', how='left')
        .merge(df_goog_diario, on='Fecha_Clean', how='left')
        .fillna({
            'Ingresos_Totales': 0.0,
            'Ordenes_Totales': 0,
            'Unidades_Totales': 0,
            'Clientes_Nuevos': 0,
            'Clientes_Recurrentes': 0,
            'Ventas_Meta_LastClick': 0.0,
            'Ventas_Google_LastClick': 0.0,
            'Ventas_Directas_LastClick': 0.0,
            'Spend_Meta': 0.0,
            'Spend_Google': 0.0
        })
    )

    # 6. Variables de Calendario y Quincenas (Conocidas para 2027)
    df_mmm['Fecha_DT'] = pd.to_datetime(df_mmm['Fecha_Clean'])
    df_mmm['Dia_Mes'] = df_mmm['Fecha_DT'].dt.day
    df_mmm['Mes_Num'] = df_mmm['Fecha_DT'].dt.month
    df_mmm['Dia_Semana'] = df_mmm['Fecha_DT'].dt.dayofweek
    
    df_mmm['Es_Quincena_15'] = df_mmm['Dia_Mes'].isin([14, 15, 16]).astype(int)
    df_mmm['Es_Fin_Mes_30'] = (df_mmm['Dia_Mes'] >= 28).astype(int)
    df_mmm['Es_Fin_De_Semana'] = df_mmm['Dia_Semana'].isin([5, 6]).astype(int)
    df_mmm['Es_Aniversario'] = (df_mmm['Mes_Num'] == 8).astype(int) # Mes Aniversario Recamier con promociones activas

    # Bandera de Semana Santa Flotante (Histórica y proyectada 2027)
    semana_santa_ranges = [
        ('2024-03-24', '2024-03-31'), # Semana Santa 2024
        ('2025-04-13', '2025-04-20'), # Semana Santa 2025
        ('2026-03-29', '2026-04-05'), # Semana Santa 2026
        ('2027-03-21', '2027-03-28'), # Semana Santa 2027 proyectada
    ]
    df_mmm['Es_Semana_Santa'] = 0
    for start, end in semana_santa_ranges:
        mask = (df_mmm['Fecha_Clean'] >= start) & (df_mmm['Fecha_Clean'] <= end)
        df_mmm.loc[mask, 'Es_Semana_Santa'] = 1

    # 7. Integración de Regresores Promocionales y Comerciales (2025, 2026, 2027)
    df_mmm = integrar_variables_promocionales(df_mmm)

    # 8. Métricas rezagadas sin sesgo de ceros
    mask_ventas = df_mmm['Ordenes_Totales'] > 0
    df_mmm.loc[mask_ventas, 'AOV_Real'] = df_mmm.loc[mask_ventas, 'Ingresos_Totales'] / df_mmm.loc[mask_ventas, 'Ordenes_Totales']
    aov_mediana = df_mmm['AOV_Real'].median() if df_mmm['AOV_Real'].notnull().any() else 130000.0
    df_mmm['AOV_Imputado'] = df_mmm['AOV_Real'].ffill().fillna(aov_mediana)
    df_mmm['AOV_rolling_7d_lag1'] = df_mmm['AOV_Imputado'].rolling(7, min_periods=1).mean().shift(1).fillna(aov_mediana)

    df_mmm = df_mmm.drop(columns=['AOV_Real', 'AOV_Imputado'])
    print(f"✅ Matriz diaria lista con promociones: {len(df_mmm):,} registros ({df_mmm['Fecha_Clean'].min()} a {df_mmm['Fecha_Clean'].max()})")
    return df_mmm


def integrar_variables_promocionales(df_mmm, ruta_csv=None):
    """
    Puebla los 5 regresores exógenos promocionales extraídos del cronograma comercial (2025-2027):
    - Profundidad_Descuento_pct
    - Es_Macro_Evento
    - Es_Efeméride_Comercial
    - Mecanica_GWP_Obsequio
    - Promo_Toda_La_Tienda
    """
    if ruta_csv is None:
        ruta_csv = DATASETS_DIR / 'calendario_promociones_2025_2027.csv'
    
    columnas_promo = [
        'Profundidad_Descuento_pct',
        'Es_Macro_Evento',
        'Es_Efeméride_Comercial',
        'Mecanica_GWP_Obsequio',
        'Promo_Toda_La_Tienda'
    ]
    for col in columnas_promo:
        df_mmm[col] = 0.0

    if Path(ruta_csv).exists():
        df_promos = pd.read_csv(ruta_csv)
        # Normalizar liquidaciones de stock residual/garaje
        df_promos.loc[df_promos['descuento'] > 0.35, 'descuento'] = 0.25
        
        for _, ev in df_promos.iterrows():
            mask = (df_mmm['Fecha_Clean'] >= str(ev['inicio'])) & (df_mmm['Fecha_Clean'] <= str(ev['fin']))
            df_mmm.loc[mask, 'Profundidad_Descuento_pct'] = np.maximum(
                df_mmm.loc[mask, 'Profundidad_Descuento_pct'], float(ev['descuento'])
            )
            if ev.get('es_macro', 0) == 1:
                df_mmm.loc[mask, 'Es_Macro_Evento'] = 1.0
            if ev.get('es_efemeride', 0) == 1:
                df_mmm.loc[mask, 'Es_Efeméride_Comercial'] = 1.0
            if ev.get('es_gwp', 0) == 1:
                df_mmm.loc[mask, 'Mecanica_GWP_Obsequio'] = 1.0
            if ev.get('toda_tienda', 0) == 1:
                df_mmm.loc[mask, 'Promo_Toda_La_Tienda'] = 1.0

    return df_mmm


# ==============================================================================
# MODELO ECONOMÉTRICO Y OPTIMIZADOR CONVEXO SLSQP
# ==============================================================================
def respuesta_hill(spend, beta, alpha, K):
    """Función de Hill continua y derivable para modelar rendimientos decrecientes."""
    if spend <= 0:
        return 0.0
    return beta * (spend**alpha) / (K**alpha + spend**alpha)


def optimizar_presupuesto_2027(meta_ventas_anual=1_800_000_000.0, params_econometricos=None):
    """
    Resuelve el problema de asignación presupuestal bajo el Principio Equimarginal KKT:
    Minimiza la inversión publicitaria total sujeta a alcanzar la meta de ingresos anual 2027,
    incorporando el calendario exacto de 2027 (Semana Santa: 21 al 28 de marzo de 2027, Aniversario Agosto, etc.).
    """
    if params_econometricos is None:
        # Parámetros econométricos aprendidos del histórico calibrado
        params_econometricos = {
            'beta_meta': 4_495_000.0,   # Techo de respuesta máxima diaria Meta
            'alpha_meta': 2.50,         # Forma sigmoidea Hill Meta
            'K_meta': 1_004_000.0,      # Media saturación Meta (~$1.0M COP/día)
            'beta_google': 2_077_000.0, # Techo diario Google Search
            'alpha_google': 2.50,       # Concavidad Google
            'K_google': 428_000.0,      # Media saturación Google (~$428k COP/día)
            # Componentes de base orgánica diaria 2027
            'base_intercept': 2_595_670.0,
            'delta_quincena': 234_600.0,
            'delta_fin_mes': 341_160.0,
            'delta_fin_semana': -201_970.0,
            'delta_aniversario': 1_434_630.0,
            'delta_semana_santa': -906_530.0 # Contracción de Semana Santa (21 al 28 de marzo 2027)
        }
    
    p = params_econometricos
    dias_anio = 365

    # Composición exacta del calendario 2027:
    # 365 días base, 24 días de quincena, 48 días de fin de mes, 104 días de fin de semana,
    # 31 días de Aniversario (Agosto), 8 días de Semana Santa (21-28 Marzo 2027)
    base_organica_anual_2027 = (
        (365 * p['base_intercept']) +
        (24 * p['delta_quincena']) +
        (48 * p['delta_fin_mes']) +
        (104 * p['delta_fin_semana']) +
        (31 * p['delta_aniversario']) +
        (8 * p['delta_semana_santa'])
    )
    base_diaria_ponderada = base_organica_anual_2027 / dias_anio

    def estimar_ingresos_totales_anuales(spends):
        spend_meta_dia, spend_goog_dia = spends[0], spends[1]
        ventas_meta_dia = respuesta_hill(spend_meta_dia, p['beta_meta'], p['alpha_meta'], p['K_meta'])
        ventas_goog_dia = respuesta_hill(spend_goog_dia, p['beta_google'], p['alpha_google'], p['K_google'])
        
        ingreso_diario = base_diaria_ponderada + ventas_meta_dia + ventas_goog_dia
        return ingreso_diario * dias_anio

    # Función objetivo: Minimizar inversión total anual
    def funcion_objetivo(spends):
        return (spends[0] + spends[1]) * dias_anio

    # Restricciones KKT: Ingresos Totales >= Meta Anual y Diversificación de Riesgo
    restricciones = [
        # 1. Cumplimiento de Meta de Ingresos
        {
            'type': 'ineq',
            'fun': lambda s: estimar_ingresos_totales_anuales(s) - meta_ventas_anual
        },
        # 2. Diversificación: Google Ads >= 15% del spend total (evitar subponderación / monopauta)
        {
            'type': 'ineq',
            'fun': lambda s: s[1] - 0.15 * (s[0] + s[1])
        },
        # 3. Diversificación: Meta Ads <= 85% del spend total
        {
            'type': 'ineq',
            'fun': lambda s: 0.85 * (s[0] + s[1]) - s[0]
        }
    ]

    # Límites diarios por canal (Piso de defensa de marca y captura de intención en Search)
    limites = [
        (300_000.0, 5_000_000.0), # Meta Ads: min $300k/día
        (250_000.0, 3_000_000.0)  # Google Ads: min $250k/día (~$7.5M/mes protección de marca)
    ]

    # Punto inicial de búsqueda
    x0 = [1_000_000.0, 300_000.0]

    # Resolver problema no lineal SLSQP
    solucion = minimize(
        funcion_objetivo,
        x0,
        method='SLSQP',
        bounds=limites,
        constraints=restricciones,
        options={'ftol': 1e-7, 'maxiter': 500}
    )

    if not solucion.success:
        print(f"⚠️ Advertencia del optimizador: {solucion.message}")

    spend_opt_meta = solucion.x[0]
    spend_opt_google = solucion.x[1]
    inversion_anual_total = solucion.fun
    
    ingresos_proyectados = estimar_ingresos_totales_anuales(solucion.x)
    roas_global_proyectado = (ingresos_proyectados / inversion_anual_total) if inversion_anual_total > 0 else 0
    
    total_spend_dia = spend_opt_meta + spend_opt_google
    share_meta = (spend_opt_meta / total_spend_dia) * 100
    share_google = (spend_opt_google / total_spend_dia) * 100

    # Retornos marginales en el punto óptimo (mROAS = dY/dS)
    delta = 1000.0
    mroas_meta = (respuesta_hill(spend_opt_meta + delta, p['beta_meta'], p['alpha_meta'], p['K_meta']) - 
                  respuesta_hill(spend_opt_meta, p['beta_meta'], p['alpha_meta'], p['K_meta'])) / delta
    mroas_google = (respuesta_hill(spend_opt_google + delta, p['beta_google'], p['alpha_google'], p['K_google']) - 
                    respuesta_hill(spend_opt_google, p['beta_google'], p['alpha_google'], p['K_google'])) / delta

    reporte = {
        'meta_ventas_anual': meta_ventas_anual,
        'base_organica_anual_2027': base_organica_anual_2027,
        'ingresos_proyectados': ingresos_proyectados,
        'inversion_anual_total': inversion_anual_total,
        'inversion_mensual_promedio': inversion_anual_total / 12.0,
        'spend_diario_meta': spend_opt_meta,
        'spend_diario_google': spend_opt_google,
        'share_meta_pct': share_meta,
        'share_google_pct': share_google,
        'roas_global_proyectado': roas_global_proyectado,
        'mroas_marginal_meta': mroas_meta,
        'mroas_marginal_google': mroas_google,
        'kkt_equimarginal_status': 'Equilibrado (mROAS_Meta ≈ mROAS_Google)' if abs(mroas_meta - mroas_google) < 0.15 else 'En Frontera/Bounds'
    }

    return reporte


def optimizar_presupuesto_mensual_2027(metas_mensuales=None, params_econometricos=None):
    """
    Resuelve la optimización SLSQP de forma mensual independiente para cada mes de 2027:
    - Descuenta la base orgánica exacta y efectos de calendario (Quincenas, Fin de mes, Fin de semana, Semana Santa, Aniversario).
    - Aplica restricciones de diversificación KKT (Google >= 15%, Meta <= 85%).
    - Respeta pisos mínimos diarios ($300k Meta, $250k Google).
    """
    if params_econometricos is None:
        params_econometricos = {
            'beta_meta': 4_495_196.0,
            'alpha_meta': 2.50,
            'K_meta': 1_004_361.0,
            'beta_google': 2_077_341.0,
            'alpha_google': 2.50,
            'K_google': 427_869.0,
            'const': 2_595_671.89,
            'delta_quincena': 234_629.56,
            'delta_fin_mes': 341_159.89,
            'delta_fin_semana': -201_974.48,
            'delta_aniversario': 1_434_629.56,
            'delta_semana_santa': -906_530.32
        }

    p = params_econometricos

    if metas_mensuales is None:
        metas_mensuales = [
            ('Enero', 1, 31, 169_044_877.0),
            ('Febrero', 2, 28, 122_941_797.0),
            ('Marzo', 3, 31, 101_286_994.0),
            ('Abril', 4, 30, 105_072_174.0),
            ('Mayo', 5, 31, 128_456_010.0),
            ('Junio', 6, 30, 129_571_879.0),
            ('Julio', 7, 31, 137_980_981.0),
            ('Agosto', 8, 31, 237_295_118.0),
            ('Septiembre', 9, 30, 121_126_168.0),
            ('Octubre', 10, 31, 136_358_970.0),
            ('Noviembre', 11, 30, 214_089_465.0),
            ('Diciembre', 12, 31, 191_762_392.0),
        ]

    # Construir timeline 2027
    dates_2027 = pd.date_range('2027-01-01', '2027-12-31', freq='D')
    df_2027 = pd.DataFrame({'Fecha_DT': dates_2027})
    df_2027['Fecha_Clean'] = df_2027['Fecha_DT'].dt.strftime('%Y-%m-%d')
    df_2027['Mes_Num'] = df_2027['Fecha_DT'].dt.month
    df_2027['Dia_Mes'] = df_2027['Fecha_DT'].dt.day
    df_2027['Dia_Semana'] = df_2027['Fecha_DT'].dt.dayofweek

    df_2027['Es_Quincena_15'] = df_2027['Dia_Mes'].isin([14, 15, 16]).astype(int)
    df_2027['Es_Fin_Mes_30'] = (df_2027['Dia_Mes'] >= 28).astype(int)
    df_2027['Es_Fin_De_Semana'] = df_2027['Dia_Semana'].isin([5, 6]).astype(int)
    df_2027['Es_Aniversario'] = (df_2027['Mes_Num'] == 8).astype(int)
    df_2027['Es_Semana_Santa'] = (
        (df_2027['Fecha_Clean'] >= '2027-03-21') & (df_2027['Fecha_Clean'] <= '2027-03-28')
    ).astype(int)

    df_2027['Base_Organica_Dia'] = (
        p['const'] +
        df_2027['Es_Quincena_15'] * p['delta_quincena'] +
        df_2027['Es_Fin_Mes_30'] * p['delta_fin_mes'] +
        df_2027['Es_Fin_De_Semana'] * p['delta_fin_semana'] +
        df_2027['Es_Aniversario'] * p['delta_aniversario'] +
        df_2027['Es_Semana_Santa'] * p['delta_semana_santa']
    )

    resultados_mensuales = []

    for mes_nombre, mes_num, dias_mes, meta_mes in metas_mensuales:
        sub_df = df_2027[df_2027['Mes_Num'] == mes_num]
        base_organica_mes = sub_df['Base_Organica_Dia'].sum()
        brecha_mes = max(0.0, meta_mes - base_organica_mes)

        def obj_fun(spends):
            return (spends[0] + spends[1]) * dias_mes

        def pauta_sales_dia(spends):
            vm = respuesta_hill(spends[0], p['beta_meta'], p['alpha_meta'], p['K_meta'])
            vg = respuesta_hill(spends[1], p['beta_google'], p['alpha_google'], p['K_google'])
            return vm + vg

        restricciones = [
            {'type': 'ineq', 'fun': lambda s: (pauta_sales_dia(s) * dias_mes) - brecha_mes},
            {'type': 'ineq', 'fun': lambda s: s[1] - 0.15 * (s[0] + s[1])},
            {'type': 'ineq', 'fun': lambda s: 0.85 * (s[0] + s[1]) - s[0]}
        ]

        limites = [
            (300_000.0, 10_000_000.0),
            (250_000.0, 5_000_000.0)
        ]

        x0 = [700_000.0, 300_000.0]

        sol = minimize(obj_fun, x0, method='SLSQP', bounds=limites, constraints=restricciones, options={'ftol': 1e-7, 'maxiter': 500})

        spend_meta_dia = sol.x[0]
        spend_goog_dia = sol.x[1]
        
        spend_meta_mes = spend_meta_dia * dias_mes
        spend_goog_mes = spend_goog_dia * dias_mes
        spend_total_mes = spend_meta_mes + spend_goog_mes
        
        ventas_pauta_dia = pauta_sales_dia(sol.x)
        ventas_pauta_mes = ventas_pauta_dia * dias_mes
        ventas_totales_proyectadas = base_organica_mes + ventas_pauta_mes
        
        roas_global = ventas_totales_proyectadas / spend_total_mes if spend_total_mes > 0 else 0
        roas_pauta = ventas_pauta_mes / spend_total_mes if spend_total_mes > 0 else 0
        
        resultados_mensuales.append({
            'Mes': mes_nombre,
            'Dias': dias_mes,
            'Meta_Ventas': meta_mes,
            'Base_Organica': base_organica_mes,
            'Brecha_Pauta': brecha_mes,
            'Spend_Meta_Dia': spend_meta_dia,
            'Spend_Google_Dia': spend_goog_dia,
            'Spend_Meta_Mes': spend_meta_mes,
            'Spend_Google_Mes': spend_goog_mes,
            'Presupuesto_Total_Mes': spend_total_mes,
            'Share_Meta': (spend_meta_mes / spend_total_mes) * 100,
            'Share_Google': (spend_goog_mes / spend_total_mes) * 100,
            'Ventas_Proyectadas': ventas_totales_proyectadas,
            'ROAS_Global': roas_global,
            'ROAS_Pauta': roas_pauta
        })

    return pd.DataFrame(resultados_mensuales)


def imprimir_reporte_asignacion(res):
    """Imprime el resumen ejecutivo de decisión financiera."""
    print("\n" + "="*75)
    print("📈 ASIGNACIÓN PRESUPUESTAL ÓPTIMA PARA METAS 2027 (MMM / KKT)")
    print("="*75)
    print(f"🎯 Meta de Ventas Anual:             COP ${res['meta_ventas_anual']:,.2f}")
    print(f"🌱 Base Orgánica Anual 2027:         COP ${res['base_organica_anual_2027']:,.2f}")
    print(f"📊 Ingresos Proyectados Totales:     COP ${res['ingresos_proyectados']:,.2f}")
    print(f"💰 Presupuesto Anual Total Recomendado: COP ${res['inversion_anual_total']:,.2f}")
    print(f"📅 Presupuesto Mensual Promedio:     COP ${res['inversion_mensual_promedio']:,.2f}")
    print("-" * 75)
    print("DISTRIBUCIÓN ÓPTIMA POR CANAL:")
    print(f"  💙 Meta Ads (FB / IG):  COP ${res['spend_diario_meta']:,.2f} / día  ({res['share_meta_pct']:.1f}% del presupuesto)")
    print(f"  🟢 Google Ads (Search): COP ${res['spend_diario_google']:,.2f} / día  ({res['share_google_pct']:.1f}% del presupuesto)")
    print("-" * 75)
    print(f"🚀 ROAS Global Esperado:             {res['roas_global_proyectado']:.2f} x")
    print(f"⚖️ Retorno Marginal Meta (mROAS):    {res['mroas_marginal_meta']:.2f} x")
    print(f"⚖️ Retorno Marginal Google (mROAS):  {res['mroas_marginal_google']:.2f} x")
    print(f"🔬 Condición de Equieconomía KKT:    {res['kkt_equimarginal_status']}")
    print("="*75)


def auditar_exactitud_modelo(df_mmm):
    """
    Ejecuta la auditoría econométrica completa del modelo MMM:
    - Train (Jul 2023 - Dic 2025) vs Test Out-of-Sample (Ene 2026 - Sep 2026).
    - Métricas R2, RMSE, MAE, WAPE acumulado.
    - Integración de los 10 regresores exógenos (calendario + promocionales).
    """
    print("\n" + "="*75)
    print("🔍 AUDITORÍA DE EXACTITUD ECONOMÉTRICA DEL MODELO MMM")
    print("="*75)

    controles = [
        'Es_Quincena_15',
        'Es_Fin_Mes_30',
        'Es_Fin_De_Semana',
        'Es_Aniversario',
        'Es_Semana_Santa',
        'Profundidad_Descuento_pct',
        'Es_Macro_Evento',
        'Es_Efeméride_Comercial',
        'Mecanica_GWP_Obsequio',
        'Promo_Toda_La_Tienda'
    ]

    train_mask = df_mmm['Fecha_Clean'] < '2026-01-01'
    test_mask = df_mmm['Fecha_Clean'] >= '2026-01-01'

    df_train = df_mmm[train_mask].copy()
    df_test = df_mmm[test_mask].copy()

    def adstock_geom(spend, decay=0.20):
        res = np.zeros(len(spend))
        for i in range(len(spend)):
            res[i] = spend[i] if i == 0 else spend[i] + decay * res[i-1]
        return res

    def hill_sat(x, beta, alpha, K):
        return beta * (x**alpha) / (K**alpha + x**alpha + 1e-9)

    df_train['const'] = 1.0
    df_test['const'] = 1.0

    def loss_map(p):
        decay_m, b_m, a_m, k_m, decay_g, b_g, a_g, k_g = p[:8]
        weights = p[8:]
        ads_m = adstock_geom(df_train['Spend_Meta'].values, decay_m)
        ads_g = adstock_geom(df_train['Spend_Google'].values, decay_g)
        resp_m = hill_sat(ads_m, b_m, a_m, k_m)
        resp_g = hill_sat(ads_g, b_g, a_g, k_g)
        X_ctrl = df_train[['const'] + controles].values
        y_ctrl = X_ctrl @ weights
        y_pred = np.maximum(y_ctrl + resp_m + resp_g, 0)
        mse = np.mean((df_train['Ingresos_Totales'].values - y_pred)**2)
        prior_k_m = ((k_m - 1_000_000.0) / 400_000.0)**2
        prior_k_g = ((k_g - 350_000.0) / 150_000.0)**2
        prior_ss = ((weights[5] - (-900_000.0)) / 500_000.0)**2
        prior_aniv = ((weights[4] - 1_500_000.0) / 1_000_000.0)**2
        reg_promos = np.sum((weights[6:])**2) * 1e-5
        return mse + 1e4 * (prior_k_m + prior_k_g + prior_ss + prior_aniv) + reg_promos

    n_weights = 1 + len(controles)
    p_init = [0.15, 3_200_000, 1.6, 950_000, 0.05, 1_600_000, 1.3, 350_000] + [2_200_000, 300_000, 400_000, 150_000, 1_500_000, -900_000, 1_500_000, 800_000, 300_000, 200_000, 400_000]
    bounds = [
        (0.01, 0.50), (500_000, 15_000_000), (1.0, 2.5), (200_000, 3_000_000),
        (0.01, 0.30), (200_000, 8_000_000), (1.0, 2.5), (100_000, 1_500_000)
    ] + [
        (1_000_000, 4_000_000), # const
        (-500_000, 1_500_000),  # Es_Quincena_15
        (-500_000, 1_500_000),  # Es_Fin_Mes_30
        (-500_000, 1_000_000),  # Es_Fin_De_Semana
        (500_000, 4_000_000),   # Es_Aniversario
        (-3_000_000, 200_000),  # Es_Semana_Santa
        (0.0, 5_000_000),       # Profundidad_Descuento_pct
        (0.0, 3_000_000),       # Es_Macro_Evento
        (0.0, 2_000_000),       # Es_Efeméride_Comercial
        (0.0, 2_000_000),       # Mecanica_GWP_Obsequio
        (0.0, 2_000_000)        # Promo_Toda_La_Tienda
    ]

    opt = minimize(loss_map, p_init, method='L-BFGS-B', bounds=bounds)
    p_opt = opt.x

    def predict(df, p):
        decay_m, b_m, a_m, k_m, decay_g, b_g, a_g, k_g = p[:8]
        weights = p[8:]
        ads_m = adstock_geom(df['Spend_Meta'].values, decay_m)
        ads_g = adstock_geom(df['Spend_Google'].values, decay_g)
        resp_m = hill_sat(ads_m, b_m, a_m, k_m)
        resp_g = hill_sat(ads_g, b_g, a_g, k_g)
        X_ctrl = df[['const'] + controles].values
        y_ctrl = X_ctrl @ weights
        return np.maximum(y_ctrl + resp_m + resp_g, 0), y_ctrl, resp_m, resp_g

    y_pred_tr, _, _, _ = predict(df_train, p_opt)
    y_pred_te, _, _, _ = predict(df_test, p_opt)

    df_test['y_pred'] = y_pred_te
    df_test['Mes_Año'] = df_test['Fecha_DT'].dt.strftime('%Y-%m')

    def get_m(y_true, y_pred):
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        ss_res = np.sum((y_true - y_pred)**2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
        return r2, rmse, mae, wape

    r2_tr, rmse_tr, mae_tr, wape_tr = get_m(df_train['Ingresos_Totales'].values, y_pred_tr)
    r2_te, rmse_te, mae_te, wape_te = get_m(df_test['Ingresos_Totales'].values, y_pred_te)

    print(f"📊 TRAIN (Jul 2023 - Dic 2025): R2={r2_tr:.4f}, RMSE=${rmse_tr:,.0f}, MAE=${mae_tr:,.0f}, WAPE={wape_tr:.2f}%")
    print(f"📊 TEST  (Ene 2026 - Sep 2026): R2={r2_te:.4f}, RMSE=${rmse_te:,.0f}, MAE=${mae_te:,.0f}, WAPE={wape_te:.2f}%")

    m_eval = df_test.groupby('Mes_Año').agg(
        Real=('Ingresos_Totales', 'sum'),
        Pred=('y_pred', 'sum'),
        Spend_Meta=('Spend_Meta', 'sum'),
        Spend_Google=('Spend_Google', 'sum')
    ).reset_index()
    m_eval['Error_Abs'] = np.abs(m_eval['Real'] - m_eval['Pred'])
    m_eval['Error_Pct'] = (m_eval['Error_Abs'] / m_eval['Real']) * 100
    m_eval['Precision_Pct'] = 100 - m_eval['Error_Pct']

    tot_r = m_eval['Real'].sum()
    tot_p = m_eval['Pred'].sum()
    wape_m = (m_eval['Error_Abs'].sum() / tot_r) * 100
    print(f"🎯 TEST MENSUAL ACUMULADO 2026: Real=${tot_r:,.0f}, Pred=${tot_p:,.0f}, WAPE={wape_m:.2f}%, Precisión Global={100-wape_m:.2f}%\n")

    print("DETALLE MENSUAL OUT-OF-SAMPLE 2026:")
    print("-" * 75)
    for idx, r in m_eval.iterrows():
        print(f"  {r['Mes_Año']}: Real=${r['Real']:>11,.0f} | Pred=${r['Pred']:>11,.0f} | Error=${r['Error_Abs']:>10,.0f} ({r['Error_Pct']:>4.1f}%) | Precisión={r['Precision_Pct']:>5.1f}%")
    print("-" * 75)
    return p_opt


def entrenar_modelo_pymc_marketing(df_mmm, chains=4, draws=1500, tune=1000):
    """
    Entrena el modelo DelayedSaturatedMMM en PyMC-Marketing:
    - Prior vectorizado en gamma_control para alinear dimensionalidad exacta con los 10 controles exógenos.
    - Priors de media saturación basados en el gasto real del negocio.
    """
    try:
        from pymc_marketing.mmm import DelayedSaturatedMMM
        import arviz as az
    except ImportError:
        print("⚠️ PyMC-Marketing o ArviZ no están instalados en este entorno.")
        print("   Para ejecutar el muestreo bayesiano MCMC instala con: pip install pymc-marketing arviz")
        return None, None

    canales_pauta = ['Spend_Meta', 'Spend_Google']
    controles_exogenos = [
        'Es_Quincena_15',
        'Es_Fin_Mes_30',
        'Es_Fin_De_Semana',
        'Es_Aniversario',
        'Es_Semana_Santa',
        'Profundidad_Descuento_pct',
        'Es_Macro_Evento',
        'Es_Efeméride_Comercial',
        'Mecanica_GWP_Obsequio',
        'Promo_Toda_La_Tienda'
    ]

    X = df_mmm[canales_pauta + controles_exogenos]
    y = df_mmm['Ingresos_Totales']

    mean_spend_meta = df_mmm['Spend_Meta'].replace(0, np.nan).mean()
    mean_spend_google = df_mmm['Spend_Google'].replace(0, np.nan).mean()

    # Configuración de Priors Bayesianos con dimensionalidad vectorizada exacta (10 elementos en gamma_control)
    custom_priors = {
        # Adstock: retención de 2 a 4 días en Meta y 0 a 1 día en Google Search
        "adstock_alpha": {
            "dist": "Beta",
            "kwargs": {
                "alpha": [2.0, 1.0], 
                "beta": [2.0, 3.0]
            }
        },
        # Saturación Hill: Sigmas escalados a la inversión promedio diaria
        "saturation_lam": {
            "dist": "HalfNormal",
            "kwargs": {
                "sigma": [
                    mean_spend_meta * 1.5 if pd.notnull(mean_spend_meta) else 1_500_000.0,
                    mean_spend_google * 1.5 if pd.notnull(mean_spend_google) else 500_000.0
                ]
            }
        },
        # Controles Exógenos y Promocionales: Vector de 10 dimensiones
        "gamma_control": {
            "dist": "Normal",
            "kwargs": {
                "mu": [
                    500_000.0,    # Es_Quincena_15
                    500_000.0,    # Es_Fin_Mes_30
                    200_000.0,    # Es_Fin_De_Semana
                    2_500_000.0,  # Es_Aniversario
                    -900_000.0,   # Es_Semana_Santa
                    1_500_000.0,  # Profundidad_Descuento_pct
                    800_000.0,    # Es_Macro_Evento
                    300_000.0,    # Es_Efeméride_Comercial
                    200_000.0,    # Mecanica_GWP_Obsequio
                    400_000.0     # Promo_Toda_La_Tienda
                ],
                "sigma": [
                    1_000_000.0,  # Es_Quincena_15
                    1_000_000.0,  # Es_Fin_Mes_30
                    500_000.0,    # Es_Fin_De_Semana
                    3_000_000.0,  # Es_Aniversario
                    1_200_000.0,  # Es_Semana_Santa
                    2_000_000.0,  # Profundidad_Descuento_pct
                    1_500_000.0,  # Es_Macro_Evento
                    1_000_000.0,  # Es_Efeméride_Comercial
                    1_000_000.0,  # Mecanica_GWP_Obsequio
                    1_000_000.0   # Promo_Toda_La_Tienda
                ]
            }
        }
    }

    print(f"\n🚀 Iniciando entrenamiento MCMC con PyMC-Marketing ({chains} cadenas, {draws} draws)...")
    mmm = DelayedSaturatedMMM(
        date_column="Fecha_Clean",
        channel_columns=canales_pauta,
        control_columns=controles_exogenos,
        adstock_max_lag=14,
        custom_priors=custom_priors
    )

    idata = mmm.fit(X=X, y=y, chains=chains, draws=draws, tune=tune, target_accept=0.95)
    
    # Diagnóstico de convergencia Gelman-Rubin
    summary_rhat = az.summary(idata, var_names=["adstock_alpha", "saturation_lam", "gamma_control"])
    print("\n--- RESUMEN DE CONVERGENCIA BAYESIANA (Gelman-Rubin R̂) ---")
    print(summary_rhat[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']])
    
    return mmm, idata


if __name__ == '__main__':
    # 1. Preparación de la matriz diaria con todas las banderas exógenas (Aniversario + Semana Santa)
    df_mmm = preparar_matriz_diaria_mmm()
    
    # 2. Ejecutar auditoría de exactitud econométrica
    p_opt = auditar_exactitud_modelo(df_mmm)

    # 3. Ejecutar optimización de presupuesto anual 2027 (KKT Equimarginal)
    meta_objetivo = 1_800_000_000.0 # $1,800M COP
    resultado = optimizar_presupuesto_2027(meta_ventas_anual=meta_objetivo)
    imprimir_reporte_asignacion(resultado)

    # 4. Ejecutar optimización mensual independiente para cada mes de 2027
    df_mensual_2027 = optimizar_presupuesto_mensual_2027()
    print("\n" + "="*80)
    print("📅 ASIGNACIÓN PRESUPUESTAL MENSUAL INDEPENDIENTE 2027 (SLSQP / KKT)")
    print("="*80)
    for idx, r in df_mensual_2027.iterrows():
        print(f"{r['Mes']:<10} ({r['Dias']}d) | Meta: COP ${r['Meta_Ventas']:>11,.0f} | Base: COP ${r['Base_Organica']:>10,.0f} | Brecha: COP ${r['Brecha_Pauta']:>10,.0f} | Meta Ads: COP ${r['Spend_Meta_Mes']:>10,.0f} ({r['Share_Meta']:>4.1f}%) | Google Ads: COP ${r['Spend_Google_Mes']:>10,.0f} ({r['Share_Google']:>4.1f}%) | Total: COP ${r['Presupuesto_Total_Mes']:>10,.0f} | ROAS: {r['ROAS_Global']:>4.2f}x")
    print("="*80)


