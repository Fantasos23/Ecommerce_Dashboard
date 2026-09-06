import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
# Importar funciones de procesamiento ETL
from etl_procesamiento import unificar_carpeta
from procesar_vtex_agrupado import generar_dataset_vtex_por_orden
from meta_api_sync import sincronizar_meta_ads
from vtex_api_sync import sincronizar_vtex_orders

warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(
    page_title="Dashboard Integral de Ventas & Atribución",
    page_icon="🚀",
    layout="wide"
)

# Directorios de Archivos
DATASETS_DIR = Path('datasets_procesados')
VTEX_AGRUPADO_PATH = DATASETS_DIR / 'dataset_vtex_agrupado_ordenes.csv'
VTEX_SKU_PATH = DATASETS_DIR / 'dataset_vtex_detalle_skus.csv' 
META_PATH = DATASETS_DIR / 'dataset_meta_unificado.csv'
GOOGLE_PATH = DATASETS_DIR / 'dataset_google_unificado.csv'

# Inicializar session_state si no existe
if 'custom_mappings' not in st.session_state:
    st.session_state.custom_mappings = {'Canal': {}, 'Origen': {}}

st.markdown("""
<style>
    /* 1. Chips de selección en multiselect */
    span[data-baseweb="tag"] {
        background-color: #334155 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    
    /* 2. Ocultar Colorbar redundante en Plotly */
    .coloraxis {
        display: none !important;
    }
    
    /* 3. Contenedor de Sidebar */
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    /* 4. Sistema de Tarjetas KPI Modernas (Tema Claro Predeterminado) */
    .kpi-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        min-height: 125px !important;
        box-sizing: border-box !important;
    }
    .kpi-header {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-bottom: 4px !important;
    }
    .kpi-title {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        display: flex !important;
        align-items: center !important;
        gap: 5px !important;
    }
    .badge-pill-red {
        background-color: #fee2e2 !important;
        color: #ef4444 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        padding: 2px 7px !important;
        border-radius: 12px !important;
        white-space: nowrap !important;
    }
    .badge-pill-green {
        background-color: #dcfce7 !important;
        color: #16a34a !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        padding: 2px 7px !important;
        border-radius: 12px !important;
        white-space: nowrap !important;
    }
    .badge-pill-gray {
        background-color: #f1f5f9 !important;
        color: #64748b !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        padding: 2px 7px !important;
        border-radius: 12px !important;
        white-space: nowrap !important;
    }
    .kpi-value {
        font-size: 22px !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        line-height: 1.2 !important;
        margin: 4px 0 4px 0 !important;
        letter-spacing: -0.5px !important;
    }
    .kpi-sub {
        font-size: 12px !important;
        color: #64748b !important;
        font-weight: 500 !important;
        line-height: 1.3 !important;
    }

    /* 5. Banners de Clientes Nuevos / Recurrentes */
    .client-banner-blue {
        background-color: #f0f6ff !important;
        border: 1px solid #dbeafe !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
        box-sizing: border-box !important;
    }
    .client-banner-green {
        background-color: #f0fdf4 !important;
        border: 1px solid #dcfce7 !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
        box-sizing: border-box !important;
    }

    /* 6. Títulos de Sección Estilizados */
    .section-title {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin: 18px 0 12px 0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    /* Soporte Tema Oscuro EXPLÍCITO de Streamlit (data-theme="dark") */
    [data-theme="dark"] .kpi-card,
    .stApp[data-theme="dark"] .kpi-card,
    div[data-testid="stAppViewContainer"][data-theme="dark"] .kpi-card {
        background: #1e293b !important;
        border-color: #334155 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3) !important;
    }
    [data-theme="dark"] .kpi-title,
    .stApp[data-theme="dark"] .kpi-title {
        color: #94a3b8 !important;
    }
    [data-theme="dark"] .kpi-value,
    .stApp[data-theme="dark"] .kpi-value {
        color: #f8fafc !important;
    }
    [data-theme="dark"] .kpi-sub,
    .stApp[data-theme="dark"] .kpi-sub {
        color: #94a3b8 !important;
    }
    [data-theme="dark"] .client-banner-blue,
    .stApp[data-theme="dark"] .client-banner-blue {
        background-color: #0f172a !important;
        border-color: #1e3a8a !important;
    }
    [data-theme="dark"] .client-banner-green,
    .stApp[data-theme="dark"] .client-banner-green {
        background-color: #062e1c !important;
        border-color: #065f46 !important;
    }
    [data-theme="dark"] .section-title,
    .stApp[data-theme="dark"] .section-title {
        color: #f8fafc !important;
    }
    [data-theme="dark"] .direct-card,
    .stApp[data-theme="dark"] .direct-card {
        background: #1e293b !important;
        border-color: #334155 !important;
    }
    [data-theme="dark"] .direct-card-val,
    .stApp[data-theme="dark"] .direct-card-val {
        color: #f8fafc !important;
    }
    [data-theme="dark"] .direct-card-badge,
    .stApp[data-theme="dark"] .direct-card-badge {
        background: #0f172a !important;
        border-color: #334155 !important;
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 1. MAPEOS Y REGLAS DE NEGOCIO (DICCIONARIOS)
# ==========================================================
DICT_CANALES = {
    'Facebook': ['facebook', 'fb', 'facebook-sitelink', 'fb-sitelink', 'facebookcpa', 'facebook-sitelink-5', 
                 'fb-sitelink-3', 'fb-sitelink-6', 'fb-sitelink-1', 'fb-sitelink-2', 'facebook_salonini', 
                 'facebook_salonin', 'trafico_andrea'],
    'Instagram': ['instagram', 'ig', 'igshopping'],
    'Google': ['google', 'google&utm_medium=cpc'],
    'YouTube': ['youtube'],
    'TikTok': ['tik tok', 'tiktok', 'tik_tok'],
    'HubSpot': ['hs_email', 'hs', 'hs_automation'],
    'Connectif': ['connectif'],
    'Icommarketing': ['icommarketing'],
    'VTEX': ['vtex', 'vtexcem'],
    'General': ['web', 'quiz'],
    'Nequi': ['nequi_app'],
    'AI': ['chatgpt.com', 'copilot.com'],
    'Directo / Sin Datos': ['nan', '', 'none', 'null']
}

DICT_ORIGENES = {
    'Publicidad (Pauta)': ['cpc', 'cpa', 'cpa+', 'cpm', 'paid', 'facebook', 'conversion', 'trafico'],
    'Orgánico / Botones Web': ['boton_tienda_superior', 'boton_tienda_inferior', 'boton_superior_tienda', 
                              'boton_inferior_tienda', 'boton_tienda', 'boton_superior_tienda_col', 'boton_inferior_tienda_col'],
    'Enlaces En Redes': ['linktree', 'social', 'content_creator'],
    'Email / Push': ['email', 'mail', 'push', 'webpush', 'abandono_carrinho', 'vtex'],
    'Alianzas': ['nequi'],
    'Directo': ['nan', '', 'none', 'null']
}

DICT_MARCAS_VARIANTES = {
    'SalonIn': ['salonin', 'salon in', 'vegan keratin collagen', 'VEGAN KERATIN'],
    'Sol Eclair': ['sol eclair', 'soleclair', 'sol-eclair'],
    'Green Code': ['green code', 'greencode'],
    'Luminance': ['luminance'],
    'Vitane': ['vitane'],
    'Muss': ['muss'],
    'Bacterion': ['bacterion'],
    'Tanga': ['tanga'],
    'CHAPSTICK': ['chapstick', 'chap stick'],
    'Deo Pies': ['deo pies', 'deopies'],
    'Coloriss': ['coloriss'],
    'Kleer Lac': ['kleer lac', 'kleerlac']
}

MARCAS_LISTA = list(DICT_MARCAS_VARIANTES.keys())

# ==========================================================
# 2. FUNCIONES AUXILIARES
# ==========================================================
def clasificar_valor(val, diccionario, custom_dict, default='Otros / No Asignados'):
    val_clean = str(val).lower().strip()
    if val_clean in custom_dict:
        return custom_dict[val_clean]
    for categoria, patrones in diccionario.items():
        if any(p in val_clean for p in patrones):
            return categoria
    return default

def formatear_cifra_corta(valor, es_moneda=True):
    simbolo = "$" if es_moneda else ""
    cifra_exacta = f"{simbolo}{valor:,.0f}"
    abs_val = abs(valor)
    
    if abs_val >= 1_000_000_000:
        corta = f"{simbolo}{valor / 1_000_000_000:.2f} Bill"
    elif abs_val >= 1_000_000:
        corta = f"{simbolo}{valor / 1_000_000:.2f} Mill"
    elif abs_val >= 100_000:
        corta = f"{simbolo}{valor / 1_000:.1f} Mil"
    else:
        corta = cifra_exacta
        
    return corta, cifra_exacta

def formatear_cifra_compacta(valor, es_moneda=False):
    abs_v = abs(valor)
    simb = "$" if es_moneda else ""
    if abs_v >= 1_000_000_000:
        return f"{simb}{valor / 1_000_000_000:.1f}B"
    elif abs_v >= 1_000_000:
        if abs_v >= 10_000_000:
            return f"{simb}{valor / 1_000_000:.0f}M"
        else:
            return f"{simb}{valor / 1_000_000:.1f}M"
    elif abs_v >= 1_000:
        return f"{simb}{valor / 1_000:.0f}k"
    else:
        return f"{simb}{valor:,.0f}"

def asignar_marca(sku):
    sku_clean = str(sku).lower()
    for marca_oficial, variantes in DICT_MARCAS_VARIANTES.items():
        if any(variante in sku_clean for variante in variantes):
            return marca_oficial
    return 'Otros / Sin Marca'

# --- OBTENER TOP PRODUCTOS (USANDO DATASET DETALLE SKU) ---
def obtener_top_productos(df_sku_filtrado, marca_filtro="Todas", top_n=10):
    if df_sku_filtrado.empty or 'SKU Name' not in df_sku_filtrado.columns:
        return pd.DataFrame()

    df_exp = df_sku_filtrado.copy()
    df_exp['SKU_Individual'] = df_exp['SKU Name'].astype(str).str.strip()

    df_exp['Marca_Detectada'] = df_exp['SKU_Individual'].apply(asignar_marca)

    if marca_filtro == "Otros":
        df_exp = df_exp[df_exp['Marca_Detectada'] == 'Otros / Sin Marca']
    elif marca_filtro != "Todas":
        df_exp = df_exp[df_exp['Marca_Detectada'] == marca_filtro]

    col_val = 'Total_Value_SKU' if 'Total_Value_SKU' in df_exp.columns else 'Total Value'

    df_ranking = df_exp.groupby(['SKU_Individual', 'Marca_Detectada']).agg(
        Ordenes=('Order', 'nunique'),
        Unidades=('Quantity_SKU', 'sum'),
        Ingresos=(col_val, 'sum')
    ).reset_index()

    return df_ranking.sort_values(by=['Unidades', 'Ingresos'], ascending=[False, False]).head(top_n)

def limpiar_cadena_descuentos(cadena_raw):
    if not cadena_raw or str(cadena_raw).lower() in ['nan', 'none', 'sin descuento', '']:
        return "Sin Descuento"
    elementos = [e.strip() for e in str(cadena_raw).replace('|', ',').split(',')]
    elementos_unicos = list(dict.fromkeys([e for e in elementos if e and e.lower() != 'nan']))
    return " + ".join(elementos_unicos) if elementos_unicos else "Sin Descuento"

# --- GRAFICO PARETO (USANDO DATASET DETALLE SKU) ---
def generar_grafico_pareto(df_sku_filtrado):
    if df_sku_filtrado.empty or 'SKU Name' not in df_sku_filtrado.columns:
        return None

    df_exp = df_sku_filtrado.copy()
    df_exp['SKU_Individual'] = df_exp['SKU Name'].astype(str).str.strip()

    col_val = 'Total_Value_SKU' if 'Total_Value_SKU' in df_exp.columns else 'Total Value'

    df_pareto = df_exp.groupby('SKU_Individual')[col_val].sum().reset_index()
    df_pareto = df_pareto.sort_values(by=col_val, ascending=False).reset_index(drop=True)
    
    total_ventas = df_pareto[col_val].sum()
    if total_ventas == 0:
        return None
        
    df_pareto['Ingreso_Acumulado'] = df_pareto[col_val].cumsum()
    df_pareto['Pct_Acumulado'] = (df_pareto['Ingreso_Acumulado'] / total_ventas) * 100
    
    df_top = df_pareto.head(15).copy()
    df_top['SKU_Corto'] = df_top['SKU_Individual'].apply(lambda x: x[:22] + '...' if len(x) > 25 else x)

    pct_alcanzado = df_top['Pct_Acumulado'].max()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=df_top['SKU_Corto'], 
            y=df_top[col_val],
            name="Ventas ($)",
            marker_color='#2563eb',
            customdata=df_top['SKU_Individual'],
            hovertemplate="<b>Producto:</b> %{customdata}<br><b>Ventas Reales:</b> $%{y:,.0f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_top['SKU_Corto'], 
            y=df_top['Pct_Acumulado'],
            name="% Acumulado",
            line=dict(color='#f59e0b', width=3),
            mode='lines+markers',
            hovertemplate="<b>% Acumulado:</b> %{y:.1f}%<extra></extra>"
        ),
        secondary_y=True
    )

    max_y_2 = min(100, max(60, int(pct_alcanzado + 15)))

    if pct_alcanzado >= 80:
        fig.add_shape(
            type="line", x0=-0.5, x1=len(df_top)-0.5, y0=80, y1=80,
            yref="y2", line=dict(color="red", width=2, dash="dash")
        )

    fig.update_layout(
        hovermode="x unified",
        height=420,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=100),
        xaxis=dict(tickangle=-45)
    )
    fig.update_yaxes(title_text="Ingresos ($ COP)", secondary_y=False)
    fig.update_yaxes(title_text="% Acumulado", secondary_y=True, range=[0, max_y_2])
    
    return fig, pct_alcanzado

def generar_matriz_cohortes(df_completo_vtex):
    if df_completo_vtex.empty or 'Client Document' not in df_completo_vtex.columns:
        return pd.DataFrame()

    df_c = df_completo_vtex.dropna(subset=['Client Document', 'Creation Date']).copy()
    df_c['Client Document'] = df_c['Client Document'].astype(str)
    
    df_c['Order_Month'] = df_c['Creation Date'].dt.to_period('M').astype(str)
    df_c['Cohort_Month'] = df_c.groupby('Client Document')['Creation Date'].transform('min').dt.to_period('M').astype(str)

    df_cohort_data = df_c.groupby(['Cohort_Month', 'Order_Month']).agg(Clientes=('Client Document', 'nunique')).reset_index()
    
    df_cohort_data['Periodo_Mes'] = (
        pd.to_datetime(df_cohort_data['Order_Month']).dt.to_period('M') - 
        pd.to_datetime(df_cohort_data['Cohort_Month']).dt.to_period('M')
    ).apply(lambda x: x.n)

    cohort_pivot = df_cohort_data.pivot(index='Cohort_Month', columns='Periodo_Mes', values='Clientes')
    
    if cohort_pivot.empty:
        return pd.DataFrame()

    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100

    return retention_matrix

def generar_grafico_tiempo_entre_compras(df_completo_vtex):
    if df_completo_vtex.empty or 'Client Document' not in df_completo_vtex.columns:
        return None

    df_frec = df_completo_vtex.dropna(subset=['Client Document', 'Creation Date']).copy()
    df_frec['Client Document'] = df_frec['Client Document'].astype(str)
    
    df_frec = df_frec.sort_values(['Client Document', 'Creation Date'])
    df_frec['Num_Orden_Cliente'] = df_frec.groupby('Client Document').cumcount() + 1
    
    df_frec['Fecha_Previo'] = df_frec.groupby('Client Document')['Creation Date'].shift(1)
    df_frec['Dias_Entre_Compras'] = (df_frec['Creation Date'] - df_frec['Fecha_Previo']).dt.days

    df_recompras = df_frec[df_frec['Num_Orden_Cliente'].isin([2, 3, 4, 5])].copy()

    if df_recompras.empty:
        return None

    df_promedios = df_recompras.groupby('Num_Orden_Cliente')['Dias_Entre_Compras'].mean().reset_index()
    
    etiquetas_map = {
        2: "1ª ➔ 2ª Compra",
        3: "2ª ➔ 3ª Compra",
        4: "3ª ➔ 4ª Compra",
        5: "4ª ➔ 5ª Compra"
    }
    df_promedios['Etiqueta'] = df_promedios['Num_Orden_Cliente'].map(etiquetas_map)

    fig = px.bar(
        df_promedios,
        x='Etiqueta',
        y='Dias_Entre_Compras',
        text_auto='.0f',
        labels={'Etiqueta': 'Salto de Recompra', 'Dias_Entre_Compras': 'Días Promedio'},
        color_discrete_sequence=['#3b82f6']
    )
    fig.update_traces(
        textposition='outside',
        hovertemplate="<b>Transición:</b> %{x}<br><b>Tiempo Promedio:</b> %{y:.1f} días<extra></extra>"
    )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Días Promedio"
    )
    return fig

# ==========================================================
# 3. CARGA Y PREPARACIÓN DE DATOS
# ==========================================================
@st.cache_data
def cargar_datos_vtex():
    if not VTEX_AGRUPADO_PATH.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(VTEX_AGRUPADO_PATH, low_memory=False)
    df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce').dt.floor('D')
    df = df.dropna(subset=['Creation Date'])
    
    df['Fecha_Clean'] = df['Creation Date'].dt.strftime('%Y-%m-%d')
    df['Año'] = df['Creation Date'].dt.year
    df['Mes_Num'] = df['Creation Date'].dt.month
    df['Día'] = df['Creation Date'].dt.date
    df['Quarter_Num'] = df['Creation Date'].dt.quarter
    df['Quarter'] = 'Q' + df['Quarter_Num'].astype(str)
    
    meses_esp = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
    df['Mes_Nombre'] = df['Mes_Num'].map(meses_esp)
    
    df['Total Value'] = pd.to_numeric(df['Total Value'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    df['Quantity_SKU'] = pd.to_numeric(df['Quantity_SKU'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    df['Discounts Names'] = df['Discounts Names'].fillna('Sin Descuento').astype(str)
    
    return df

@st.cache_data
def cargar_datos_sku():
    """
    Carga directamente el 'dataset_vtex_detalle_skus.csv' generado por el ETL,
    utilizando las columnas limpias 'SKU Name', 'Quantity_SKU' y 'Total_Value_SKU'.
    """
    if not VTEX_SKU_PATH.exists():
        return pd.DataFrame()
    
    df_sku = pd.read_csv(VTEX_SKU_PATH, low_memory=False)
    
    # 1. Parseo de Fechas
    if 'Creation Date' in df_sku.columns:
        df_sku['Creation Date'] = pd.to_datetime(df_sku['Creation Date'], format='mixed', dayfirst=True, errors='coerce').dt.floor('D')
        df_sku = df_sku.dropna(subset=['Creation Date'])
        df_sku['Fecha_Clean'] = df_sku['Creation Date'].dt.strftime('%Y-%m-%d')
        df_sku['Año'] = df_sku['Creation Date'].dt.year
        df_sku['Mes_Num'] = df_sku['Creation Date'].dt.month
        df_sku['Quarter_Num'] = df_sku['Creation Date'].dt.quarter
        
        meses_esp = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
        df_sku['Mes_Nombre'] = df_sku['Mes_Num'].map(meses_esp)

    # 2. Asegurar tipos numéricos limpios del ETL
    if 'Quantity_SKU' in df_sku.columns:
        df_sku['Quantity_SKU'] = pd.to_numeric(df_sku['Quantity_SKU'], errors='coerce').fillna(0)
    
    # La columna creada por tu ETL se llama 'Total_Value_SKU'
    if 'Total_Value_SKU' in df_sku.columns:
        df_sku['Total_Value_SKU'] = pd.to_numeric(df_sku['Total_Value_SKU'], errors='coerce').fillna(0)
    elif 'Total Value' in df_sku.columns:
        df_sku['Total_Value_SKU'] = pd.to_numeric(df_sku['Total Value'], errors='coerce').fillna(0)

    # 3. Sanitizar SKU Name por seguridad
    if 'SKU Name' in df_sku.columns:
        df_sku['SKU Name'] = df_sku['SKU Name'].astype(str).str.strip()
        df_sku = df_sku[~df_sku['SKU Name'].str.lower().isin(['nan', 'none', '', 'sin sku', 'null'])]

    return df_sku

@st.cache_data
def cargar_inversion_ads():
    df_meta, df_google = pd.DataFrame(), pd.DataFrame()

    def limpiar_numero_col(serie, default=0.0):
        def _parse(val):
            if pd.isna(val) or val == '':
                return default
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).replace('$', '').strip()
            if not s or s.lower() in ['nan', 'none', 'null']:
                return default
            # Si tiene formato europeo/español con punto de miles y coma decimal (ej: 1.234,56 o 1.234)
            if ',' in s:
                s = s.replace('.', '').replace(',', '.')
            elif s.count('.') > 1: # Formato miles con puntos múltiples (ej: 1.234.567)
                s = s.replace('.', '')
            try:
                return float(s)
            except ValueError:
                return default
        return serie.apply(_parse)

    # 1. Google Ads
    if GOOGLE_PATH.exists():
        df_g = pd.read_csv(GOOGLE_PATH, low_memory=False)
        col_fecha_g = next((c for c in ['Día', 'Dia', 'Day', 'Fecha'] if c in df_g.columns), None)
        
        if col_fecha_g:
            fechas_parsed = pd.to_datetime(df_g[col_fecha_g], errors='coerce').dt.floor('D')
            df_g['Fecha_Clean'] = fechas_parsed.dt.strftime('%Y-%m-%d')
            df_g['Inversion'] = limpiar_numero_col(df_g['Coste']) if 'Coste' in df_g.columns else 0.0
            
            # Limpieza de métricas operativas Google
            df_g['Impresiones'] = limpiar_numero_col(df_g['Impr.']) if 'Impr.' in df_g.columns else 0.0
            df_g['Clics'] = limpiar_numero_col(df_g['Clics']) if 'Clics' in df_g.columns else 0.0
            df_g['Compras'] = limpiar_numero_col(df_g['Resultados']) if 'Resultados' in df_g.columns else 0.0
            
            df_google = df_g.dropna(subset=['Fecha_Clean'])

    # 2. Meta Ads
    if META_PATH.exists():
        df_m = pd.read_csv(META_PATH, low_memory=False)
        col_fecha_m = next((c for c in ['Inicio del informe', 'Reporting starts', 'Day', 'Día', 'Fecha'] if c in df_m.columns), None)

        if col_fecha_m:
            fechas_parsed = pd.to_datetime(df_m[col_fecha_m], errors='coerce').dt.floor('D')
            df_m['Fecha_Clean'] = fechas_parsed.dt.strftime('%Y-%m-%d')
            df_m['Inversion'] = limpiar_numero_col(df_m['Importe gastado (COP)']) if 'Importe gastado (COP)' in df_m.columns else 0.0
            
            # Limpieza de métricas operativas Meta
            df_m['Impresiones'] = limpiar_numero_col(df_m['Impresiones']) if 'Impresiones' in df_m.columns else 0.0
            df_m['Clics'] = limpiar_numero_col(df_m['Clics en el enlace']) if 'Clics en el enlace' in df_m.columns else 0.0
            df_m['Visitas_LP'] = limpiar_numero_col(df_m['Visitas a la página de destino']) if 'Visitas a la página de destino' in df_m.columns else 0.0
            df_m['Carrito'] = limpiar_numero_col(df_m['Artículos agregados al carrito']) if 'Artículos agregados al carrito' in df_m.columns else 0.0
            df_m['Compras'] = limpiar_numero_col(df_m['Compras']) if 'Compras' in df_m.columns else 0.0
            df_m['Frecuencia'] = limpiar_numero_col(df_m['Frecuencia'], default=1.0) if 'Frecuencia' in df_m.columns else 1.0
            
            df_meta = df_m.dropna(subset=['Fecha_Clean'])

    return df_meta, df_google

# ==========================================================
# FUNCIONES CORREGIDAS PARA FUNNELS Y DIAGNÓSTICO DE PAUTA
# ==========================================================
def generar_funnel_meta(df_meta_f):
    """Genera el gráfico de embudo de conversión para Meta Ads omite Carritos."""
    if df_meta_f.empty:
        return None
    
    imp = df_meta_f['Impresiones'].sum() if 'Impresiones' in df_meta_f.columns else 0
    clics = df_meta_f['Clics'].sum() if 'Clics' in df_meta_f.columns else 0
    lp = df_meta_f['Visitas_LP'].sum() if 'Visitas_LP' in df_meta_f.columns else 0
    compras = df_meta_f['Compras'].sum() if 'Compras' in df_meta_f.columns else 0

    etiquetas = ['Impresiones', 'Clics Enlace', 'Visitas LP', 'Compras']
    valores = [imp, clics, lp, compras]

    if max(valores) == 0:
        return None

    fig = go.Figure(go.Funnel(
        y=etiquetas,
        x=valores,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=["#1e3a8a", "#1d4ed8", "#2563eb", "#10b981"])
    ))
    
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, -apple-system, system-ui, sans-serif")
    )
    return fig


def generar_funnel_google(df_google_f):
    """Genera el gráfico de embudo de conversión exacto para Google Ads."""
    if df_google_f.empty:
        return None

    imp = df_google_f['Impresiones'].sum() if 'Impresiones' in df_google_f.columns else 0
    clics = df_google_f['Clics'].sum() if 'Clics' in df_google_f.columns else 0
    compras = df_google_f['Compras'].sum() if 'Compras' in df_google_f.columns else 0

    if max([imp, clics, compras]) == 0:
        return None

    fig = go.Figure(go.Funnel(
        y=['Impresiones (Impr.)', 'Clics', 'Compras (Resultados)'],
        x=[imp, clics, compras],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=["#c2410c", "#f97316", "#10b981"])
    ))
    
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, -apple-system, system-ui, sans-serif")
    )
    return fig


def generar_scatter_ctr_compras(df_meta_f):
    """Matriz Cuadrante (Scatter Plot): CTR vs Compras por Anuncio/Campaña en Meta."""
    if df_meta_f.empty:
        return None

    # Buscar columna identificadora de anuncio o campaña
    col_ad = next((c for c in df_meta_f.columns if c.lower() in ['nombre del anuncio', 'ad_name', 'nombre de la campaña', 'campaign_name', 'campaña']), None)

    df_meta_f['CTR_Calc'] = np.where(df_meta_f['Impresiones'] > 0, (df_meta_f['Clics'] / df_meta_f['Impresiones']) * 100, 0)

    if col_ad:
        df_ads = df_meta_f.groupby(col_ad).agg(
            CTR=('CTR_Calc', 'mean'),
            Compras=('Compras', 'sum')
        ).reset_index()
    else:
        # Si no existe la columna de anuncio individual, agrupa por fecha
        df_ads = df_meta_f.groupby('Fecha_Clean').agg(
            CTR=('CTR_Calc', 'mean'),
            Compras=('Compras', 'sum')
        ).reset_index().rename(columns={'Fecha_Clean': 'Etiqueta'})
        col_ad = 'Etiqueta'

    df_ads = df_ads[(df_ads['CTR'] > 0) | (df_ads['Compras'] > 0)]

    if df_ads.empty:
        return None

    mediana_ctr = df_ads['CTR'].median()
    mediana_compras = df_ads['Compras'].median()

    fig = px.scatter(
        df_ads,
        x='CTR',
        y='Compras',
        hover_name=col_ad,
        size=df_ads['Compras'] + 1,
        color='Compras',
        color_continuous_scale='Blues',
        labels={'CTR': 'CTR (%)', 'Compras': 'Compras (Unidades)'}
    )

    fig.add_vline(x=mediana_ctr, line_dash="dash", line_color="#9ca3af")
    fig.add_hline(y=mediana_compras, line_dash="dash", line_color="#9ca3af")

    fig.update_layout(
        height=420,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, -apple-system, system-ui, sans-serif")
    )
    return fig


def generar_linea_frecuencia_conversion(df_meta_f):
    """Gráfico de Fatiga de Audiencia: Frecuencia vs CTR."""
    if df_meta_f.empty or 'Frecuencia' not in df_meta_f.columns:
        return None

    df_meta_f['Frec_Rango'] = df_meta_f['Frecuencia'].round(1)
    df_meta_f['CTR_Calc'] = np.where(df_meta_f['Impresiones'] > 0, (df_meta_f['Clics'] / df_meta_f['Impresiones']) * 100, 0)
    
    df_frec = df_meta_f.groupby('Frec_Rango').agg(CTR_Promedio=('CTR_Calc', 'mean')).reset_index()
    df_frec = df_frec[df_frec['Frec_Rango'] > 0].sort_values('Frec_Rango')

    if df_frec.empty:
        return None

    fig = px.line(
        df_frec,
        x='Frec_Rango',
        y='CTR_Promedio',
        markers=True,
        labels={'Frec_Rango': 'Frecuencia Acumulada', 'CTR_Promedio': 'CTR Promedio (%)'}
    )
    fig.update_traces(line_color='#ef4444', line_width=3, marker_size=8)
    
    # Línea guía de alerta en Frecuencia 2.5
    fig.add_vline(x=2.5, line_dash="dot", line_color="#dc2626", annotation_text="⚠️ Saturación (~2.5)")

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, -apple-system, system-ui, sans-serif")
    )
    return fig

# ==========================================================
# 4. ENCABEZADO Y PROCESAMIENTO
# ==========================================================
st.title("Data Tienda Colombia")

col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns([1.2, 1, 1, 1, 1])
with col_b1:
    st.caption("Consolidado por Órdenes Únicas, Canales, Descuentos e Inversión.")

with col_b2:
    if st.button("🔄 Sync VTEX API", type="primary", use_container_width=True, help="Descarga e incrementa los pedidos más recientes desde la API de VTEX OMS"):
        with st.spinner("Sincronizando pedidos desde VTEX API..."):
            try:
                res = sincronizar_vtex_orders()
                st.cache_data.clear()
                st.success(f"¡VTEX sincronizado! (+{res.get('nuevas_ordenes', 0)} órdenes)")
                st.rerun()
            except Exception as e:
                st.error(f"Error sincronizando VTEX: {e}")

with col_b3:
    if st.button("🔄 Sync Meta API", type="secondary", use_container_width=True, help="Descarga e incrementa los datos más recientes desde Meta Ads Graph API"):
        with st.spinner("Sincronizando con Meta Ads API..."):
            try:
                res = sincronizar_meta_ads()
                st.cache_data.clear()
                st.success(f"¡Meta Ads sincronizado! (Hasta {res.get('ultima_fecha', 'hoy')})")
                st.rerun()
            except Exception as e:
                st.error(f"Error sincronizando Meta: {e}")

with col_b4:
    if st.button("1.Unificar ETL", type="secondary", use_container_width=True, help="Unifica todos los archivos de VTEX, Meta y Google"):
        with st.spinner("Unificando archivos..."):
            unificar_carpeta('VTEX', ';', 'Order')
            unificar_carpeta('Meta', ',')
            unificar_carpeta('Google', ',')
            st.cache_data.clear()
        st.success("¡Completado!")
        st.rerun()

with col_b5:
    if st.button("2.Agrupar Clientes", type="secondary", use_container_width=True, help="Agrupa a nivel de cliente y orden única"):
        with st.spinner("Agrupando a nivel de Orden Única..."):
            generar_dataset_vtex_por_orden()
            st.cache_data.clear()
        st.success("¡Agregación lista!")
        st.rerun()

st.divider()

# Cargar datasets
df_vtex = cargar_datos_vtex()
df_sku = cargar_datos_sku()
df_meta, df_google = cargar_inversion_ads()

if df_vtex.empty:
    st.info("Haz clic en los botones superiores para procesar y agrupar los datos.")
    st.stop()

# ==========================================================
# 5. RECLASIFICACIÓN DE CANALES/ORÍGENES
# ==========================================================
df_vtex['Canal_Estandar'] = df_vtex['UtmSource'].apply(
    lambda x: clasificar_valor(x, DICT_CANALES, st.session_state.custom_mappings['Canal'])
)
df_vtex['Origen_Estandar'] = df_vtex['UtmMedium'].apply(
    lambda x: clasificar_valor(x, DICT_ORIGENES, st.session_state.custom_mappings['Origen'])
)

nuevos_canales = df_vtex[df_vtex['Canal_Estandar'] == 'Otros / No Asignados']['UtmSource'].dropna().unique().tolist()
nuevos_origenes = df_vtex[df_vtex['Origen_Estandar'] == 'Otros / No Asignados']['UtmMedium'].dropna().unique().tolist()

if nuevos_canales or nuevos_origenes:
    with st.expander("⚠️ Atributos no asignados detectados", expanded=False):
        if nuevos_canales:
            st.markdown("**Nuevos Canales:**")
            for nc in nuevos_canales[:5]:
                cat_sel = st.selectbox(f"Asignar '{nc}' a:", list(DICT_CANALES.keys()), key=f"nc_{nc}")
                if st.button(f"Guardar regla para {nc}"):
                    st.session_state.custom_mappings['Canal'][str(nc).lower().strip()] = cat_sel
                    st.rerun()

# ==========================================================
# 6A. BARRA SUPERIOR DE FILTRO POR FECHAS (SINCRONIZADO)
# ==========================================================
anios_disp = sorted(df_vtex['Año'].unique(), reverse=True)
col_año, col_segmento, col_mes, col_dias = st.columns([1.2, 3.5, 1.8, 1.8])

with col_año:
    anio_sel = st.selectbox("Año", anios_disp, index=0, label_visibility="collapsed")

# Filtrar ambos DataFrames
df_f = df_vtex[df_vtex['Año'] == anio_sel].copy()
df_sku_f = df_sku[df_sku['Año'] == anio_sel].copy() if not df_sku.empty else pd.DataFrame()

with col_segmento:
    opciones_periodo = ["Año", "Q1", "Q2", "Q3", "Q4", "Mes"]
    try:
        periodo_sel = st.segmented_control("Periodo", options=opciones_periodo, default="Año", label_visibility="collapsed")
    except AttributeError:
        periodo_sel = st.radio("Periodo", options=opciones_periodo, horizontal=True, label_visibility="collapsed")

MAP_MESES = {
    'Ene': 'Enero', 'Feb': 'Febrero', 'Mar': 'Marzo', 'Abr': 'Abril',
    'May': 'Mayo', 'Jun': 'Junio', 'Jul': 'Julio', 'Ago': 'Agosto',
    'Sep': 'Septiembre', 'Oct': 'Octubre', 'Nov': 'Noviembre', 'Dic': 'Diciembre'
}
MAP_MESES_REV = {v: k for k, v in MAP_MESES.items()}

mes_sel_nombre = None
opcion_dias_sel = "Todo el mes"

if periodo_sel == "Mes":
    meses_cortos_presentes = df_f['Mes_Nombre'].unique()
    meses_largos_presentes = [MAP_MESES[m] for m in meses_cortos_presentes if m in MAP_MESES]
    if not meses_largos_presentes:
        meses_largos_presentes = list(MAP_MESES.values())

    with col_mes:
        mes_sel_nombre = st.selectbox("Mes", meses_largos_presentes, index=len(meses_largos_presentes)-1, label_visibility="collapsed")
    with col_dias:
        opcion_dias_sel = st.selectbox("Días", ["Todo el mes", "Seleccionar Rango"], label_visibility="collapsed")

# Lógica de Filtrado por Tiempo Sincronizada
if periodo_sel == "Año":
    modo_tiempo = "Año Completo (YoY)"

elif periodo_sel in ["Q1", "Q2", "Q3", "Q4"]:
    modo_tiempo = "Quarter vs Q Año Anterior"
    q_num = int(periodo_sel.replace("Q", ""))
    df_f = df_f[df_f['Quarter_Num'] == q_num]
    if not df_sku_f.empty and 'Quarter_Num' in df_sku_f.columns:
        df_sku_f = df_sku_f[df_sku_f['Quarter_Num'] == q_num]

elif periodo_sel == "Mes":
    modo_tiempo = "Mes vs Mes Anterior (MoM)"
    if mes_sel_nombre:
        mes_corto_sel = MAP_MESES_REV.get(mes_sel_nombre, mes_sel_nombre)
        df_f = df_f[df_f['Mes_Nombre'] == mes_corto_sel]
        if not df_sku_f.empty and 'Mes_Nombre' in df_sku_f.columns:
            df_sku_f = df_sku_f[df_sku_f['Mes_Nombre'] == mes_corto_sel]
        mes_num_sel = df_f['Mes_Num'].iloc[0] if not df_f.empty else 1

    if opcion_dias_sel == "Seleccionar Rango":
        min_date = df_f['Día'].min()
        max_date = df_f['Día'].max()
        if pd.notnull(min_date) and pd.notnull(max_date):
            rango = st.date_input("Rango", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(rango, tuple) and len(rango) == 2:
                df_f = df_f[(df_f['Día'] >= rango[0]) & (df_f['Día'] <= rango[1])]
                if not df_sku_f.empty:
                    f_min_str = rango[0].strftime('%Y-%m-%d')
                    f_max_str = rango[1].strftime('%Y-%m-%d')
                    df_sku_f = df_sku_f[(df_sku_f['Fecha_Clean'] >= f_min_str) & (df_sku_f['Fecha_Clean'] <= f_max_str)]

# ==========================================================
# 6B. FILTROS SECUNDARIOS (SIDEBAR)
# ==========================================================
st.sidebar.header("Filtros de Segmentación")

tipo_cliente_sel = st.sidebar.multiselect("Tipo de Cliente:", ["Nuevo", "Recurrente"], default=[], placeholder="Todos los clientes")
if tipo_cliente_sel:
    df_f = df_f[df_f['Tipo_Cliente'].isin(tipo_cliente_sel)]
    if not df_sku_f.empty and 'Tipo_Cliente' in df_sku_f.columns:
        df_sku_f = df_sku_f[df_sku_f['Tipo_Cliente'].isin(tipo_cliente_sel)]

canales_disp = sorted(df_f['Canal_Estandar'].unique().tolist())
canales_sel = st.sidebar.multiselect("Canal (UtmSource):", canales_disp, default=[], placeholder="Todos los canales")
if canales_sel:
    df_f = df_f[df_f['Canal_Estandar'].isin(canales_sel)]
    if not df_sku_f.empty and 'Canal_Estandar' in df_sku_f.columns:
        df_sku_f = df_sku_f[df_sku_f['Canal_Estandar'].isin(canales_sel)]

origenes_disp = sorted(df_f['Origen_Estandar'].unique().tolist())
origenes_sel = st.sidebar.multiselect("Origen (UtmMedium):", origenes_disp, default=[], placeholder="Todos los orígenes")
if origenes_sel:
    df_f = df_f[df_f['Origen_Estandar'].isin(origenes_sel)]
    if not df_sku_f.empty and 'Origen_Estandar' in df_sku_f.columns:
        df_sku_f = df_sku_f[df_sku_f['Origen_Estandar'].isin(origenes_sel)]

df_f['Discounts_Clean'] = df_f['Discounts Names'].apply(limpiar_cadena_descuentos)
descuentos_sel = []

medios_pago_disp = sorted(df_f['Payment System Name'].fillna('No Especificado').unique().tolist())
medios_sel = st.sidebar.multiselect("Medio de Pago:", medios_pago_disp, default=[], placeholder="Todos los medios de pago")
if medios_sel:
    df_f = df_f[df_f['Payment System Name'].fillna('No Especificado').isin(medios_sel)]
    if not df_sku_f.empty and 'Payment System Name' in df_sku_f.columns:
        df_sku_f = df_sku_f[df_sku_f['Payment System Name'].fillna('No Especificado').isin(medios_sel)]

opciones_marca = ["Todas"] + MARCAS_LISTA + ["Otros"]
marca_sel = st.sidebar.selectbox("Filtrar por Marca:", opciones_marca)

if marca_sel == "Otros":
    todas_variantes = [v for lista in DICT_MARCAS_VARIANTES.values() for v in lista]
    patron_todas = '|'.join(todas_variantes)
    df_f = df_f[~df_f['SKU Name'].str.lower().str.contains(patron_todas, na=False)]
    if not df_sku_f.empty:
        df_sku_f = df_sku_f[~df_sku_f['SKU Name'].str.lower().str.contains(patron_todas, na=False)]
elif marca_sel != "Todas":
    variantes_marca = DICT_MARCAS_VARIANTES.get(marca_sel, [marca_sel.lower()])
    patron_marca = '|'.join(variantes_marca)
    df_f = df_f[df_f['SKU Name'].str.lower().str.contains(patron_marca, na=False)]
    if not df_sku_f.empty:
        df_sku_f = df_sku_f[df_sku_f['SKU Name'].str.lower().str.contains(patron_marca, na=False)]

# ==========================================================
# 7. COMPARATIVAS Y CÁLCULO DIRECTO DE INVERSIÓN
# ==========================================================
if modo_tiempo == "Año Completo (YoY)":
    df_comp = df_vtex[df_vtex['Año'] == (anio_sel - 1)]
    etiqueta_comp = f"vs Año {anio_sel - 1}"
elif modo_tiempo == "Mes vs Mes Anterior (MoM)":
    if mes_num_sel == 1:
        df_comp = df_vtex[(df_vtex['Año'] == anio_sel - 1) & (df_vtex['Mes_Num'] == 12)]
    else:
        df_comp = df_vtex[(df_vtex['Año'] == anio_sel) & (df_vtex['Mes_Num'] == mes_num_sel - 1)]
    etiqueta_comp = "vs Mes Anterior"
else:
    q_num = int(periodo_sel.replace('Q', ''))
    df_comp = df_vtex[(df_vtex['Año'] == anio_sel - 1) & (df_vtex['Quarter_Num'] == q_num)]
    etiqueta_comp = f"vs {periodo_sel} {anio_sel - 1}"

if not df_comp.empty:
    if tipo_cliente_sel: df_comp = df_comp[df_comp['Tipo_Cliente'].isin(tipo_cliente_sel)]
    if canales_sel: df_comp = df_comp[df_comp['Canal_Estandar'].isin(canales_sel)]
    if origenes_sel: df_comp = df_comp[df_comp['Origen_Estandar'].isin(origenes_sel)]
    if marca_sel != "Todas": df_comp = df_comp[df_comp['SKU Name'].str.lower().str.contains(marca_sel.lower(), na=False)]

ventas_actual = df_f['Total Value'].sum()
ventas_comp = df_comp['Total Value'].sum() if not df_comp.empty else 0
var_ventas = ((ventas_actual - ventas_comp) / ventas_comp * 100) if ventas_comp > 0 else 0

ordenes_actual = df_f['Order'].nunique()
ordenes_comp = df_comp['Order'].nunique() if not df_comp.empty else 0
var_ordenes = ((ordenes_actual - ordenes_comp) / ordenes_comp * 100) if ordenes_comp > 0 else 0

unidades_actual = df_f['Quantity_SKU'].sum()

dias_filtrados_str = df_f['Fecha_Clean'].dropna().unique().tolist()

ventas_meta = df_f[df_f['Canal_Estandar'].isin(['Facebook', 'Instagram'])]['Total Value'].sum()
ventas_google = df_f[df_f['Canal_Estandar'] == 'Google']['Total Value'].sum()

inv_meta_tot = df_meta[df_meta['Fecha_Clean'].isin(dias_filtrados_str)]['Inversion'].sum() if not df_meta.empty else 0.0
inv_google_tot = df_google[df_google['Fecha_Clean'].isin(dias_filtrados_str)]['Inversion'].sum() if not df_google.empty else 0.0

inversion_total = inv_meta_tot + inv_google_tot
roas = (ventas_actual / inversion_total) if inversion_total > 0 else 0.0
roas_meta = (ventas_meta / inv_meta_tot) if inv_meta_tot > 0 else 0.0
roas_google = (ventas_google / inv_google_tot) if inv_google_tot > 0 else 0.0
aov_actual = (ventas_actual / ordenes_actual) if ordenes_actual > 0 else 0.0

# ==========================================================
# 8. MÉTRICAS CLAVE (KPIs - 5 TARJETAS MODERNAS)
# ==========================================================
df_nuevos = df_f[df_f['Tipo_Cliente'] == 'Nuevo']
df_recurrentes = df_f[df_f['Tipo_Cliente'] == 'Recurrente']

nuevos_clientes_cant = df_nuevos['Client Document'].nunique() if 'Client Document' in df_nuevos.columns else 0
recurrentes_cant = df_recurrentes['Client Document'].nunique() if 'Client Document' in df_recurrentes.columns else 0

cac = (inversion_total / nuevos_clientes_cant) if nuevos_clientes_cant > 0 else 0.0
ingresos_recurrentes = df_recurrentes['Total Value'].sum()
ltv_recurrente = (ingresos_recurrentes / recurrentes_cant) if recurrentes_cant > 0 else 0.0
ratio_ltv_cac = (ltv_recurrente / cac) if cac > 0 else 0.0

badge_ventas_class = "badge-pill-green" if var_ventas >= 0 else "badge-pill-red"
badge_ventas_text = f"{var_ventas:+.1f}% vs {anio_sel-1}" if modo_tiempo == "Año Completo (YoY)" else f"{var_ventas:+.1f}% {etiqueta_comp}"

badge_ordenes_class = "badge-pill-green" if var_ordenes >= 0 else "badge-pill-red"
badge_ordenes_text = f"{var_ordenes:+.1f}% vs {anio_sel-1}" if modo_tiempo == "Año Completo (YoY)" else f"{var_ordenes:+.1f}% {etiqueta_comp}"

if roas >= 4.0:
    badge_roas_html = '<span class="badge-pill-green">🎯 Óptimo</span>'
elif roas >= 2.5:
    badge_roas_html = '<span class="badge-pill-gray">⚡ Regular</span>'
else:
    badge_roas_html = '<span class="badge-pill-red">⚠️ Bajo</span>'

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-title">🛒 INGRESOS TOTALES</div>
            <span class="{badge_ventas_class}">{badge_ventas_text}</span>
        </div>
        <div class="kpi-value">${ventas_actual:,.0f}</div>
        <div class="kpi-sub">{unidades_actual:,.0f} unidades vendidas</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-title">📦 ÓRDENES TOTALES</div>
            <span class="{badge_ordenes_class}">{badge_ordenes_text}</span>
        </div>
        <div class="kpi-value">{ordenes_actual:,}</div>
        <div class="kpi-sub">AOV (Ticket): ${aov_actual:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-title">💰 INVERSIÓN PAUTA</div>
        </div>
        <div class="kpi-value">${inversion_total:,.0f}</div>
        <div class="kpi-sub">
            <span>💙 Meta: ${inv_meta_tot:,.0f}</span>
            <span style="margin-left:8px;">🟢 Google: ${inv_google_tot:,.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-title">🎯 ROAS GENERAL</div>
            {badge_roas_html}
        </div>
        <div class="kpi-value">{roas:.2f} x</div>
        <div class="kpi-sub">
            <span>💙 Meta: {roas_meta:.2f}x</span>
            <span style="margin-left:8px;">🟢 Google: {roas_google:.2f}x</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    ratio_str = f"{ratio_ltv_cac:.1f}x" if ratio_ltv_cac > 0 else "N/A"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-title">👥 CAC & LTV</div>
        </div>
        <div class="kpi-value">CAC: ${cac:,.0f}</div>
        <div class="kpi-sub" style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#16a34a; font-weight:600;">LTV: ${ltv_recurrente:,.0f}</span>
            <span style="color:#64748b; font-size:11px;">Ratio: {ratio_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

total_ord_sub = df_f['Order'].nunique()
ord_nuevos = df_f[df_f['Tipo_Cliente'] == 'Nuevo']['Order'].nunique()
ord_recurrentes = df_f[df_f['Tipo_Cliente'] == 'Recurrente']['Order'].nunique()

pct_nuevos = (ord_nuevos / total_ord_sub * 100) if total_ord_sub > 0 else 0
pct_recurrentes = (ord_recurrentes / total_ord_sub * 100) if total_ord_sub > 0 else 0

# 2 Banners de Clientes
col_cl1, col_cl2 = st.columns(2)
with col_cl1:
    st.markdown(f"""
    <div class="client-banner-blue">
        <div style="color: #2563eb; font-weight: 700; font-size: 13px; margin-bottom: 4px;">👤 Clientes Nuevos</div>
        <div style="color: #1d4ed8; font-weight: 800; font-size: 22px;">
            {ord_nuevos:,} órdenes <span style="font-size: 14px; font-weight: 600; color: #3b82f6;">({pct_nuevos:.1f}%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_cl2:
    st.markdown(f"""
    <div class="client-banner-green">
        <div style="color: #16a34a; font-weight: 700; font-size: 13px; margin-bottom: 4px;">🔄 Clientes Recurrentes</div>
        <div style="color: #15803d; font-weight: 800; font-size: 22px;">
            {ord_recurrentes:,} órdenes <span style="font-size: 14px; font-weight: 600; color: #22c55e;">({pct_recurrentes:.1f}%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# 2 Columnas de Gráficos (Ventas por Canal + Origen de Tráfico)
col_graf1, col_graf2 = st.columns([1.1, 0.9])

with col_graf1:
    st.markdown('<div class="section-title" style="font-size:16px; margin: 4px 0 10px 0;">🌐 Ventas por Canal de Adquisición (UtmSource)</div>', unsafe_allow_html=True)
    
    df_canal_agg = df_f.groupby('Canal_Estandar').agg(
        Ingresos=('Total Value', 'sum'),
        Ordenes=('Order', 'nunique')
    ).reset_index()
    
    # 1. Tarjeta dedicada para Ventas Directas / Sin Atribución
    row_directo = df_canal_agg[df_canal_agg['Canal_Estandar'] == 'Directo / Sin Datos']
    ingresos_directo = row_directo['Ingresos'].sum() if not row_directo.empty else 0.0
    ordenes_directo = int(row_directo['Ordenes'].sum()) if not row_directo.empty else 0
    pct_directo = (ingresos_directo / df_f['Total Value'].sum() * 100) if df_f['Total Value'].sum() > 0 else 0.0
    
    st.markdown(f"""
    <div class="direct-card" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <div>
            <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">📍 Ventas Directas / Sin Atribución</div>
            <div class="direct-card-val" style="font-size: 17px; font-weight: 800; color: #0f172a; margin-top: 2px;">
                ${ingresos_directo:,.0f} <span style="font-size: 12px; font-weight: 600; color: #64748b;">({pct_directo:.1f}% del total)</span>
            </div>
        </div>
        <div class="direct-card-badge" style="text-align: right; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 4px 10px;">
            <span style="font-size: 12px; font-weight: 700; color: #334155;">{ordenes_directo:,}</span> <span style="font-size: 11px; color: #64748b;">órdenes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Gráfico de barras horizontales excluyendo 'Directo / Sin Datos'
    df_canal_bars = df_canal_agg[df_canal_agg['Canal_Estandar'] != 'Directo / Sin Datos'].sort_values('Ingresos', ascending=True)
    
    colores_canales_map = {
        'Facebook': '#1877f2',
        'VTEX': '#e11d48',
        'Connectif': '#38bdf8',
        'Google': '#22c55e',
        'Instagram': '#e1306c',
        'General': '#3b82f6',
        'Nequi': '#94a3b8',
        'AI': '#6366f1',
        'HubSpot': '#f97316',
        'TikTok': '#0f172a',
        'Icommarketing': '#a855f7',
        'YouTube': '#ef4444'
    }
    
    bar_colors = [colores_canales_map.get(c, '#3b82f6') for c in df_canal_bars['Canal_Estandar']]
    bar_texts = [formatear_cifra_compacta(v) for v in df_canal_bars['Ingresos']]
    
    fig_canal = go.Figure(go.Bar(
        y=df_canal_bars['Canal_Estandar'],
        x=df_canal_bars['Ingresos'],
        orientation='h',
        marker_color=bar_colors,
        text=bar_texts,
        textposition='auto',
        textfont=dict(color='white', size=11, family='Arial, sans-serif', weight='bold'),
        hovertemplate="<b>Canal:</b> %{y}<br><b>Ingresos:</b> $%{x:,.0f}<br><b>Órdenes:</b> %{customdata:,d}<extra></extra>",
        customdata=df_canal_bars['Ordenes']
    ))
    
    fig_canal.update_layout(
        margin=dict(l=10, r=20, t=10, b=30),
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Ingresos (COP)",
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.15)',
            tickformat='~s'
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=11)
        )
    )
    st.plotly_chart(fig_canal, use_container_width=True)

with col_graf2:
    st.markdown('<div class="section-title" style="font-size:16px; margin: 4px 0 10px 0;">🎯 Origen de Tráfico (UtmMedium)</div>', unsafe_allow_html=True)
    
    colores_origenes_map = {
        'Directo': '#22c55e',
        'Publicidad (Pauta)': '#2563eb',
        'Email / Push': '#f59e0b',
        'Orgánico / Botones Web': '#8b5cf6',
        'Enlaces En Redes': '#06b6d4',
        'Alianzas': '#ec4899',
        'Otros / No Asignados': '#94a3b8'
    }
    
    df_origen_agg = df_f.groupby('Origen_Estandar').agg(
        Ingresos=('Total Value', 'sum'),
        Ordenes=('Order', 'nunique')
    ).reset_index().sort_values('Ordenes', ascending=False)
    
    origen_colors = [colores_origenes_map.get(o, '#94a3b8') for o in df_origen_agg['Origen_Estandar']]
    
    fig_origen = go.Figure(data=[go.Pie(
        labels=df_origen_agg['Origen_Estandar'],
        values=df_origen_agg['Ordenes'],
        hole=0.55,
        marker=dict(colors=origen_colors),
        textinfo='percent+label',
        textposition='auto',
        insidetextorientation='horizontal',
        textfont=dict(size=11),
        hovertemplate="<b>Origen:</b> %{label}<br><b>Órdenes:</b> %{value:,d} (%{percent})<br><b>Ingresos:</b> $%{customdata:,.0f}<extra></extra>",
        customdata=df_origen_agg['Ingresos']
    )])
    
    fig_origen.update_layout(
        showlegend=False,
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_origen, use_container_width=True)

# ==========================================================
# 10. ANALÍTICA PROFUNDA & PESTAÑAS
# ==========================================================
st.markdown("### Analítica Profunda & Rendimiento")

tab_tendencias, tab_pauta, tab_carrito, tab_retencion, tab_funnel = st.tabs([
    " Tendencia Temporal & Adquisición", 
    " Eficiencia de Pauta (ROAS)", 
    " Comportamiento de Compra (AOV & Pagos)",
    " Retención & Pareto (80/20)",
    " Funnel & Diagnóstico Anuncios"
])

# ----------------------------------------------------------
# PESTAÑA 1: TENDENCIA TEMPORAL Y ADQUISICIÓN
# ----------------------------------------------------------
with tab_tendencias:
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("Evolución Diaria: Ventas vs Inversión Ads")
        
        df_ventas_dia = df_f.groupby('Fecha_Clean').agg(
            Ingresos=('Total Value', 'sum'),
            Ordenes=('Order', 'nunique')
        ).reset_index()

        if not df_ventas_dia.empty:
            fecha_min_str = df_ventas_dia['Fecha_Clean'].min()
            fecha_max_str = df_ventas_dia['Fecha_Clean'].max()
            
            rango_fechas = pd.date_range(start=fecha_min_str, end=fecha_max_str, freq='D').strftime('%Y-%m-%d')
            df_timeline = pd.DataFrame({'Fecha_Clean': rango_fechas})
            
            df_meta_f = df_meta[(df_meta['Fecha_Clean'] >= fecha_min_str) & (df_meta['Fecha_Clean'] <= fecha_max_str)] if not df_meta.empty else pd.DataFrame()
            df_goog_f = df_google[(df_google['Fecha_Clean'] >= fecha_min_str) & (df_google['Fecha_Clean'] <= fecha_max_str)] if not df_google.empty else pd.DataFrame()
            
            df_inv_combined = pd.concat([df_meta_f, df_goog_f], ignore_index=True) if (not df_meta_f.empty or not df_goog_f.empty) else pd.DataFrame(columns=['Fecha_Clean', 'Inversion'])
            df_inv_dia = df_inv_combined.groupby('Fecha_Clean')['Inversion'].sum().reset_index() if not df_inv_combined.empty else pd.DataFrame(columns=['Fecha_Clean', 'Inversion'])

            df_tendencia = pd.merge(df_timeline, df_ventas_dia, on='Fecha_Clean', how='left')
            df_tendencia = pd.merge(df_tendencia, df_inv_dia, on='Fecha_Clean', how='left').fillna(0)
            df_tendencia = df_tendencia.sort_values('Fecha_Clean')

            fig_tendencia = go.Figure()

            fig_tendencia.add_trace(
                go.Bar(
                    x=df_tendencia['Fecha_Clean'], 
                    y=df_tendencia['Inversion'], 
                    name="Inversión Ads ($)", 
                    marker_color='rgba(245, 158, 11, 0.45)',
                    hovertemplate="<b>Fecha:</b> %{x}<br><b>Inversión:</b> $%{y:,.0f}<extra></extra>"
                )
            )

            fig_tendencia.add_trace(
                go.Scatter(
                    x=df_tendencia['Fecha_Clean'], 
                    y=df_tendencia['Ingresos'], 
                    name="Ventas ($ COP)", 
                    line=dict(color='#2563eb', width=2.5),
                    hovertemplate="<b>Ventas:</b> $%{y:,.0f}<extra></extra>"
                )
            )

            fig_tendencia.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=30, b=10),
                height=380,
                xaxis_title="",
                yaxis_title="Monto ($ COP)"
            )

            st.plotly_chart(fig_tendencia, use_container_width=True)
        else:
            st.info("No hay datos de ventas en el período seleccionado.")

    with col_t2:
        st.subheader("Adquisición Temporal: Nuevos vs Recurrentes")
        
        df_acq_dia = df_f.groupby(['Fecha_Clean', 'Tipo_Cliente']).agg(Ordenes=('Order', 'nunique')).reset_index()
        
        fig_apiladas = px.bar(
            df_acq_dia,
            x='Fecha_Clean',
            y='Ordenes',
            color='Tipo_Cliente',
            color_discrete_map={'Nuevo': '#2563eb', 'Recurrente': '#10b981'},
            labels={'Ordenes': 'Órdenes', 'Fecha_Clean': ''}
        )
        fig_apiladas.update_layout(
            barmode='stack', 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        fig_apiladas.update_traces(hovertemplate="<b>Fecha:</b> %{x}<br><b>Órdenes:</b> %{y:,d}<extra></extra>")
        st.plotly_chart(fig_apiladas, use_container_width=True)

# ----------------------------------------------------------
# PESTAÑA 2: RENDIMIENTO FINANCIERO Y EFICIENCIA DE PAUTA
# ----------------------------------------------------------
with tab_pauta:
    st.subheader("Eficiencia de Pauta: Inversión vs ROAS por Canal")
    
    ventas_meta = df_f[df_f['Canal_Estandar'].isin(['Facebook', 'Instagram'])]['Total Value'].sum()
    ventas_google = df_f[df_f['Canal_Estandar'] == 'Google']['Total Value'].sum()
    
    data_pauta = [
        {'Canal': 'Meta (FB / IG)', 'Inversion': inv_meta_tot, 'Ventas': ventas_meta, 'ROAS': (ventas_meta / inv_meta_tot) if inv_meta_tot > 0 else 0},
        {'Canal': 'Google Ads', 'Inversion': inv_google_tot, 'Ventas': ventas_google, 'ROAS': (ventas_google / inv_google_tot) if inv_google_tot > 0 else 0}
    ]
    df_roas_canal = pd.DataFrame(data_pauta)
    
    fig_roas = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_roas.add_trace(
        go.Bar(
            x=df_roas_canal['Canal'], 
            y=df_roas_canal['Inversion'], 
            name="Inversión ($)", 
            marker_color='#2563eb', 
            text=df_roas_canal['Inversion'].apply(lambda x: f"${x:,.0f}"), 
            textposition='inside',
            textfont=dict(color='white', size=13, family='Arial Black')
        ),
        secondary_y=False
    )
    
    fig_roas.add_trace(
        go.Scatter(
            x=df_roas_canal['Canal'], 
            y=df_roas_canal['ROAS'], 
            name="ROAS (x)", 
            mode='lines+markers+text', 
            text=df_roas_canal['ROAS'].apply(lambda x: f"{x:.2f}x"), 
            textposition='top center', 
            line=dict(color='#ef4444', width=3), 
            marker=dict(size=12, color='#ef4444')
        ),
        secondary_y=True
    )
    
    fig_roas.update_layout(
        hovermode="x unified", 
        showlegend=True, 
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_roas.update_yaxes(title_text="Inversión ($ COP)", secondary_y=False)
    fig_roas.update_yaxes(title_text="ROAS (Múltiplo)", secondary_y=True, range=[0, max(df_roas_canal['ROAS'].max()*1.3, 2)])
    
    st.plotly_chart(fig_roas, use_container_width=True)

# ----------------------------------------------------------
# PESTAÑA 3: COMPORTAMIENTO DE COMPRA (AOV & PAGOS LIMPIOS)
# ----------------------------------------------------------
with tab_carrito:
    col_aov, col_dona = st.columns(2)
    
    with col_aov:
        st.subheader("Ticket Medio (AOV) por Medio de Pago")
        
        df_pago_clean = df_f.copy()
        df_pago_clean['Payment System Name'] = df_pago_clean['Payment System Name'].fillna('No Especificado').astype(str)
        
        df_aov = df_pago_clean.groupby('Payment System Name').agg(
            Ingresos=('Total Value', 'sum'),
            Ordenes=('Order', 'nunique')
        ).reset_index()
        
        df_aov['AOV'] = (df_aov['Ingresos'] / df_aov['Ordenes']).fillna(0)
        df_aov = df_aov.sort_values('AOV', ascending=True).tail(8)
        
        fig_aov = px.bar(
            df_aov,
            y='Payment System Name',
            x='AOV',
            orientation='h',
            text_auto='.2s',
            labels={'Payment System Name': '', 'AOV': 'Valor Promedio Orden ($)'},
            color_discrete_sequence=['#10b981']
        )
        fig_aov.update_traces(
            textposition='inside',
            hovertemplate="<b>Medio:</b> %{y}<br><b>AOV:</b> $%{x:,.0f}<extra></extra>"
        )
        fig_aov.update_layout(height=380)
        st.plotly_chart(fig_aov, use_container_width=True)

    with col_dona:
        st.subheader("Participación por Medio de Pago")
        
        df_pago_donuts = df_pago_clean.groupby('Payment System Name').agg(
            Ordenes=('Order', 'nunique')
        ).reset_index().sort_values('Ordenes', ascending=False)
        
        total_ordenes_pagos = df_pago_donuts['Ordenes'].sum()
        
        if total_ordenes_pagos > 0:
            df_pago_donuts['Pct'] = (df_pago_donuts['Ordenes'] / total_ordenes_pagos) * 100
            
            df_principales = df_pago_donuts[df_pago_donuts['Pct'] >= 2.0].copy()
            df_menores = df_pago_donuts[df_pago_donuts['Pct'] < 2.0]
            
            if not df_menores.empty:
                otros_row = pd.DataFrame([{
                    'Payment System Name': 'Otros',
                    'Ordenes': df_menores['Ordenes'].sum(),
                    'Pct': df_menores['Pct'].sum()
                }])
                df_pago_donuts = pd.concat([df_principales, otros_row], ignore_index=True)
            else:
                df_pago_donuts = df_principales

        fig_dona = px.pie(
            df_pago_donuts,
            names='Payment System Name',
            values='Ordenes',
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_dona.update_traces(
            textinfo='percent+label',
            insidetextorientation='radial',
            hovertemplate="<b>Medio:</b> %{label}<br><b>Órdenes:</b> %{value:,d}<extra></extra>"
        )
        fig_dona.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_dona, use_container_width=True)

# ----------------------------------------------------------
# PESTAÑA 4: RETENCIÓN, PARETO & FRECUENCIA (USA DATASET DETALLE SKU)
# ----------------------------------------------------------
with tab_retencion:
    col_p1, col_p2 = st.columns(2)
    
    # PARETO USANDO DATASET DE DETALLE SKU
    with col_p1:
        st.subheader("⚖️ Concentración de Ventas (Pareto)")
        
        # Evalúa primero df_sku_f y pasa df_f sin lanzar KeyError
        dataset_pareto = df_sku_f if (not df_sku_f.empty and 'SKU Name' in df_sku_f.columns) else df_f
        res_pareto = generar_grafico_pareto(dataset_pareto)
        
        if res_pareto:
            fig_p, pct_alc = res_pareto
            st.caption(f"El Top 15 de productos concentra el **{pct_alc:.1f}%** del total de ingresos del período.")
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Sin datos suficientes para calcular Pareto.")

    with col_p2:
        st.subheader("🔄 Cohortes de Retención (% Recompra)")
        st.caption("Porcentaje de clientes que vuelven a comprar a partir del **Mes 1** (excluyendo la 1ª compra).")
        
        matrix_ret = generar_matriz_cohortes(df_vtex)
        
        if not matrix_ret.empty and matrix_ret.shape[1] > 1:
            matrix_show = matrix_ret.tail(8).iloc[:, 1:7]
            
            y_labels = [str(idx) for idx in matrix_show.index]
            x_labels = [f"Mes {i}" for i in matrix_show.columns]

            fig_cohort = px.imshow(
                matrix_show.values,
                labels=dict(x="Meses Después de 1ª Compra", y="Cohorte (1ª Compra)", color="% Recompra"),
                x=x_labels,
                y=y_labels,
                text_auto=".1f",
                color_continuous_scale="Blues",
                aspect="auto"
            )
            
            fig_cohort.update_layout(height=420, coloraxis_showscale=False)
            fig_cohort.update_traces(hovertemplate="<b>Cohorte:</b> %{y}<br><b>%{x}:</b> %{z:.1f}% de recompra<extra></extra>")
            st.plotly_chart(fig_cohort, use_container_width=True)
        else:
            st.info("No hay datos de recompras suficientes para generar la matriz de cohortes.")

    st.divider()

    st.subheader("🕒 Frecuencia de Compra (Tiempo Promedio Entre Órdenes)")
    st.caption("Días promedio que transcurren para que un cliente realice su siguiente transacción.")
    
    fig_frecuencia = generar_grafico_tiempo_entre_compras(df_vtex)
    if fig_frecuencia:
        st.plotly_chart(fig_frecuencia, use_container_width=True)
    else:
        st.info("No se registraron recompras en la base para medir el tiempo entre transacciones.")

# ----------------------------------------------------------
# PESTAÑA 5: FUNNEL Y DIAGNÓSTICO DE ANUNCIOS
# ----------------------------------------------------------
with tab_funnel:
    st.subheader("📊 Diagnóstico de Salud de Anuncios & Ratios Calculados")
    
    # Filtrar Ads al período activo por fecha
    df_meta_act = df_meta[(df_meta['Fecha_Clean'] >= fecha_min_str) & (df_meta['Fecha_Clean'] <= fecha_max_str)] if not df_meta.empty else pd.DataFrame()
    df_google_act = df_google[(df_google['Fecha_Clean'] >= fecha_min_str) & (df_google['Fecha_Clean'] <= fecha_max_str)] if not df_google.empty else pd.DataFrame()

    c_imp_m = df_meta_act['Impresiones'].sum() if not df_meta_act.empty else 0
    c_clics_m = df_meta_act['Clics'].sum() if not df_meta_act.empty else 0
    c_lp_m = df_meta_act['Visitas_LP'].sum() if not df_meta_act.empty else 0
    c_comp_m = df_meta_act['Compras'].sum() if not df_meta_act.empty else 0

    ctr_m = (c_clics_m / c_imp_m * 100) if c_imp_m > 0 else 0
    tasa_web_m = (c_lp_m / c_clics_m * 100) if c_clics_m > 0 else 0
    conv_final_m = (c_comp_m / c_lp_m * 100) if c_lp_m > 0 else 0

    # Tarjetas Métricas Rápidas de Diagnóstico
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CTR Promedio (Meta)", f"{ctr_m:.2f}%", help="Objetivo e-commerce: > 1.5%")
    m2.metric("Retención Carga Web", f"{tasa_web_m:.1f}%", help="Tráfico retenido tras hacer clic. Ideal > 80%")
    m3.metric("Tasa Conversión Final", f"{conv_final_m:.2f}%", help="Compras / Visitas Landing Page")
    m4.metric(
    "Diagnóstico Web", 
    "Saludable 🟢" if tasa_web_m >= 65 else "Fuga de Tráfico 🔴",
    help="Alerta cuando más del 35% de los clics abandonan antes de cargar la web"
)

    st.divider()

    # 2. EMBUDOS DE CONVERSIÓN
    col_fn1, col_fn2 = st.columns(2)
    
    with col_fn1:
        st.subheader("🔻 Embudo Meta Ads (Fuga de Tráfico)")
        st.caption("Evolución: Impresiones ➔ Clics ➔ Visitas LP ➔ Carritos ➔ Compras")
        fig_fn_m = generar_funnel_meta(df_meta_act)
        if fig_fn_m:
            st.plotly_chart(fig_fn_m, use_container_width=True)
        else:
            st.info("Sin datos de embudo para Meta Ads en el período seleccionado.")

    with col_fn2:
        st.subheader("🔻 Embudo Google Ads (Intención de Búsqueda)")
        st.caption("Evolución: Impresiones ➔ Clics ➔ Compras (Resultados)")
        fig_fn_g = generar_funnel_google(df_google_act)
        if fig_fn_g:
            st.plotly_chart(fig_fn_g, use_container_width=True)
        else:
            st.info("Sin datos de embudo para Google Ads en el período seleccionado.")

    st.divider()

    # 3. SCATTER PLOT Y FATIGA DE CREATIVO
    col_sc1, col_sc2 = st.columns(2)

    with col_sc1:
        st.subheader("🎯 Matriz de Anuncios: CTR vs Compras")
        st.caption("Identifica elementos **Ganadores** (alto CTR + altas compras) vs **Clickbaits**.")
        fig_sc = generar_scatter_ctr_compras(df_meta_act)
        if fig_sc:
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("No hay datos de impresiones/compras suficientes para generar la matriz.")

    with col_sc2:
        st.subheader("📉 Punto de Cansancio (Frecuencia vs CTR)")
        st.caption("Detecta cuando la audiencia se fatiga por ver la pauta repetidas veces.")
        fig_frec = generar_linea_frecuencia_conversion(df_meta_act)
        if fig_frec:
            st.plotly_chart(fig_frec, use_container_width=True)
        else:
            st.info("No hay columna de 'Frecuencia' válida para procesar el gráfico.")

# ==========================================================
# 11. TOP PRODUCTOS & TABLA (FORZADO A DATASET SKUs)
# ==========================================================
st.subheader(f"Top Productos — {marca_sel}")

# Pasa el dataset desglosado con fallback seguro
dataset_top = df_sku_f if (not df_sku_f.empty and 'SKU Name' in df_sku_f.columns) else df_f
df_top_prod = obtener_top_productos(dataset_top, marca_filtro=marca_sel, top_n=10)

if not df_top_prod.empty:
    max_uds = df_top_prod['Unidades'].max() if df_top_prod['Unidades'].max() > 0 else 1
    with st.container():
        for idx, row in df_top_prod.reset_index(drop=True).iterrows():
            posicion = idx + 1
            nombre_prod = row['SKU_Individual']
            uds = int(row['Unidades'])
            ordenes = int(row['Ordenes'])
            porcentaje_barra = int((uds / max_uds) * 100)
            color_num = "#f59e0b" if posicion <= 3 else "#9ca3af"

            st.markdown(f"""
            <div style="margin-bottom: 12px; padding: 4px 8px; border-bottom: 1px solid #f3f4f6;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-weight: bold; color: {color_num}; margin-right: 10px; font-size: 1.1em;">{posicion}</span>
                    <span style="flex-grow: 1; font-weight: 500; font-size: 0.95em; color: #374151;">{nombre_prod}</span>
                    <span style="font-weight: bold; color: #1f2937; font-size: 0.9em; margin-left: 10px;">
                        {uds:,} uds <span style="font-weight: normal; color: #6b7280; font-size: 0.85em;">({ordenes:,} órd.)</span>
                    </span>
                </div>
                <div style="width: 100%; background-color: #f3f4f6; height: 6px; border-radius: 3px;">
                    <div style="width: {porcentaje_barra}%; background-color: #f59e0b; height: 6px; border-radius: 3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info(f"No se encontraron productos registrados en 'dataset_vtex_detalle_sku.csv' para la marca '{marca_sel}' en la selección actual.")