#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Extracción y Sincronización Automática con Meta Ads Graph API.
Permite sincronización incremental (Delta Load) o completa directamente
hacia datasets_procesados/dataset_meta_unificado.csv

Diseñado con compatibilidad universal (Python estándar + Pandas).
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
import urllib.error

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / 'datasets_procesados'
META_OUTPUT_CSV = DATASETS_DIR / 'dataset_meta_unificado.csv'
ENV_FILE = BASE_DIR / '.env'

GRAPH_API_VERSION = 'v20.0'
META_GRAPH_BASE_URL = f'https://graph.facebook.com/{GRAPH_API_VERSION}'

COLUMNAS_ESTANDAR = [
    'Día',
    'Alcance',
    'Impresiones',
    'Frecuencia',
    'Importe gastado (COP)',
    'Clics en el enlace',
    'Visitas a la página de destino',
    'Artículos agregados al carrito',
    'Pagos iniciados',
    'Información de pago agregada',
    'Compras',
    'Inicio del informe',
    'Fin del informe'
]


def cargar_configuracion():
    """
    Carga variables de entorno desde el archivo .env o del sistema.
    """
    config = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    config[k.strip()] = v.strip().strip('"').strip("'")
    
    app_id = os.environ.get('META_APP_ID', config.get('META_APP_ID', ''))
    app_secret = os.environ.get('META_APP_SECRET', config.get('META_APP_SECRET', ''))
    access_token = os.environ.get('META_ACCESS_TOKEN', config.get('META_ACCESS_TOKEN', ''))
    ad_accounts = os.environ.get('META_AD_ACCOUNTS', config.get('META_AD_ACCOUNTS', ''))

    account_list = []
    if ad_accounts:
        for acc in ad_accounts.split(','):
            acc_clean = acc.strip()
            if acc_clean:
                if not acc_clean.startswith('act_'):
                    acc_clean = f'act_{acc_clean}'
                account_list.append(acc_clean)

    return {
        'META_APP_ID': app_id,
        'META_APP_SECRET': app_secret,
        'META_ACCESS_TOKEN': access_token,
        'META_AD_ACCOUNTS': account_list
    }


def extraer_valor_accion(actions_list, posibles_tipos):
    """
    Extrae el valor numérico de una lista de acciones de Meta según tipos prioritarios.
    """
    if not isinstance(actions_list, list):
        return 0.0
    for tipo in posibles_tipos:
        for item in actions_list:
            if item.get('action_type') == tipo:
                try:
                    return float(item.get('value', 0))
                except (ValueError, TypeError):
                    return 0.0
    return 0.0


def consultar_insights_cuenta(ad_account_id, access_token, since_date, until_date):
    """
    Consulta los insights diarios de una cuenta publicitaria en el rango de fechas especificado.
    Maneja paginación automáticamente.
    """
    params = {
        'access_token': access_token,
        'level': 'account',
        'time_increment': '1',
        'time_range': json.dumps({'since': since_date, 'until': until_date}),
        'fields': 'date_start,date_stop,reach,impressions,frequency,spend,inline_link_clicks,actions'
    }

    url = f"{META_GRAPH_BASE_URL}/{ad_account_id}/insights?{urllib.parse.urlencode(params)}"
    registros = []

    while url:
        req = urllib.request.Request(url, headers={'User-Agent': 'TiendaCo-MetaSync/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                
                if 'error' in res_data:
                    raise Exception(f"Meta API Error: {res_data['error'].get('message')}")
                
                data = res_data.get('data', [])
                for item in data:
                    fecha = item.get('date_start', '')
                    reach = float(item.get('reach', 0) or 0)
                    impressions = float(item.get('impressions', 0) or 0)
                    frequency = float(item.get('frequency', 1.0) or 1.0)
                    spend = float(item.get('spend', 0.0) or 0.0)
                    clicks = float(item.get('inline_link_clicks', 0) or 0)
                    
                    actions = item.get('actions', [])
                    
                    lp_views = extraer_valor_accion(actions, ['landing_page_view', 'omni_landing_page_view'])
                    add_to_cart = extraer_valor_accion(actions, ['add_to_cart', 'omni_add_to_cart', 'offsite_conversion.fb_pixel_add_to_cart'])
                    initiate_checkout = extraer_valor_accion(actions, ['initiate_checkout', 'omni_initiated_checkout', 'offsite_conversion.fb_pixel_initiate_checkout'])
                    add_payment = extraer_valor_accion(actions, ['add_payment_info', 'omni_add_payment_info', 'offsite_conversion.fb_pixel_add_payment_info'])
                    purchases = extraer_valor_accion(actions, ['purchase', 'omni_purchase', 'offsite_conversion.fb_pixel_purchase'])
                    
                    registros.append({
                        'Día': fecha,
                        'Alcance': int(reach),
                        'Impresiones': int(impressions),
                        'Frecuencia': round(frequency, 6),
                        'Importe gastado (COP)': int(round(spend)),
                        'Clics en el enlace': int(clicks) if clicks > 0 else '',
                        'Visitas a la página de destino': int(lp_views) if lp_views > 0 else '',
                        'Artículos agregados al carrito': int(add_to_cart) if add_to_cart > 0 else '',
                        'Pagos iniciados': int(initiate_checkout) if initiate_checkout > 0 else '',
                        'Información de pago agregada': int(add_payment) if add_payment > 0 else '',
                        'Compras': int(purchases) if purchases > 0 else '',
                        'Inicio del informe': fecha,
                        'Fin del informe': item.get('date_stop', fecha)
                    })
                
                # Siguiente página si existe
                paging = res_data.get('paging', {})
                url = paging.get('next')

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8', errors='ignore')
            raise Exception(f"HTTP Error {e.code} consultando cuenta {ad_account_id}: {err_msg}")
        except Exception as e:
            raise Exception(f"Error consultando cuenta {ad_account_id}: {str(e)}")

    return registros


def leer_dataset_existente(ruta_csv):
    """
    Lee el archivo CSV histórico retornando una lista de diccionarios con formato estandarizado.
    """
    if not ruta_csv.exists():
        return []
    
    filas = []
    try:
        with open(ruta_csv, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Estandarizar nombre de columna fecha si viniera distinto
                fecha = row.get('Día') or row.get('Inicio del informe') or row.get('Fecha') or row.get('Day', '')
                if fecha:
                    clean_row = {col: row.get(col, '') for col in COLUMNAS_ESTANDAR}
                    clean_row['Día'] = fecha.strip()
                    clean_row['Inicio del informe'] = clean_row['Inicio del informe'] or clean_row['Día']
                    clean_row['Fin del informe'] = clean_row['Fin del informe'] or clean_row['Día']
                    filas.append(clean_row)
    except Exception as e:
        print(f"⚠️ Advertencia al leer CSV previo: {e}")
        return []
    
    return filas


def parsear_fecha(str_fecha):
    """
    Convierte un string de fecha (YYYY-MM-DD) a datetime.
    """
    try:
        return datetime.strptime(str_fecha.strip()[:10], '%Y-%m-%d')
    except Exception:
        return None


def sincronizar_meta_ads(lookback_dias=7, full_sync=False, fecha_inicio_default='2023-01-01'):
    """
    Ejecuta el proceso de sincronización incremental o completa con Meta Ads API
    y actualiza el archivo datasets_procesados/dataset_meta_unificado.csv.
    
    Retorna un diccionario con el resumen de la operación.
    """
    config = cargar_configuracion()
    access_token = config.get('META_ACCESS_TOKEN')
    ad_accounts = config.get('META_AD_ACCOUNTS')

    if not access_token:
        raise ValueError("❌ No se encontró META_ACCESS_TOKEN en el archivo .env ni en variables de entorno.")
    
    if not ad_accounts:
        raise ValueError("❌ No se encontraron cuentas en META_AD_ACCOUNTS.")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Leer dataset previo si existe y determinar fecha de inicio
    filas_existentes = [] if full_sync else leer_dataset_existente(META_OUTPUT_CSV)
    fecha_hoy_str = datetime.now().strftime('%Y-%m-%d')
    since_date = fecha_inicio_default

    if filas_existentes and not full_sync:
        fechas = [parsear_fecha(r['Día']) for r in filas_existentes if parsear_fecha(r['Día'])]
        if fechas:
            max_fecha = max(fechas)
            fecha_lookback = max_fecha - timedelta(days=lookback_dias)
            since_date = fecha_lookback.strftime('%Y-%m-%d')

    until_date = fecha_hoy_str

    print("=" * 60)
    print("🚀 SINCRONIZACIÓN DE META ADS GRAPH API")
    print(f"📅 Rango a consultar: {since_date} al {until_date}")
    print(f"💼 Cuentas a procesar: {', '.join(ad_accounts)}")
    print(f"⚙️ Modo: {'COMPLETO' if full_sync else f'INCREMENTAL (lookback={lookback_dias} días)'}")
    print("=" * 60)

    # 2. Consultar registros de cada cuenta
    registros_nuevos = []
    for acc in ad_accounts:
        print(f"📡 Consultando cuenta {acc}...")
        registros_acc = consultar_insights_cuenta(acc, access_token, since_date, until_date)
        print(f"   └── Obtenidos {len(registros_acc)} registros diarios.")
        registros_nuevos.extend(registros_acc)

    if not registros_nuevos:
        print("ℹ️ No se recibieron nuevos registros desde la API.")
        return {
            'status': 'no_data',
            'mensaje': 'No se encontraron datos en el rango seleccionado.',
            'filas_totales': len(filas_existentes)
        }

    # 3. Consolidar por día si hay múltiples cuentas publicitarias
    mapa_nuevos = {}
    for r in registros_nuevos:
        dia = r['Día']
        if dia not in mapa_nuevos:
            mapa_nuevos[dia] = r.copy()
        else:
            prev = mapa_nuevos[dia]
            prev['Alcance'] = int(round(float(prev['Alcance'] or 0))) + int(round(float(r['Alcance'] or 0)))
            prev['Impresiones'] = int(round(float(prev['Impresiones'] or 0))) + int(round(float(r['Impresiones'] or 0)))
            prev['Importe gastado (COP)'] = int(round(float(prev['Importe gastado (COP)'] or 0) + float(r['Importe gastado (COP)'] or 0)))
            
            # Sumar métricas numéricas opcionales
            for k in ['Clics en el enlace', 'Visitas a la página de destino', 'Artículos agregados al carrito',
                      'Pagos iniciados', 'Información de pago agregada', 'Compras']:
                v_p = float(prev[k]) if str(prev[k]).strip() else 0.0
                v_r = float(r[k]) if str(r[k]).strip() else 0.0
                total_k = int(round(v_p + v_r))
                prev[k] = total_k if total_k > 0 else ''
            
            # Recalcular frecuencia ponderada
            if prev['Alcance'] > 0:
                prev['Frecuencia'] = round(prev['Impresiones'] / prev['Alcance'], 6)

    # 4. Combinar con dataset histórico (sobrescribiendo los días en mapa_nuevos)
    mapa_final = {}
    for r in filas_existentes:
        dia = r.get('Día')
        if dia and dia not in mapa_nuevos:
            mapa_final[dia] = r

    for dia, r in mapa_nuevos.items():
        mapa_final[dia] = r

    # 5. Ordenar cronológicamente ascendente
    dias_ordenados = sorted(mapa_final.keys(), key=lambda x: parsear_fecha(x) or datetime.min)
    filas_finales = [mapa_final[d] for d in dias_ordenados if d]

    # 6. Guardar en CSV
    with open(META_OUTPUT_CSV, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_ESTANDAR)
        writer.writeheader()
        for row in filas_finales:
            writer.writerow({col: row.get(col, '') for col in COLUMNAS_ESTANDAR})

    primera_fecha = dias_ordenados[0] if dias_ordenados else 'N/A'
    ultima_fecha = dias_ordenados[-1] if dias_ordenados else 'N/A'

    print("\n" + "=" * 60)
    print("✅ SINCRONIZACIÓN COMPLETADA CON ÉXITO")
    print(f"📊 Filas totales guardadas: {len(filas_finales):,}")
    print(f"📅 Rango total de fechas: {primera_fecha} a {ultima_fecha}")
    print(f"📁 Guardado en: {META_OUTPUT_CSV}")
    print("=" * 60)

    return {
        'status': 'success',
        'dias_actualizados': len(mapa_nuevos),
        'filas_totales': len(filas_finales),
        'primera_fecha': str(primera_fecha),
        'ultima_fecha': str(ultima_fecha),
        'archivo': str(META_OUTPUT_CSV)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sincronizador de Meta Ads API')
    parser.add_argument('--lookback', type=int, default=7, help='Días hacia atrás para actualizar atribución (def: 7)')
    parser.add_argument('--full', action='store_true', help='Ejecutar extracción completa de todo el histórico')
    parser.add_argument('--since', type=str, default='2023-01-01', help='Fecha inicial para extracción completa (YYYY-MM-DD)')

    args = parser.parse_args()

    try:
        sincronizar_meta_ads(
            lookback_dias=args.lookback,
            full_sync=args.full,
            fecha_inicio_default=args.since
        )
    except Exception as err:
        print(f"\n❌ Error durante la sincronización: {err}", file=sys.stderr)
        sys.exit(1)
