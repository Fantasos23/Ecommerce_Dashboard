#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Extracción y Sincronización Automática con la API de Google Ads.
Permite sincronización incremental (Delta Load) o completa directamente
hacia datasets_procesados/dataset_google_unificado.csv

Diseñado con el SDK oficial google-ads y Pandas.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / 'datasets_procesados'
GOOGLE_OUTPUT_CSV = DATASETS_DIR / 'dataset_google_unificado.csv'
ENV_FILE = BASE_DIR / '.env'

COLUMNAS_ESTANDAR = [
    'Día',
    'Impr.',
    'Usuarios únicos',
    'Clics',
    'Coste',
    'Resultados'
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
    
    dev_token = os.environ.get('GOOGLE_ADS_DEVELOPER_TOKEN', config.get('GOOGLE_ADS_DEVELOPER_TOKEN', ''))
    client_id = os.environ.get('GOOGLE_ADS_CLIENT_ID', config.get('GOOGLE_ADS_CLIENT_ID', ''))
    client_secret = os.environ.get('GOOGLE_ADS_CLIENT_SECRET', config.get('GOOGLE_ADS_CLIENT_SECRET', ''))
    refresh_token = os.environ.get('GOOGLE_ADS_REFRESH_TOKEN', config.get('GOOGLE_ADS_REFRESH_TOKEN', ''))
    login_customer_id = os.environ.get('GOOGLE_ADS_LOGIN_CUSTOMER_ID', config.get('GOOGLE_ADS_LOGIN_CUSTOMER_ID', ''))
    customer_id = os.environ.get('GOOGLE_ADS_CUSTOMER_ID', config.get('GOOGLE_ADS_CUSTOMER_ID', '1513120194'))

    # Limpiar guiones de IDs si vienen con formato 123-456-7890
    login_customer_id = str(login_customer_id).replace('-', '').strip()
    customer_id = str(customer_id).replace('-', '').strip()

    return {
        'developer_token': dev_token,
        'client_id': client_id,
        'client_secret': client_secret,
        'refresh_token': refresh_token,
        'login_customer_id': login_customer_id,
        'customer_id': customer_id,
        'use_proto_plus': True
    }


def obtener_cliente_google_ads(config):
    """
    Inicializa el cliente de Google Ads usando el SDK oficial.
    """
    try:
        from google.ads.googleads.client import GoogleAdsClient
    except ImportError:
        raise ImportError(
            "El paquete 'google-ads' no está instalado. "
            "Instálalo ejecutando: pip install google-ads"
        )
    
    dict_creds = {
        'developer_token': config['developer_token'],
        'client_id': config['client_id'],
        'client_secret': config['client_secret'],
        'refresh_token': config['refresh_token'],
        'use_proto_plus': True
    }
    if config['login_customer_id']:
        dict_creds['login_customer_id'] = config['login_customer_id']

    return GoogleAdsClient.load_from_dict(dict_creds)


def obtener_rango_fechas_incremental(dias_retroactivos=7):
    """
    Calcula el rango de fechas para sincronización incremental.
    Comienza unos días antes de la última fecha existente para capturar
    conversiones de atribución tardía.
    """
    hoy = datetime.now()
    fecha_fin = hoy.strftime('%Y-%m-%d')
    fecha_inicio_default = '2023-06-01'

    if GOOGLE_OUTPUT_CSV.exists():
        try:
            df = pd.read_csv(GOOGLE_OUTPUT_CSV, low_memory=False)
            col_fecha = next((c for c in ['Día', 'Dia', 'Day', 'Fecha'] if c in df.columns), None)
            if col_fecha and not df.empty:
                fechas = pd.to_datetime(df[col_fecha], errors='coerce').dropna()
                if not fechas.empty:
                    ultima_fecha = fechas.max()
                    fecha_inicio_calc = ultima_fecha - timedelta(days=dias_retroactivos)
                    return fecha_inicio_calc.strftime('%Y-%m-%d'), fecha_fin
        except Exception as e:
            print(f"⚠️ Error leyendo dataset existente para fecha incremental: {e}")

    return fecha_inicio_default, fecha_fin


def consultar_metricas_diarias(client, customer_id, fecha_inicio, fecha_fin):
    """
    Ejecuta una consulta GAQL en Google Ads API para extraer métricas diarias.
    """
    ga_service = client.get_service("GoogleAdsService")
    
    query = f"""
        SELECT
            segments.date,
            metrics.impressions,
            metrics.clicks,
            metrics.cost_micros,
            metrics.conversions,
            metrics.conversions_value
        FROM customer
        WHERE segments.date >= '{fecha_inicio}' AND segments.date <= '{fecha_fin}'
        ORDER BY segments.date ASC
    """
    
    registros = []
    response = ga_service.search(customer_id=customer_id, query=query)
    
    for row in response:
        fecha = row.segments.date
        impresiones = int(row.metrics.impressions)
        clics = int(row.metrics.clicks)
        # Convertir micros a COP (1 millón de micros = 1 COP)
        coste_cop = round(row.metrics.cost_micros / 1_000_000.0, 2)
        conversiones = round(row.metrics.conversions, 2)
        
        registros.append({
            'Día': fecha,
            'Impr.': impresiones,
            'Usuarios únicos': '--',
            'Clics': clics,
            'Coste': coste_cop,
            'Resultados': conversiones
        })
        
    return registros


def sincronizar_google_ads(modo='incremental', fecha_inicio=None, fecha_fin=None, verbose=True):
    """
    Función principal de sincronización de Google Ads.
    
    Parámetros:
    -----------
    modo : str ('incremental' o 'full')
    fecha_inicio : str ('YYYY-MM-DD', opcional)
    fecha_fin : str ('YYYY-MM-DD', opcional)
    verbose : bool
    
    Retorna:
    --------
    dict con estado de sincronización y métricas.
    """
    config = cargar_configuracion()
    
    if not config['developer_token'] or not config['refresh_token']:
        msg = "Faltan credenciales de Google Ads en .env (GOOGLE_ADS_DEVELOPER_TOKEN o GOOGLE_ADS_REFRESH_TOKEN)"
        if verbose:
            print(f"❌ {msg}")
        return {'status': 'error', 'message': msg}

    customer_id = config['customer_id']
    if not customer_id:
        msg = "No se configuró GOOGLE_ADS_CUSTOMER_ID en .env"
        if verbose:
            print(f"❌ {msg}")
        return {'status': 'error', 'message': msg}

    # Determinar fechas
    if not fecha_fin:
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        
    if not fecha_inicio:
        if modo == 'full':
            fecha_inicio = '2023-06-01'
        else:
            fecha_inicio, fecha_fin = obtener_rango_fechas_incremental()

    if verbose:
        print("="*60)
        print("🚀 INICIANDO SINCRONIZACIÓN GOOGLE ADS API")
        print("="*60)
        print(f"• Cuenta (Customer ID): {customer_id}")
        print(f"• Modo: {modo.upper()}")
        print(f"• Rango de Consulta: {fecha_inicio} al {fecha_fin}")

    try:
        client = obtener_cliente_google_ads(config)
        nuevos_registros = consultar_metricas_diarias(client, customer_id, fecha_inicio, fecha_fin)
        
        if verbose:
            print(f"📥 Registros diarios descargados de Google Ads: {len(nuevos_registros):,}")
            
        df_nuevos = pd.DataFrame(nuevos_registros)
        
        if df_nuevos.empty:
            if verbose:
                print("⚠️ No se encontraron registros en el rango especificado.")
            return {
                'status': 'warning',
                'message': 'No se encontraron registros nuevos en el rango especificado.',
                'registros_sincronizados': 0
            }

        # Asegurar columnas estándar
        for col in COLUMNAS_ESTANDAR:
            if col not in df_nuevos.columns:
                df_nuevos[col] = ''
        df_nuevos = df_nuevos[COLUMNAS_ESTANDAR]

        # Fusionar con dataset histórico existente
        if GOOGLE_OUTPUT_CSV.exists() and modo != 'overwrite':
            try:
                df_existente = pd.read_csv(GOOGLE_OUTPUT_CSV, low_memory=False)
                # Estandarizar nombre de columna fecha
                col_fecha_ex = next((c for c in ['Día', 'Dia', 'Day', 'Fecha'] if c in df_existente.columns), 'Día')
                if col_fecha_ex != 'Día':
                    df_existente.rename(columns={col_fecha_ex: 'Día'}, inplace=True)
                    
                # Combinar y desduplicar manteniendo el más reciente de la API
                df_final = pd.concat([df_existente, df_nuevos], ignore_index=True)
                df_final['Día'] = pd.to_datetime(df_final['Día'], errors='coerce').dt.strftime('%Y-%m-%d')
                df_final = df_final.dropna(subset=['Día'])
                df_final = df_final.drop_duplicates(subset=['Día'], keep='last')
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error al combinar con archivo existente ({e}), usando solo nuevos datos.")
                df_final = df_nuevos
        else:
            df_final = df_nuevos

        # Ordenar por fecha descendente
        df_final = df_final.sort_values(by='Día', ascending=False).reset_index(drop=True)
        
        # Guardar archivo consolidado
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(GOOGLE_OUTPUT_CSV, index=False, encoding='utf-8-sig')

        min_fecha = df_final['Día'].min()
        max_fecha = df_final['Día'].max()

        if verbose:
            print("="*60)
            print("✅ SINCRONIZACIÓN GOOGLE ADS COMPLETADA CON ÉXITO")
            print(f"• Total filas en dataset consolidado: {len(df_final):,}")
            print(f"• Rango histórico total: {min_fecha} a {max_fecha}")
            print(f"• Archivo guardado: {GOOGLE_OUTPUT_CSV}")
            print("="*60)

        return {
            'status': 'success',
            'registros_descargados': len(df_nuevos),
            'total_filas': len(df_final),
            'min_fecha': min_fecha,
            'max_fecha': max_fecha,
            'rango_sincronizado': f"{fecha_inicio} a {fecha_fin}",
            'archivo': str(GOOGLE_OUTPUT_CSV)
        }

    except Exception as e:
        msg = f"Error durante la sincronización de Google Ads: {str(e)}"
        if verbose:
            print(f"❌ {msg}")
        return {'status': 'error', 'message': msg}


def main():
    parser = argparse.ArgumentParser(description="Sincronizador API de Google Ads")
    parser.add_argument('--full', action='store_true', help="Ejecutar sincronización histórica completa (desde 2023)")
    parser.add_argument('--start-date', type=str, default=None, help="Fecha inicio en formato YYYY-MM-DD")
    parser.add_argument('--end-date', type=str, default=None, help="Fecha fin en formato YYYY-MM-DD")
    args = parser.parse_args()

    modo = 'full' if args.full else 'incremental'
    sincronizar_google_ads(modo=modo, fecha_inicio=args.start_date, fecha_fin=args.end_date)


if __name__ == '__main__':
    main()
