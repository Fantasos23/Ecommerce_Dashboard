import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Rutas de archivos
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / '.env'
DATASETS_DIR = BASE_DIR / 'datasets_procesados'
VTEX_UNIFICADO_PATH = DATASETS_DIR / 'dataset_vtex_unificado.csv'


def cargar_env():
    """Carga variables del archivo .env si existen sin depender de python-dotenv."""
    env_vars = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env_vars[k.strip()] = v.strip()
    return env_vars


def get_vtex_credentials():
    """Obtiene las credenciales de VTEX desde .env o variables de entorno del sistema."""
    env = cargar_env()
    app_key = os.environ.get('VTEX_APP_KEY', env.get('VTEX_APP_KEY', 'vtexappkey-recamierco-FRNBER'))
    app_token = os.environ.get('VTEX_APP_TOKEN', env.get('VTEX_APP_TOKEN', 'MCOXVTTWTYEBSCXRKJYLPGQSJWKCCWYPNNMENBOIJSHLGEGYFJYHCXODUZXSVDDDDEYCQWGTJCWQSGMCRKEYJZTVTEOPJFUSTJBQFLMLRYEPACBUJSJSCZNOLFEPISXL'))
    account_name = os.environ.get('VTEX_ACCOUNT_NAME', env.get('VTEX_ACCOUNT_NAME', 'recamierco'))
    environment = os.environ.get('VTEX_ENVIRONMENT', env.get('VTEX_ENVIRONMENT', 'vtexcommercestable.com.br'))
    
    return {
        'app_key': app_key,
        'app_token': app_token,
        'account_name': account_name,
        'environment': environment,
        'base_url': f"https://{account_name}.{environment}/api/oms/pvt/orders"
    }


def realizar_peticion_vtex(url, credentials, max_retries=4):
    """Realiza una petición HTTP a la API de VTEX con control de reintentos y rate-limiting."""
    headers = {
        'X-VTEX-API-AppKey': credentials['app_key'],
        'X-VTEX-API-AppToken': credentials['app_token'],
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 429: # Rate limit
                wait_time = (attempt + 1) * 1.5
                time.sleep(wait_time)
                continue
            elif e.code in [500, 502, 503, 504]:
                time.sleep((attempt + 1) * 1.0)
                continue
            else:
                try:
                    err_msg = e.read().decode('utf-8')
                except Exception:
                    err_msg = str(e)
                print(f"⚠️ Error HTTP {e.code} en {url}: {err_msg}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0)
                continue
            print(f"⚠️ Error de conexión en {url}: {e}")
            return None
    return None


def listar_ordenes_periodo(credentials, fecha_inicio_iso, fecha_fin_iso):
    """
    Consulta todas las órdenes en el rango de fechas f_creationDate.
    Retorna la lista de orderIds.
    """
    base_url = credentials['base_url']
    per_page = 100
    page = 1
    total_orders = []
    
    # Formato de rango para VTEX f_creationDate: creationDate:[2026-08-01T00:00:00.000Z TO 2026-09-04T23:59:59.000Z]
    filtro_fecha = f"creationDate:[{fecha_inicio_iso} TO {fecha_fin_iso}]"
    
    print(f"🔍 Consultando listado de órdenes VTEX ({fecha_inicio_iso[:10]} al {fecha_fin_iso[:10]})...")
    
    while True:
        params = {
            'f_creationDate': filtro_fecha,
            'per_page': per_page,
            'page': page,
            'orderBy': 'creationDate,desc'
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{base_url}?{query_string}"
        
        data = realizar_peticion_vtex(url, credentials)
        if not data or not isinstance(data, dict):
            break
            
        orders_list = data.get('list', [])
        if not orders_list:
            break
            
        for o in orders_list:
            oid = o.get('orderId')
            if oid:
                total_orders.append(oid)
                
        paging = data.get('paging', {})
        total_pages = paging.get('pages', 1)
        current_page = paging.get('currentPage', page)
        
        print(f"  └─ Página {current_page}/{total_pages} — {len(total_orders)} órdenes acumuladas...")
        
        if current_page >= total_pages or page >= 100: # Límite de paginación VTEX OMS
            break
            
        page += 1
        time.sleep(0.2) # Pausa amigable para respetar el rate limit de VTEX
        
    return list(dict.fromkeys(total_orders)) # Eliminar duplicados manteniendo orden


def obtener_detalle_orden(order_id, credentials):
    """Consulta el detalle completo de una orden."""
    url = f"{credentials['base_url']}/{order_id}"
    return realizar_peticion_vtex(url, credentials)


def parsear_orden_a_filas(order_data, credentials):
    """
    Convierte el JSON completo de una orden de VTEX a una lista de diccionarios
    con el formato idéntico de columnas de 'dataset_vtex_unificado.csv'.
    """
    if not order_data or not isinstance(order_data, dict):
        return []
        
    order_id = order_data.get('orderId', '')
    sequence = order_data.get('sequence', '')
    origin = order_data.get('origin', 'Marketplace')
    
    # 1. Fecha de Creación y Último Cambio
    creation_date_raw = order_data.get('creationDate', '')
    if creation_date_raw:
        try:
            dt = pd.to_datetime(creation_date_raw)
            creation_date_str = dt.strftime('%Y-%m-%d %H:%M:%SZ')
        except Exception:
            creation_date_str = str(creation_date_raw)
    else:
        creation_date_str = ''
        
    last_change_raw = order_data.get('lastChange', '')
    if last_change_raw:
        try:
            dt_lc = pd.to_datetime(last_change_raw)
            last_change_str = dt_lc.strftime('%Y-%m-%d %H:%M:%SZ')
        except Exception:
            last_change_str = str(last_change_raw)
    else:
        last_change_str = ''

    # 2. Cliente
    client = order_data.get('clientProfileData') or {}
    first_name = client.get('firstName') or ''
    last_name = client.get('lastName') or ''
    client_name = f"{first_name}".strip()
    client_last_name = f"{last_name}".strip()
    client_doc = client.get('document') or ''
    email = client.get('email') or ''
    phone = client.get('phone') or ''

    # 3. Dirección y Envíos
    shipping = order_data.get('shippingData') or {}
    address = shipping.get('address') or {}
    city = address.get('city') or ''
    uf = address.get('state') or ''
    addr_id = address.get('addressId') or ''
    addr_type = address.get('addressType') or 'residential'
    receiver_name = address.get('receiverName') or ''
    street = address.get('street') or ''
    number = address.get('number') or ''
    complement = address.get('complement') or ''
    neighborhood = address.get('neighborhood') or ''
    reference = address.get('reference') or ''
    postal_code = address.get('postalCode') or ''
    
    logistics = shipping.get('logisticsInfo') or []
    sla_type = logistics[0].get('selectedSla') if logistics else ''
    courrier = logistics[0].get('deliveryCompany') if logistics else ''
    delivery_deadline = logistics[0].get('shippingEstimate') if logistics else ''
    est_date_raw = logistics[0].get('shippingEstimateDate') if logistics else ''
    estimate_delivery_date = ''
    if est_date_raw:
        try:
            estimate_delivery_date = pd.to_datetime(est_date_raw).strftime('%Y-%m-%d %H:%M:%SZ')
        except Exception:
            estimate_delivery_date = str(est_date_raw)

    # 4. Estado
    status_raw = order_data.get('status') or ''
    status_desc = order_data.get('statusDescription') or ''
    if status_raw.lower() in ['invoiced', 'faturado']:
        status_clean = 'Faturado'
    elif status_raw.lower() in ['canceled', 'cancelado']:
        status_clean = 'Cancelado'
    elif status_desc:
        status_clean = status_desc
    else:
        status_clean = status_raw.capitalize()

    # 5. Marketing (UTMs & Cupones)
    marketing = order_data.get('marketingData') or {}
    utm_source = marketing.get('utmSource') or ''
    utm_medium = marketing.get('utmMedium') or ''
    utm_campaign = marketing.get('utmCampaign') or ''
    coupon = marketing.get('coupon') or ''
    marketing_tags_list = marketing.get('marketingTags') or []
    marketing_tags = ', '.join(marketing_tags_list) if isinstance(marketing_tags_list, list) else str(marketing_tags_list)

    # 6. Pago
    payment_data = order_data.get('paymentData') or {}
    transactions = payment_data.get('transactions') or []
    payment_system_name = ''
    installments = 1
    payment_value = 0.0
    auth_id = ''
    tid = ''
    nsu = ''
    transaction_id = ''
    payment_id = ''
    
    if transactions:
        tx = transactions[0]
        transaction_id = tx.get('transactionId') or ''
        payments = tx.get('payments') or []
        if payments:
            p = payments[0]
            payment_system_name = p.get('paymentSystemName') or ''
            installments = p.get('installments') or 1
            # VTEX almacena en centavos: dividir entre 100
            payment_value = (p.get('value') or 0) / 100.0
            tid = p.get('tid') or ''
            payment_id = p.get('id') or ''
            conn_resp = p.get('connectorResponses') or {}
            auth_id = conn_resp.get('authId') or ''
            nsu = conn_resp.get('nsu') or ''

    # 7. Descuentos y Totales
    rates = order_data.get('ratesAndBenefitsData') or {}
    rate_ids = rates.get('rateAndBenefitsIdentifiers') or []
    discount_names_list = [r.get('name') for r in rate_ids if r.get('name')]
    discounts_names = ', '.join(discount_names_list)

    totals = order_data.get('totals') or []
    discounts_totals = 0.0
    shipping_value = 0.0
    for t in totals:
        if t.get('id') == 'Discounts':
            discounts_totals = (t.get('value') or 0) / 100.0
        elif t.get('id') == 'Shipping':
            shipping_value = (t.get('value') or 0) / 100.0

    # Total Value de la orden (en COP)
    total_value_orden = (order_data.get('value') or 0) / 100.0

    # 8. Ítems / SKUs
    items = order_data.get('items') or []
    filas_orden = []

    for item_idx, item in enumerate(items):
        item_id = item.get('id') or ''
        ref_id = item.get('refId') or ''
        item_name = item.get('name') or ''
        quantity = int(item.get('quantity') or 1)
        sku_val = (item.get('price') or 0) / 100.0
        sku_selling = (item.get('sellingPrice') or 0) / 100.0
        sku_total = (sku_selling * quantity)
        
        # Shipping list price de este ítem
        item_ship_list_price = 0.0
        if item_idx < len(logistics):
            item_ship_list_price = (logistics[item_idx].get('listPrice') or 0) / 100.0

        fila = {
            'Origin': origin,
            'Order': order_id,
            'Sequence': sequence,
            'Creation Date': creation_date_str,
            'Client Name': client_name,
            'Client Last Name': client_last_name,
            'Client Document': client_doc,
            'Email': email,
            'Phone': phone,
            'UF': uf,
            'City': city,
            'Address Identification': addr_id,
            'Address Type': addr_type,
            'Receiver Name': receiver_name,
            'Street': street,
            'Number': number,
            'Complement': complement,
            'Neighborhood': neighborhood,
            'Reference': reference,
            'Postal Code': postal_code,
            'SLA Type': sla_type,
            'Courrier': courrier,
            'Estimate Delivery Date': estimate_delivery_date,
            'Delivery Deadline': delivery_deadline,
            'Status': status_clean,
            'Last Change Date': last_change_str,
            'UtmMedium': utm_medium,
            'UtmSource': utm_source,
            'UtmCampaign': utm_campaign,
            'Coupon': coupon,
            'Payment System Name': payment_system_name,
            'Installments': installments,
            'Payment Value': payment_value,
            'Quantity_SKU': quantity,
            'ID_SKU': item_id,
            'Category Ids Sku': item.get('additionalInfo', {}).get('categoriesIds', '') if isinstance(item.get('additionalInfo'), dict) else '',
            'Reference Code': ref_id,
            'SKU Name': item_name,
            'SKU Value': sku_val,
            'SKU Selling Price': sku_selling,
            'SKU Total Price': sku_total,
            'SKU Path': item.get('detailUrl', ''),
            'Item Attachments': '',
            'List Id': '',
            'List Type Name': '',
            'Service (Price/ Selling Price)': '',
            'Shipping List Price': item_ship_list_price,
            'Shipping Value': shipping_value,
            'Total Value': total_value_orden,
            'Discounts Totals': discounts_totals,
            'Discounts Names': discounts_names,
            'Call Center Email': '',
            'Call Center Code': '',
            'Tracking Number': '',
            'Host': credentials['account_name'],
            'GiftRegistry ID': '',
            'Seller Name': 'Recamier S.A.',
            'Status TimeLine': '',
            'Obs': '',
            'UtmiPart': '',
            'UtmiCampaign': '',
            'UtmiPage': '',
            'Seller Order Id': '',
            'Acquirer': '',
            'Authorization Id': auth_id,
            'TID': tid,
            'NSU': nsu,
            'Card First Digits': '',
            'Card Last Digits': '',
            'Payment Approved By': '',
            'Cancelled By': '',
            'Cancellation Reason': '',
            'Gift Card Name': '',
            'Gift Card Caption': '',
            'Authorized Date': creation_date_str,
            'Corporate Name': '',
            'Corporate Document': '',
            'TransactionId': transaction_id,
            'PaymentId': payment_id,
            'PaymentOrigin': '',
            'SalesChannel': 1,
            'marketingTags': marketing_tags,
            'Delivered': 0,
            'SKU RewardValue': 0,
            'Is Marketplace cetified': '',
            'Is Checked In': '',
            'Currency Code': 'COP',
            'Taxes': '',
            'Invoice Numbers': '',
            'Country': 'COL',
            'Input Invoices Numbers': '',
            'Output Invoices Numbers': '',
            'Status raw value (temporary)': status_clean,
            'Cancellation Data': ''
        }
        filas_orden.append(fila)

    return filas_orden


def sincronizar_vtex_orders(lookback_days=7, fecha_inicio=None, fecha_fin=None, max_workers=8):
    """
    Función principal de sincronización con VTEX OMS API.
    1. Determina la ventana de fechas incremental.
    2. Consulta la lista de órdenes nuevas/modificadas.
    3. Descarga concurrentemente el detalle completo de cada orden.
    4. Convierte los ítems a filas de dataset estructurado.
    5. Fusiona de forma limpia y desduplicada con dataset_vtex_unificado.csv.
    6. Ejecuta automáticamente la agregación de órdenes y SKUs (procesar_vtex_agrupado).
    """
    print("\n" + "="*60)
    print("🚀 INICIANDO SINCRONIZACIÓN DE PEDIDOS VTEX OMS API")
    print("="*60)
    
    credentials = get_vtex_credentials()
    now_utc = datetime.now(timezone.utc)
    
    # Determinar ventana de tiempo
    if fecha_inicio and fecha_fin:
        dt_ini = pd.to_datetime(fecha_inicio).tz_localize('UTC') if pd.to_datetime(fecha_inicio).tz is None else pd.to_datetime(fecha_inicio)
        dt_fin = pd.to_datetime(fecha_fin).tz_localize('UTC') if pd.to_datetime(fecha_fin).tz is None else pd.to_datetime(fecha_fin)
    else:
        # Modo incremental: busca la última fecha registrada en dataset_vtex_unificado
        if VTEX_UNIFICADO_PATH.exists():
            try:
                df_existente_fechas = pd.read_csv(VTEX_UNIFICADO_PATH, usecols=['Creation Date'], low_memory=False)
                df_existente_fechas['dt'] = pd.to_datetime(df_existente_fechas['Creation Date'], errors='coerce')
                max_dt = df_existente_fechas['dt'].max()
                if pd.notnull(max_dt):
                    # Retrocedemos los días de lookback para capturar cambios de estado en órdenes recientes
                    dt_ini = (max_dt - timedelta(days=lookback_days)).tz_localize('UTC') if max_dt.tz is None else (max_dt - timedelta(days=lookback_days))
                else:
                    dt_ini = now_utc - timedelta(days=lookback_days)
            except Exception:
                dt_ini = now_utc - timedelta(days=lookback_days)
        else:
            dt_ini = now_utc - timedelta(days=lookback_days)
            
        dt_fin = now_utc + timedelta(days=1)

    inicio_iso = dt_ini.strftime('%Y-%m-%dT00:00:00.000Z')
    fin_iso = dt_fin.strftime('%Y-%m-%dT23:59:59.000Z')
    
    print(f"📅 Rango de consulta: {inicio_iso[:10]} al {fin_iso[:10]}")
    
    # 1. Obtener lista de IDs de órdenes
    order_ids = listar_ordenes_periodo(credentials, inicio_iso, fin_iso)
    print(f"📦 Total de órdenes encontradas en el período: {len(order_ids):,}")
    
    if not order_ids:
        print("✨ No se encontraron órdenes nuevas o modificadas en el período.")
        return {'status': 'success', 'nuevas_ordenes': 0, 'filas_agregadas': 0}

    # 2. Descargar detalles en paralelo
    print(f"⚡ Descargando detalle de {len(order_ids):,} órdenes con {max_workers} hilos...")
    todas_filas_nuevas = []
    ordenes_descargadas = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_order = {executor.submit(obtener_detalle_orden, oid, credentials): oid for oid in order_ids}
        for future in as_completed(future_to_order):
            oid = future_to_order[future]
            try:
                order_json = future.result()
                if order_json:
                    filas = parsear_orden_a_filas(order_json, credentials)
                    todas_filas_nuevas.extend(filas)
                    ordenes_descargadas += 1
                    if ordenes_descargadas % 50 == 0 or ordenes_descargadas == len(order_ids):
                        print(f"  └─ Progreso: {ordenes_descargadas}/{len(order_ids)} órdenes procesadas...")
            except Exception as e:
                print(f"⚠️ Error procesando orden {oid}: {e}")

    print(f"✅ Descarga completada: {ordenes_descargadas} órdenes procesadas ({len(todas_filas_nuevas):,} ítems/filas generadas).")
    
    if not todas_filas_nuevas:
        return {'status': 'success', 'nuevas_ordenes': 0, 'filas_agregadas': 0}

    df_nuevas = pd.DataFrame(todas_filas_nuevas)

    # 3. Fusionar con dataset histórico existente
    if VTEX_UNIFICADO_PATH.exists():
        print(f"📂 Fusionando con dataset histórico existente ({VTEX_UNIFICADO_PATH.name})...")
        df_historico = pd.read_csv(VTEX_UNIFICADO_PATH, low_memory=False)
        
        # Eliminar las órdenes que estamos actualizando del histórico para evitar duplicados y actualizar estados
        ordenes_actualizadas_set = set(df_nuevas['Order'].dropna().unique())
        df_historico_filtrado = df_historico[~df_historico['Order'].isin(ordenes_actualizadas_set)]
        
        # Unir histórico + nuevas
        df_consolidado = pd.concat([df_historico_filtrado, df_nuevas], ignore_index=True)
    else:
        df_consolidado = df_nuevas

    # Limpieza final de duplicados exactos
    df_consolidado = df_consolidado.drop_duplicates(keep='last')
    
    # Guardar en CSV unificado
    VTEX_UNIFICADO_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_consolidado.to_csv(VTEX_UNIFICADO_PATH, index=False, encoding='utf-8-sig')
    print(f"💾 Guardado dataset unificado VTEX: {len(df_consolidado):,} filas totales en {VTEX_UNIFICADO_PATH.name}")

    # 4. Actualizar agregados (dataset_vtex_agrupado_ordenes y dataset_vtex_detalle_skus)
    try:
        from procesar_vtex_agrupado import generar_dataset_vtex_por_orden
        print("\n🔄 Actualizando datasets derivados (órdenes agrupadas y detalle SKUs)...")
        generar_dataset_vtex_por_orden()
    except Exception as e:
        print(f"⚠️ Error al generar datasets derivados: {e}")

    max_fecha_final = df_consolidado['Creation Date'].dropna().max()
    print(f"\n🎉 Sincronización VTEX finalizada con éxito. Última fecha registrada: {max_fecha_final}")
    
    return {
        'status': 'success',
        'nuevas_ordenes': ordenes_descargadas,
        'filas_agregadas': len(todas_filas_nuevas),
        'ultima_fecha': max_fecha_final
    }


if __name__ == '__main__':
    # Ejecución manual directa desde terminal
    resultado = sincronizar_vtex_orders(lookback_days=7)
    print("\nResultado:", json.dumps(resultado, indent=2))
