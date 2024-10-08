import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

# Lista empresas argentinas que cotizan en el MERVAL GENERAL
tickers_arg = [
    "YPFD.BA", "GGAL.BA", "BMA.BA", "TECO2.BA", "CRES.BA", 
    "EDN.BA", "BBAR.BA", "ALUA.BA", "COME.BA", "PAMP.BA"
]

# Función para obtener los datos de acciones y calcular rendimientos y covarianza
def obtener_datos(tickers, start, end=None, interval="1d"):
    try:
        print(f"Descargando datos para: {tickers}")
        data = yf.download(tickers, start=start, end=end, interval=interval)['Adj Close']
        data = data.interpolate().dropna()  # Interpola valores faltantes y luego elimina si quedan NaN

        # Calcular rendimientos y covarianza
        rendimientos = data.pct_change().mean() * 100  # Rendimientos promedio
        covarianza = data.pct_change().cov()  # Matriz de covarianza
        return rendimientos, covarianza
    except Exception as e:
        print(f"Error al obtener datos: {e}")
        return None, None

# Función para calcular el rendimiento esperado del portafolio
def rendimiento_portafolio(pesos, rendimientos):
    return np.sum(pesos * rendimientos)

# Función para calcular el riesgo (volatilidad) del portafolio
def riesgo_portafolio(pesos, covarianza):
    return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))

# Función para calcular el Sharpe ratio
def sharpe_ratio(pesos, rendimientos, covarianza, rf=0.4):  # rf es la tasa libre de riesgo (40% en Argentina, oct-2024)
    return (rendimiento_portafolio(pesos, rendimientos) - rf) / riesgo_portafolio(pesos, covarianza)

# Función para optimizar el portafolio basado en rendimientos y covarianza
def optimizar_portafolio(rendimientos, covarianza):
    num_acciones = len(rendimientos)
    pesos_iniciales = np.ones(num_acciones) / num_acciones  # Inicializar con pesos iguales
    limites = [(0, 1) for _ in range(num_acciones)]  # Los pesos deben estar entre 0 y 1

    # Restricción: la suma de los pesos debe ser 1
    restricciones = {'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1}

    # Optimización con Scipy
    resultado = minimize(lambda pesos: -sharpe_ratio(pesos, rendimientos, covarianza), 
                         pesos_iniciales, method='SLSQP', bounds=limites, constraints=restricciones)

    return resultado.x  # Retorna los pesos óptimos

# Función para encontrar la combinación óptima de tickets
def optimizar_tickets(cantidad_tickets, start):
    mejores_tickets = None
    mejor_sharpe = -np.inf

    # Probar todas las combinaciones posibles de tickets
    for combinacion in combinations(tickers_arg, cantidad_tickets):
        print(f"Probando combinación: {combinacion}")
        rendimientos, covarianza = obtener_datos(combinacion, start=start)  # Asegúrate de llamar aquí

        if rendimientos is not None and covarianza is not None:
            # Optimizar el portafolio para esta combinación
            pesos_optimos = optimizar_portafolio(rendimientos, covarianza)
            sharpe = sharpe_ratio(pesos_optimos, rendimientos, covarianza)

            # Mantener la mejor combinación basada en el Sharpe ratio
            if sharpe > mejor_sharpe:
                mejor_sharpe = sharpe
                mejores_tickets = combinacion

    return mejores_tickets

# Función principal para calcular el portafolio óptimo
def calcular_portafolio_optimo(cantidad_tickets, start):
    # Encontrar los mejores tickets que optimizan el rendimiento/riesgo
    mejores_tickets = optimizar_tickets(cantidad_tickets, start)

    print(f'Tickets seleccionados: {mejores_tickets}')
    
    # Obtener datos de los mejores tickets
    rendimientos, covarianza = obtener_datos(mejores_tickets, start=start)  # Asegúrate de llamar aquí
    
    if rendimientos is None or covarianza is None:
        print("No se pudieron obtener datos para los tickets seleccionados.")
        return
    
    # Optimizar el portafolio
    pesos_optimos = optimizar_portafolio(rendimientos, covarianza)
    
    # Mostrar resultados
    for i, ticker in enumerate(mejores_tickets):
        print(f'{ticker}: {pesos_optimos[i]:.2%} del portafolio')

    rendimiento_esperado = rendimiento_portafolio(pesos_optimos, rendimientos)
    riesgo_esperado = riesgo_portafolio(pesos_optimos, covarianza)
    
    print(f'Rendimiento esperado del portafolio: {rendimiento_esperado:.2%}')
    print(f'Riesgo (volatilidad) del portafolio: {riesgo_esperado:.2%}')

# Ejemplo de uso
cantidad_tickets = 3 # El usuario elige la cantidad de tickets
calcular_portafolio_optimo(cantidad_tickets, start="2023-01-01")


