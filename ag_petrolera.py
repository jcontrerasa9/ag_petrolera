import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt

# PARAMETROS
SEMILLA = 42 # Semilla del generador aleatorio (hace la corrida reproducible)
PRESUPUESTO = 165 # Presupuesto maximo disponible
TAM_POBLACION = 10 # Numero de individuos en cada iteracion
N_GENERACIONES = 100 # Numero de iteraciones
P_CRUCE = 0.89 # La probabilidad de que dos padres se crucen
P_MUTACION = 0.18 # La probabilida de que cada bit mute

# Zonas petroleras: (galones por pozo (Gi), costo por pozo (Ci), pozos disponibles)
ZONAS = [(409, 41, 1), (171, 28, 3), (200, 33, 1), (208, 32, 1), (201, 30, 1)]

# Funcion objetivo: maximizar la produccion total de galones
# Restriccion: costo total <= PRESUPUESTO

# CODIFICACIÓN
BITS = [] # Array que guarda cuantos bits usa cada gen
for zona in ZONAS:
    disp = zona[2]                            # pozos disponibles
    bits = math.ceil(math.log2(disp + 1))     # bits necesarios
    BITS.append(bits)                         # guardarlos en la lista
L = sum(BITS)  # largo total del cromosoma


def texto(cromosoma):
    """Cromosoma como cadena de bits, por ejemplo '101111'."""
    return "".join(map(str, cromosoma))


def decodificar(cromosoma):
    """Convierte el cromosoma en la cantidad de pozos elegidos por zona."""
    cantidades = []
    inicio = 0
    for n in BITS:
        bits = cromosoma[inicio:inicio + n] # Corta el trozo del gen actual
        cantidades.append(int(texto(bits), 2)) # Lee el trozo como numero binario
        inicio += n # EL siguiente gen inicia donde finalizo el anterior
    return cantidades


def evaluar(cromosoma):
    """Devuelve (galones, costo, factible) para una selección."""
    cantidades = decodificar(cromosoma)
    galones = sum(cantidades[j] * ZONAS[j][0] for j in range(len(ZONAS)))
    costo = sum(cantidades[j] * ZONAS[j][1] for j in range(len(ZONAS)))
    pide_de_mas = any(
        cantidades[j] > ZONAS[j][2] for j in range(len(ZONAS))
    )
    factible = costo <= PRESUPUESTO and not pide_de_mas
    return galones, costo, factible


def fitness(cromosoma):
    """Devuelve los galones si la selección es factible; si no, devuelve 0."""
    galones, _, factible = evaluar(cromosoma)
    return galones if factible else 0


# OPERADORES GENETICOS
def cruzar(padre1, padre2):
    """Cruce de un punto: con probabilidad P_CRUCE los padres intercambian colas."""
    if random.random() < P_CRUCE:
        corte = random.randint(1, L - 1) # Entero al azar entre 1 y L(longitud)-1
        return padre1[:corte] + padre2[corte:], padre2[:corte] + padre1[corte:]
    return padre1[:], padre2[:]  # sin cruce: los hijos son copias


def mutar(cromosoma):
    """Cada bit se invierte con probabilidad P_MUTACION."""
    return [1 - bit if random.random() < P_MUTACION else bit for bit in cromosoma]


# ALGORITMO
def poblacion_inicial():
    """Bits al azar; los individuos no factibles se descartan."""
    poblacion = []
    while len(poblacion) < TAM_POBLACION: # Hasta tener TAM_POBLACION individuos
        cromosoma = [random.randint(0, 1) for _ in range(L)] # Arreglo de L bits
        if evaluar(cromosoma)[2]: # Si el tercer valor de la tupla (factibilidad) es true
            poblacion.append(cromosoma)
    return poblacion


def algoritmo_genetico():
    poblacion = poblacion_inicial()
    inicial = [c[:] for c in poblacion] # Copia de la poblacion inicial (para citarla)
    historial = []  # Mejor individuo (cromosoma) de cada generación

    for gen in range(N_GENERACIONES):
        fits = [fitness(c) for c in poblacion] # Calcula el fitness de cada individuo
        elite = poblacion[fits.index(max(fits))] # Selecciona el indice del mejor
        historial.append(elite[:]) # Guarda el mejor individuo de esta generacion

        if gen == N_GENERACIONES - 1:
            break  # La última generación no necesita hijos (no se evaluaran)

        # Nueva población: el élite pasa intacto y el resto son hijos
        nueva = [elite[:]]
        while len(nueva) < TAM_POBLACION:
            # Ruleta: más fitness = más probabilidad de ser elegido
            padre1, padre2 = random.choices(poblacion, weights=fits, k=2) # Dos padres con probabilidad proporcional a su fitness
            for hijo in cruzar(padre1, padre2): # Devuelve dos hijos y el for procesa uno por uno
                hijo = mutar(hijo) # Se le aplica mutacion a cada hijo
                if evaluar(hijo)[2] and len(nueva) < TAM_POBLACION: # Si es factible y hay espacio
                    nueva.append(hijo)
        poblacion = nueva

    return elite, historial, inicial


# RESULTADOS: TABLAS Y GRAFICA
def guardar_tablas(historial, carpeta):
    """Guarda la tabla completa (todas las iteraciones) y la de cambios de fitness."""
    filas = []
    for i, c in enumerate(historial, 1):
        galones, costo, _ = evaluar(c)
        filas.append([i, texto(c), decodificar(c), galones, costo])

    # Cambios: primera iteracion, cada vez que el fitness cambia y la ultima
    cambios = [f for k, f in enumerate(filas)
               if k in (0, len(filas) - 1) or f[3] != filas[k - 1][3]]

    cabecera = ["Iteracion", "Genotipo", "Fenotipo", "Fitness", "Costo"]
    rutas = []
    for nombre, datos in (("tabla_iteraciones.csv", filas), ("tabla_cambios.csv", cambios)):
        ruta = carpeta / nombre
        with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
            escritor = csv.writer(f)
            escritor.writerow(cabecera)
            escritor.writerows(datos)
        rutas.append(ruta)
    return rutas


def graficar(historial, carpeta):
    """Guarda la evolución del fitness del mejor individuo por iteración."""
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(historial) + 1), [fitness(c) for c in historial],
             marker="o", markersize=3)
    plt.xlabel("Iteración")
    plt.ylabel("Fitness (galones)")
    plt.title("Fitness del mejor individuo por iteración")
    plt.grid(alpha=0.3)
    ruta = carpeta / "fitness_por_iteracion.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    return ruta


def optimo_real():
    """Prueba los 2^L cromosomas posibles (solo para comparar con el resultado)."""
    return max(fitness([int(b) for b in format(n, f"0{L}b")]) for n in range(2 ** L))


# EJECUCIÓN
if __name__ == "__main__":
    random.seed(SEMILLA)
    carpeta = Path(__file__).resolve().parent
    mejor, historial, inicial = algoritmo_genetico()
    rutas = guardar_tablas(historial, carpeta)
    ruta_grafica = graficar(historial, carpeta)
    galones, costo, _ = evaluar(mejor)

    print(f"Semilla: {SEMILLA} | Poblacion: {TAM_POBLACION} | Generaciones: {N_GENERACIONES}")
    print("\nPoblacion inicial:")
    for i, c in enumerate(inicial, 1):
        print(f"  {i:2d}. {texto(c)} -> {decodificar(c)} | "
              f"Fitness: {fitness(c)} | Costo: {evaluar(c)[1]}")

    print("\nCromosoma solucion:", texto(mejor))
    for j, cantidad in enumerate(decodificar(mejor)):
        galones_zona = cantidad * ZONAS[j][0]
        costo_zona = cantidad * ZONAS[j][1]
        print(
            f"Zona {j + 1}: {cantidad} de {ZONAS[j][2]} pozos | "
            f"Galones: {galones_zona} | Costo: {costo_zona}"
        )
    print(f"\nCosto total: {costo} (presupuesto {PRESUPUESTO})")
    optimo = optimo_real()
    print(f"Produccion del cromosoma: {galones} galones")
    print(f"\nOptimo exacto de produccion: {optimo} galones")
    print(f"¿El cromosoma encontro el optimo?: {'Si' if galones == optimo else 'No'}")
    for ruta in rutas + [ruta_grafica]:
        print(f"Guardado: {ruta}")