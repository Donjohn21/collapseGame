import random
import time
import copy
from typing import Callable, List

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

FILAS = 6
COLUMNAS = 6
COLORES = [1, 2, 3, 4]

NUM_MATCHES = 5
TIME_LIMITS = [1, 3, 10]
MAX_DEPTH = 10

# ==============================
# UTILIDADES DEL JUEGO
# ==============================

def crear_tablero():
    return [[random.choice(COLORES) for _ in range(COLUMNAS)] for _ in range(FILAS)]

def mostrar_tablero(tablero):
    for fila in tablero:
        print(" ".join(str(v) if v != 0 else "." for v in fila))
    print()

def buscar_grupo(tablero, f, c, valor, visitados):
    if f < 0 or f >= FILAS or c < 0 or c >= COLUMNAS:
        return []
    if (f, c) in visitados or tablero[f][c] != valor:
        return []

    visitados.add((f, c))
    grupo = [(f, c)]
    grupo += buscar_grupo(tablero, f+1, c, valor, visitados)
    grupo += buscar_grupo(tablero, f-1, c, valor, visitados)
    grupo += buscar_grupo(tablero, f, c+1, valor, visitados)
    grupo += buscar_grupo(tablero, f, c-1, valor, visitados)
    return grupo

def eliminar_grupo(tablero, grupo):
    for f, c in grupo:
        tablero[f][c] = 0

def colapsar_tablero(tablero):
    for c in range(COLUMNAS):
        columna = [tablero[f][c] for f in range(FILAS) if tablero[f][c] != 0]
        while len(columna) < FILAS:
            columna.insert(0, 0)
        for f in range(FILAS):
            tablero[f][c] = columna[f]

def obtener_movimientos(tablero):
    movimientos = []
    visitados = set()

    for f in range(FILAS):
        for c in range(COLUMNAS):
            if tablero[f][c] != 0 and (f, c) not in visitados:
                grupo = buscar_grupo(tablero, f, c, tablero[f][c], set())
                for pos in grupo:
                    visitados.add(pos)
                if len(grupo) > 1:
                    movimientos.append(grupo)
    return movimientos

def hay_movimientos(tablero):
    return len(obtener_movimientos(tablero)) > 0

# ==============================
# HEURÍSTICAS
# ==============================

def heuristica1(tablero):
    return sum(len(g) for g in obtener_movimientos(tablero))

def heuristica2(tablero):
    return len(obtener_movimientos(tablero))

def heuristica3(tablero):
    vacios = sum(fila.count(0) for fila in tablero)
    return -vacios

def heuristica4(tablero):
    movimientos = obtener_movimientos(tablero)
    return max((len(g) for g in movimientos), default=0)

def heuristica5(tablero):
    colores = {v for fila in tablero for v in fila if v != 0}
    return -len(colores)

TODAS_HEURISTICAS = [
    heuristica1,
    heuristica2,
    heuristica3,
    heuristica4,
    heuristica5
]

def evaluar(tablero, pesos, num_heur):
    valores = [h(tablero) for h in TODAS_HEURISTICAS[:num_heur]]
    pesos = pesos[:num_heur]
    return sum(p*h for p, h in zip(pesos, valores))

# ==============================
# IA MINIMAX + ALPHA-BETA + IDS
# ==============================

class IA:
    def __init__(self, tiempo_max=3, pesos=None, num_heur=5):
        self.tiempo_max = tiempo_max
        self.pesos = pesos or [1,1,1,1,1]
        self.num_heur = num_heur
        self.reset_metricas()

    def reset_metricas(self):
        self.nodos = 0
        self.profundidad_max = 0
        self.tiempo = 0

    def minimax(self, tablero, profundidad, prof_actual, alpha, beta, maximizando, inicio):

        if time.time() - inicio > self.tiempo_max:
            return evaluar(tablero, self.pesos, self.num_heur), None

        self.nodos += 1
        self.profundidad_max = max(self.profundidad_max, prof_actual)

        movimientos = obtener_movimientos(tablero)

        if profundidad == 0 or not movimientos:
            return evaluar(tablero, self.pesos, self.num_heur), None

        mejor_mov = None

        if maximizando:
            max_eval = float('-inf')
            for mov in movimientos:
                nuevo = copy.deepcopy(tablero)
                eliminar_grupo(nuevo, mov)
                colapsar_tablero(nuevo)

                val, _ = self.minimax(
                    nuevo,
                    profundidad-1,
                    prof_actual+1,
                    alpha,
                    beta,
                    False,
                    inicio
                )

                if val > max_eval:
                    max_eval = val
                    mejor_mov = mov

                alpha = max(alpha, val)
                if beta <= alpha:
                    break

            return max_eval, mejor_mov

        else:
            min_eval = float('inf')
            for mov in movimientos:
                nuevo = copy.deepcopy(tablero)
                eliminar_grupo(nuevo, mov)
                colapsar_tablero(nuevo)

                val, _ = self.minimax(
                    nuevo,
                    profundidad-1,
                    prof_actual+1,
                    alpha,
                    beta,
                    True,
                    inicio
                )

                if val < min_eval:
                    min_eval = val
                    mejor_mov = mov

                beta = min(beta, val)
                if beta <= alpha:
                    break

            return min_eval, mejor_mov

    def mejor_movimiento(self, tablero):
        self.reset_metricas()
        inicio = time.time()
        mejor = None
        profundidad = 1

        while time.time() - inicio < self.tiempo_max and profundidad <= MAX_DEPTH:
            _, mov = self.minimax(
                tablero,
                profundidad,
                0,
                float('-inf'),
                float('inf'),
                True,
                inicio
            )
            if mov:
                mejor = mov
            profundidad += 1

        self.tiempo = time.time() - inicio
        return mejor

# ==============================
# JUGADORES
# ==============================

def jugador_humano(tablero):
    mostrar_tablero(tablero)
    try:
        f = int(input("Fila: "))
        c = int(input("Columna: "))
    except:
        return None

    if 0 <= f < FILAS and 0 <= c < COLUMNAS and tablero[f][c] != 0:
        grupo = buscar_grupo(tablero, f, c, tablero[f][c], set())
        if len(grupo) > 1:
            return grupo
    return None

def jugador_random(tablero):
    movs = obtener_movimientos(tablero)
    return random.choice(movs) if movs else None

def jugador_greedy(tablero):
    movs = obtener_movimientos(tablero)
    return max(movs, key=len) if movs else None

def jugador_peor(tablero):
    movs = obtener_movimientos(tablero)
    return min(movs, key=len) if movs else None

# ==============================
# PARTIDA
# ==============================

def jugar_partida_jugadores(jugadores: List[Callable]):
    tablero = crear_tablero()
    n = len(jugadores)
    puntos = [0]*n
    turno = 0

    while hay_movimientos(tablero):
        jugador = jugadores[turno % n]
        mov = jugador(tablero)

        if not mov:
            turno += 1
            continue

        puntos[turno % n] += len(mov)**2
        eliminar_grupo(tablero, mov)
        colapsar_tablero(tablero)
        turno += 1

    max_p = max(puntos)
    ganadores = [i for i,p in enumerate(puntos) if p == max_p]
    return puntos, ganadores

def jugar_varias_partidas(jugadores, cantidad):
    resultados = []
    for _ in range(cantidad):
        puntos, ganadores = jugar_partida_jugadores(jugadores)
        resultados.append((puntos, ganadores))
    return resultados

# ==============================
# BARRA PROGRESO
# ==============================

def barra_progreso(actual, total):
    ancho = 30
    llenos = int(ancho * actual / total)
    barra = "█"*llenos + "-"*(ancho-llenos)
    print(f"\rProgreso |{barra}| {actual}/{total}", end="")

# ==============================
# BENCHMARK AUTOMÁTICO + ANÁLISIS
# ==============================

def ejecutar_benchmark():

    incluir_humano = input("¿Incluir humano en benchmark? (s/n): ").lower() == "s"

    configuraciones_pesos = [
        [1,1,1,1,1],
        [2,1,1,2,1]
    ]

    heuristicas = [1,2,3,4,5]

    oponentes = {
        "random": jugador_random,
        "greedy": jugador_greedy,
        "peor": jugador_peor,
        "ia_igual": "ia_igual"
    }

    if incluir_humano:
        oponentes["humano"] = jugador_humano

    total_tests = (len(configuraciones_pesos) *
                   len(heuristicas) *
                   len(TIME_LIMITS) *
                   len(oponentes) *
                   NUM_MATCHES)

    progreso = 0
    resultados = []

    print("\n===== INICIANDO BENCHMARK =====\n")

    for pesos in configuraciones_pesos:
        for h in heuristicas:
            for t in TIME_LIMITS:
                for nombre_op, op in oponentes.items():
                    for _ in range(NUM_MATCHES):

                        ia = IA(tiempo_max=t, pesos=pesos, num_heur=h)

                        def jia(tab):
                            return ia.mejor_movimiento(tab)

                        if nombre_op == "ia_igual":
                            ia2 = IA(tiempo_max=t, pesos=pesos, num_heur=h)
                            def jia2(tab):
                                return ia2.mejor_movimiento(tab)
                            puntos, ganadores = jugar_partida_jugadores([jia, jia2])
                        else:
                            puntos, ganadores = jugar_partida_jugadores([jia, op])

                        resultados.append({
                            "op": nombre_op,
                            "pesos": tuple(pesos),
                            "heur": h,
                            "t": t,
                            "p_ia": puntos[0],
                            "gan_ia": 0 in ganadores,
                            "tiempo": ia.tiempo,
                            "nodos": ia.nodos,
                            "prof": ia.profundidad_max
                        })

                        progreso += 1
                        barra_progreso(progreso, total_tests)

    print("\n\n===== FIN BENCHMARK =====")
    print("Total partidas:", len(resultados))

    # ==============================
    # ANÁLISIS ESTRUCTURADO
    # ==============================

    print("\n\n===== RESUMEN ESTRUCTURADO =====\n")

    resumen = {}

    for r in resultados:
        clave = (r["op"], r["pesos"], r["heur"], r["t"])

        if clave not in resumen:
            resumen[clave] = {
                "wins": 0,
                "points": 0,
                "nodos": 0,
                "prof": 0,
                "tiempo": 0,
                "partidas": 0
            }

        resumen[clave]["wins"] += 1 if r["gan_ia"] else 0
        resumen[clave]["points"] += r["p_ia"]
        resumen[clave]["nodos"] += r["nodos"]
        resumen[clave]["prof"] += r["prof"]
        resumen[clave]["tiempo"] += r["tiempo"]
        resumen[clave]["partidas"] += 1

    for clave, datos in sorted(resumen.items()):
        op, pesos, heur, t = clave
        partidas = datos["partidas"]

        print(f"Contra: {op}")
        print(f"Pesos: {pesos} | Heurísticas: {heur} | Tiempo: {t}s")
        print(f"Victorias IA: {datos['wins']} / {partidas}")
        print(f"Promedio puntos: {datos['points']/partidas:.2f}")
        print(f"Promedio nodos: {datos['nodos']/partidas:.2f}")
        print(f"Profundidad promedio: {datos['prof']/partidas:.2f}")
        print(f"Tiempo promedio: {datos['tiempo']/partidas:.4f}s")
        print("-"*60)

# ==============================
# CONFIGURACIÓN JUGADORES
# ==============================

def seleccionar_jugador(idx):

    print(f"\nJugador {idx+1}:")
    print("1. Humano")
    print("2. IA")
    print("3. Random")
    print("4. Greedy")
    print("5. Peor")

    op = input("Opción: ")

    if op == "1":
        return jugador_humano

    if op == "2":
        t = int(input("Tiempo IA: "))
        ia = IA(tiempo_max=t)
        return lambda tab: ia.mejor_movimiento(tab)

    if op == "3":
        return jugador_random

    if op == "4":
        return jugador_greedy

    if op == "5":
        return jugador_peor

    return jugador_random

# ==============================
# MAIN
# ==============================

def main():

    print("1. Jugar partida")
    print("2. Ejecutar benchmark automático")

    modo = input("Modo: ")

    if modo == "1":

        n = int(input("Cantidad de jugadores: "))
        partidas = int(input("Cantidad de partidas: "))

        jugadores = []
        for i in range(n):
            jugadores.append(seleccionar_jugador(i))

        resultados = jugar_varias_partidas(jugadores, partidas)

        for i, (puntos, ganadores) in enumerate(resultados, 1):
            print(f"\nPartida {i}")
            print("Puntos:", puntos)
            print("Ganadores:", ganadores)

    elif modo == "2":
        ejecutar_benchmark()

if __name__ == "__main__":
    main()