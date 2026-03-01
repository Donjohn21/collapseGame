import random
import time
import math
from typing import List, Tuple

# =====================================
# CONFIGURACION GLOBAL
# =====================================
SIZE = 4
JOKER = "J"
COLLAPSED = "X"
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

MAX_MOV = 12
MAX_CELDAS = 16
MAX_DIST = 6

RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

# =====================================
# TABLERO
# =====================================

def crear_tablero(num_jugadores=2):
    cartas = [JOKER]*num_jugadores + [1]*4 + [2]*4 + [3]*4 + [4]*2
    while len(cartas) < SIZE*SIZE:
        cartas.append(1)
    random.shuffle(cartas)
    return [cartas[i*SIZE:(i+1)*SIZE] for i in range(SIZE)]


def encontrar_jokers(tab):
    pos = []
    for r in range(SIZE):
        for c in range(SIZE):
            if tab[r][c] == JOKER:
                pos.append((r,c))
    return pos


def imprimir_tablero(tab, p1=None, p2=None):
    for r in range(SIZE):
        fila = []
        for c in range(SIZE):
            if p1 is not None and (r,c) == p1:
                fila.append(RED+"A"+RESET)
            elif p2 is not None and (r,c) == p2:
                fila.append(BLUE+"B"+RESET)
            else:
                fila.append(str(tab[r][c]))
        print(" ".join(fila))
    print()

# =====================================
# TOROIDAL
# =====================================

def wrap(r,c):
    return r % SIZE, c % SIZE

# =====================================
# MOVIMIENTOS
# =====================================

def dfs_movimientos(tab, start, steps, rival):
    stack = [(start[0], start[1], steps, {start})]
    res = set()
    while stack:
        r,c,s,vis = stack.pop()
        if s == 0:
            if (r,c)!=start and (r,c)!=rival:
                res.add((r,c))
            continue
        for dr,dc in DIRS:
            nr,nc = wrap(r+dr,c+dc)
            if (nr,nc) in vis: continue
            if tab[nr][nc] == COLLAPSED: continue
            stack.append((nr,nc,s-1, vis | {(nr,nc)}))
    return res


def pasos_casilla(tab,pos,primer_turno):
    if primer_turno: return 1
    v = tab[pos[0]][pos[1]]
    if isinstance(v,int): return v
    return 1


def movimientos_validos(tab,pos,rival,primer_turno=False):
    steps = pasos_casilla(tab,pos,primer_turno)
    return dfs_movimientos(tab,pos,steps,rival)

# =====================================
# COLAPSO
# =====================================

def colapsar(tab,pos):
    tab[pos[0]][pos[1]] = COLLAPSED

# =====================================
# HEURISTICAS
# =====================================

def contar_mov(tab,pos,rival):
    return len(movimientos_validos(tab,pos,rival,False))


def celdas_restantes(tab):
    return sum(1 for r in tab for v in r if v!=COLLAPSED)


def dist_toroidal(a,b):
    dr = min(abs(a[0]-b[0]), SIZE-abs(a[0]-b[0]))
    dc = min(abs(a[1]-b[1]), SIZE-abs(a[1]-b[1]))
    return dr+dc


def heuristicas(tab,yo,rival):
    mov_yo = contar_mov(tab,yo,rival)
    mov_rival = contar_mov(tab,rival,yo)
    h1 = mov_yo / MAX_MOV
    h2 = mov_rival / MAX_MOV
    h3 = (mov_yo - mov_rival) / MAX_MOV
    h4 = celdas_restantes(tab) / MAX_CELDAS
    h5 = 1 - (dist_toroidal(yo,rival) / MAX_DIST)
    return [h1, -h2, h3, h4, h5]


def evaluar(tab,yo,rival,pesos,usar_n):
    hs = heuristicas(tab,yo,rival)[:usar_n]
    ws = pesos[:usar_n]
    return sum(w*h for w,h in zip(ws,hs))

# =====================================
# MINIMAX + ALPHA BETA + IDS
# =====================================

class Stats:
    def __init__(self):
        self.nodos = 0
        self.profundidad = 0
        self.tiempo = 0


def minimax(tab,yo,rival,prof,alpha,beta,maxim,stats,pesos,usar_n,fin_t):
    if time.time()>=fin_t:
        raise TimeoutError

    stats.nodos += 1
    stats.profundidad = max(stats.profundidad, prof)

    movs = movimientos_validos(tab,yo,rival)

    if prof==0 or not movs:
        if not movs: return -1
        return evaluar(tab,yo,rival,pesos,usar_n)

    if maxim:
        val=-math.inf
        for m in movs:
            ntab=[r[:] for r in tab]
            colapsar(ntab,yo)
            val=max(val,minimax(ntab,rival,m,prof-1,alpha,beta,False,stats,pesos,usar_n,fin_t))
            alpha=max(alpha,val)
            if beta<=alpha: break
        return val
    else:
        val=math.inf
        for m in movs:
            ntab=[r[:] for r in tab]
            colapsar(ntab,yo)
            val=min(val,minimax(ntab,rival,m,prof-1,alpha,beta,True,stats,pesos,usar_n,fin_t))
            beta=min(beta,val)
            if beta<=alpha: break
        return val


def mejor_movimiento_ids(tab,yo,rival,max_time,pesos,usar_n):
    inicio=time.time()
    fin=inicio+max_time
    mejor=None
    prof=1
    stats=Stats()

    while True:
        try:
            movs=movimientos_validos(tab,yo,rival)
            best_val=-math.inf
            best_move=None
            for m in movs:
                ntab=[r[:] for r in tab]
                colapsar(ntab,yo)
                val=minimax(ntab,rival,m,prof-1,-math.inf,math.inf,False,
                            stats,pesos,usar_n,fin)
                if val>best_val:
                    best_val=val
                    best_move=m
            if time.time()<fin:
                mejor=best_move
                prof+=1
            else:
                break
        except TimeoutError:
            break

    stats.tiempo=time.time()-inicio
    return mejor,stats,prof-1

# =====================================
# JUGADORES
# =====================================

def jugador_random(tab,pos,rival):
    movs=list(movimientos_validos(tab,pos,rival))
    return random.choice(movs) if movs else None


def jugador_greedy(tab,pos,rival,pesos,usar_n):
    movs=movimientos_validos(tab,pos,rival)
    best=None; bestv=-1e9
    for m in movs:
        ntab=[r[:] for r in tab]
        colapsar(ntab,pos)
        v=evaluar(ntab,m,rival,pesos,usar_n)
        if v>bestv: bestv=v; best=m
    return best


def jugador_peor(tab,pos,rival,pesos,usar_n):
    movs=movimientos_validos(tab,pos,rival)
    worst=None; worstv=1e9
    for m in movs:
        ntab=[r[:] for r in tab]
        colapsar(ntab,pos)
        v=evaluar(ntab,m,rival,pesos,usar_n)
        if v<worstv: worstv=v; worst=m
    return worst

# =====================================
# PARTIDA ORIGINAL (2 JUGADORES)
# =====================================

def jugar_partida(tipoA,tipoB,max_time,pesos,usar_n,mostrar=True):

    tab=crear_tablero(2)
    p1,p2=encontrar_jokers(tab)

    turnoA=True
    primerA=True
    primerB=True

    statsA=Stats()
    statsB=Stats()

    while True:

        if mostrar:
            imprimir_tablero(tab,p1,p2)

        if turnoA:

            movs=movimientos_validos(tab,p1,p2,primerA)
            if not movs:
                return "B",statsA,statsB,celdas_restantes(tab)

            if tipoA in ("humano","humano_experto"):
                print("Movimientos:",movs)
                m=eval(input("Movimiento: "))

            elif tipoA=="random":
                m=jugador_random(tab,p1,p2)

            elif tipoA=="greedy":
                m=jugador_greedy(tab,p1,p2,pesos,usar_n)

            elif tipoA=="peor":
                m=jugador_peor(tab,p1,p2,pesos,usar_n)

            else:
                m,s,_=mejor_movimiento_ids(tab,p1,p2,max_time,pesos,usar_n)
                statsA.nodos+=s.nodos
                statsA.profundidad=max(statsA.profundidad,s.profundidad)
                statsA.tiempo+=s.tiempo

            colapsar(tab,p1)
            p1=m
            primerA=False

        else:

            movs=movimientos_validos(tab,p2,p1,primerB)
            if not movs:
                return "A",statsA,statsB,celdas_restantes(tab)

            if tipoB in ("humano","humano_experto"):
                print("Movimientos:",movs)
                m=eval(input("Movimiento: "))

            elif tipoB=="random":
                m=jugador_random(tab,p2,p1)

            elif tipoB=="greedy":
                m=jugador_greedy(tab,p2,p1,pesos,usar_n)

            elif tipoB=="peor":
                m=jugador_peor(tab,p2,p1,pesos,usar_n)

            else:
                m,s,_=mejor_movimiento_ids(tab,p2,p1,max_time,pesos,usar_n)
                statsB.nodos+=s.nodos
                statsB.profundidad=max(statsB.profundidad,s.profundidad)
                statsB.tiempo+=s.tiempo

            colapsar(tab,p2)
            p2=m
            primerB=False

        turnoA=not turnoA

# =====================================
# PARTIDA MULTI-JUGADOR
# =====================================

def jugar_partida_multi(tipos, max_time, pesos, usar_n, mostrar=True):

    n = len(tipos)

    tab = crear_tablero(n)
    posiciones = encontrar_jokers(tab)

    primeros = [True]*n
    stats = [Stats() for _ in range(n)]

    turno = 0

    while True:

        yo = turno
        rival = (turno+1) % n

        if mostrar:
            print("Turno jugador",yo)

        pos_yo = posiciones[yo]
        pos_rival = posiciones[rival]

        movs = movimientos_validos(tab,pos_yo,pos_rival,primeros[yo])

        if not movs:
            ganador = (yo-1) % n
            return ganador,stats,celdas_restantes(tab)

        tipo = tipos[yo]

        if tipo in ("humano","humano_experto"):
            print("Movimientos:",movs)
            m = eval(input("Movimiento: "))

        elif tipo=="random":
            m = jugador_random(tab,pos_yo,pos_rival)

        elif tipo=="greedy":
            m = jugador_greedy(tab,pos_yo,pos_rival,pesos,usar_n)

        elif tipo=="peor":
            m = jugador_peor(tab,pos_yo,pos_rival,pesos,usar_n)

        else:
            if n==2:
                m,s,_ = mejor_movimiento_ids(tab,pos_yo,pos_rival,max_time,pesos,usar_n)
                stats[yo].nodos+=s.nodos
                stats[yo].profundidad=max(stats[yo].profundidad,s.profundidad)
                stats[yo].tiempo+=s.tiempo
            else:
                m = jugador_random(tab,pos_yo,pos_rival)

        colapsar(tab,pos_yo)
        posiciones[yo] = m
        primeros[yo] = False

        turno = (turno+1) % n

# =====================================
# BENCHMARK
# =====================================

def benchmark():

    configs_pesos=[("pesos1",[0.4,0.3,0.5,0.2,0.2]),
                   ("pesos2",[0.6,0.4,0.6,0.3,0.3])]

    tiempos=[1,3,10]
    rivales=["random","greedy","peor","minimax"]
    heuristicas_n=[1,2,3,4,5]

    print("config heur tiempo rival ganador puntos nodos prof tiempo")

    for nombre,pesos in configs_pesos:
        for n in heuristicas_n:
            for t in tiempos:
                for r in rivales:
                    g,sA,sB,puntos=jugar_partida("minimax",r,t,pesos,n,False)
                    print(nombre,n,t,r,g,puntos,
                          sA.nodos,sA.profundidad,round(sA.tiempo,3))

# =====================================
# MENU
# =====================================

def menu():

    while True:

        print("\n1. Partida 2 jugadores (clasica)")
        print("2. Partida con N jugadores")
        print("3. Benchmark")
        print("4. Salir")

        op=input("> ")

        if op=="1":

            jugar_partida(
                "humano",
                "minimax",
                3,
                [0.4,0.3,0.5,0.2,0.2],
                5,
                True
            )

        elif op=="2":

            n=int(input("Cantidad de jugadores: "))

            tipos=[]
            for i in range(n):
                print(f"Jugador {i} tipo (humano, humano_experto, minimax, random, greedy, peor):")
                tipos.append(input("> "))

            ganador,stats,puntos = jugar_partida_multi(
                tipos,
                3,
                [0.4,0.3,0.5,0.2,0.2],
                5,
                True
            )

            print("Ganador:",ganador,"Puntos:",puntos)

        elif op=="3":
            benchmark()

        else:
            break


if __name__=="__main__":
    menu()