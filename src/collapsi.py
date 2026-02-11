import random

FILAS = 6
COLUMNAS = 6
COLORES = [1, 2, 3, 4]

def crear_tablero():
    return [[random.choice(COLORES) for _ in range(COLUMNAS)] for _ in range(FILAS)]

def mostrar_tablero(tablero):
    print("\nTablero:")
    for fila in tablero:
        for valor in fila:
            print(valor if valor != 0 else ".", end=" ")
        print()
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

def hay_movimientos(tablero):
    for f in range(FILAS):
        for c in range(COLUMNAS):
            if tablero[f][c] != 0:
                grupo = buscar_grupo(tablero, f, c, tablero[f][c], set())
                if len(grupo) > 1:
                    return True
    return False

def main():
    tablero = crear_tablero()

    while True:
        mostrar_tablero(tablero)

        if not hay_movimientos(tablero):
            print("Juego terminado. No hay más movimientos.")
            break

        try:
            f = int(input("Fila (0-5): "))
            c = int(input("Columna (0-5): "))
        except ValueError:
            print("Entrada inválida.")
            continue

        if f < 0 or f >= FILAS or c < 0 or c >= COLUMNAS:
            print("Posición fuera del tablero.")
            continue

        valor = tablero[f][c]
        if valor == 0:
            print("Espacio vacío.")
            continue

        grupo = buscar_grupo(tablero, f, c, valor, set())

        if len(grupo) < 2:
            print("No hay bloques suficientes para eliminar.")
            continue

        eliminar_grupo(tablero, grupo)
        colapsar_tablero(tablero)

if __name__ == "__main__":
    main()
