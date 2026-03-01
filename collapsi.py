from __future__ import annotations
import random
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

# =========================
# CONFIG
# =========================
SIZE = 4
JOKER = "J"
COLLAPSED = "X"

DIR_MAP = {
    "ARRIBA": (-1, 0),
    "ABAJO": (1, 0),
    "IZQUIERDA": (0, -1),
    "DERECHA": (0, 1),
}

# Máximos teóricos (para normalización)
MAX_OPEN = 16
MAX_REACH = 16
MAX_MOB = 14
MAX_NEI = 4
MAX_JOK = 1

WIN_SCORE = 16

# =========================
# COLORES (ANSI)
# =========================
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"

ANSI_RED = "\033[31m"
ANSI_BLUE = "\033[34m"
ANSI_YELLOW = "\033[33m"
ANSI_GRAY = "\033[90m"

def supports_ansi() -> bool:
    # En Windows moderno funciona; en algunos entornos puede no, pero si no, solo “no colorea”.
    # No intentamos detectar terminal avanzado para no complicar.
    return True

USE_COLOR = supports_ansi()

def colorize(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{ANSI_RESET}"

def player_label(player: int) -> str:
    return "A" if player == 0 else "B"

def colored_player(player: int) -> str:
    if player == 0:
        return colorize("A", ANSI_RED + ANSI_BOLD)
    return colorize("B", ANSI_BLUE + ANSI_BOLD)

# =========================
# TIPOS
# =========================
Pos = Tuple[int, int]
Board = List[List[object]]  # object para mezclar int / str

PLAYER_A = 0
PLAYER_B = 1


class PlayerKind:
    HUMAN = "human"
    RANDOM = "random"
    GREEDY = "greedy"
    WORST = "worst"
    MINIMAX = "minimax"


@dataclass(frozen=True)
class Move:
    """Camino completo: lista de posiciones visitadas incluyendo el final, NO incluye la inicial."""
    steps: Tuple[Pos, ...]
    k: int  # cantidad de pasos


@dataclass
class GameState:
    board: Board
    pos: Tuple[Pos, Pos]           # posiciones A, B
    to_move: int                   # 0 o 1
    moves_made: Tuple[int, int]    # para regla: primer turno de cada jugador = 1 paso fijo

    def clone(self) -> "GameState":
        new_board = [row[:] for row in self.board]
        return GameState(
            board=new_board,
            pos=(self.pos[0], self.pos[1]),
            to_move=self.to_move,
            moves_made=(self.moves_made[0], self.moves_made[1]),
        )


@dataclass
class SearchStats:
    nodes: int = 0
    max_depth: int = 0
    cutoffs: int = 0
    elapsed: float = 0.0


# =========================
# TABLERO / DISPLAY
# =========================
def crear_tablero() -> Board:
    cartas: List[object] = (
        [JOKER] * 2
        + [1] * 4
        + [2] * 4
        + [3] * 4
        + [4] * 2
    )
    assert len(cartas) == 16
    random.shuffle(cartas)
    return [cartas[i * SIZE:(i + 1) * SIZE] for i in range(SIZE)]


def encontrar_jokers(tab: Board) -> List[Pos]:
    jokers: List[Pos] = []
    for r in range(SIZE):
        for c in range(SIZE):
            if tab[r][c] == JOKER:
                jokers.append((r, c))
    return jokers


def imprimir_tablero(tab: Board, pA: Pos, pB: Pos) -> None:
    print("\nTABLERO:")
    header = "   " + " ".join(str(c) for c in range(SIZE))
    print(header)

    for r in range(SIZE):
        fila = []
        for c in range(SIZE):
            if (r, c) == pA:
                fila.append(colored_player(PLAYER_A))
            elif (r, c) == pB:
                fila.append(colored_player(PLAYER_B))
            else:
                v = tab[r][c]
                if v == COLLAPSED:
                    fila.append(colorize("X", ANSI_GRAY))
                elif v == JOKER:
                    fila.append(colorize("J", ANSI_YELLOW))
                else:
                    fila.append(str(v))
        print(f"{r}  " + " ".join(fila))
    print()


# =========================
# TORUS HELPERS
# =========================
def wrap_pos(r: int, c: int) -> Pos:
    return (r % SIZE, c % SIZE)


def add_dir(p: Pos, d: Tuple[int, int]) -> Pos:
    return wrap_pos(p[0] + d[0], p[1] + d[1])


# =========================
# REGLAS: k requerido
# =========================
def posibles_k(state: GameState, player: int) -> List[int]:
    """Regla de 'cuántos pasos debe/puede mover'."""
    # primer turno de ese jugador: 1 fijo
    if state.moves_made[player] == 0:
        return [1]

    r, c = state.pos[player]
    cell = state.board[r][c]
    if cell == JOKER:
        return [1, 2, 3, 4]
    if cell == COLLAPSED:
        return []
    return [int(cell)]


# =========================
# GENERACIÓN DE MOVIMIENTOS VÁLIDOS
# =========================
def generar_movimientos_validos(state: GameState, player: int) -> List[Move]:
    """Devuelve todos los movimientos legales posibles desde la posición actual."""
    ks = posibles_k(state, player)
    if not ks:
        return []

    start = state.pos[player]
    opp = state.pos[1 - player]

    moves: List[Move] = []

    def dfs(curr: Pos, remaining: int, visited: Set[Pos], path: List[Pos], k: int) -> None:
        if remaining == 0:
            end = curr
            if end != start and end != opp:
                moves.append(Move(steps=tuple(path), k=k))
            return

        for d in DIR_MAP.values():
            nxt = add_dir(curr, d)
            if state.board[nxt[0]][nxt[1]] == COLLAPSED:
                continue
            if nxt in visited:
                continue

            visited.add(nxt)
            path.append(nxt)
            dfs(nxt, remaining - 1, visited, path, k)
            path.pop()
            visited.remove(nxt)

    for k in ks:
        visited = {start}
        dfs(start, k, visited, [], k)

    return moves


def hay_movimientos_posibles(state: GameState, player: int) -> bool:
    return len(generar_movimientos_validos(state, player)) > 0


# =========================
# APLICAR MOVIMIENTO
# =========================
def aplicar_movimiento(state: GameState, move: Move) -> GameState:
    """Aplica movimiento YA validado: colapsa inicio, mueve ficha, cambia turno."""
    new_state = state.clone()
    p = state.to_move
    start = state.pos[p]
    end = move.steps[-1]

    sr, sc = start
    new_state.board[sr][sc] = COLLAPSED

    posA, posB = new_state.pos
    if p == PLAYER_A:
        posA = end
    else:
        posB = end
    new_state.pos = (posA, posB)

    mm = list(new_state.moves_made)
    mm[p] += 1
    new_state.moves_made = (mm[0], mm[1])

    new_state.to_move = 1 - p
    return new_state


# =========================
# PUNTOS / RONDAS
# =========================
def contar_cartas_boca_arriba(tab: Board) -> int:
    return sum(1 for r in range(SIZE) for c in range(SIZE) if tab[r][c] != COLLAPSED)


def inicializar_ronda(starting_player: int) -> GameState:
    tab = crear_tablero()
    jokers = encontrar_jokers(tab)
    if len(jokers) != 2:
        raise RuntimeError("El tablero no tiene exactamente 2 Jokers.")
    p1 = jokers[0]
    p2 = jokers[1]
    return GameState(board=tab, pos=(p1, p2), to_move=starting_player, moves_made=(0, 0))


# =========================
# HEURÍSTICAS (≥ 5) NORMALIZADAS
# =========================
def vecinos_no_colapsados(tab: Board, p: Pos) -> int:
    cnt = 0
    for d in DIR_MAP.values():
        q = add_dir(p, d)
        if tab[q[0]][q[1]] != COLLAPSED:
            cnt += 1
    return cnt


def reachable_area(tab: Board, start: Pos, blocked: Set[Pos]) -> int:
    if tab[start[0]][start[1]] == COLLAPSED:
        return 0
    q = [start]
    seen = {start}
    while q:
        u = q.pop()
        for d in DIR_MAP.values():
            v = add_dir(u, d)
            if v in seen or v in blocked:
                continue
            if tab[v[0]][v[1]] == COLLAPSED:
                continue
            seen.add(v)
            q.append(v)
    return len(seen)


def esta_en_joker(tab: Board, p: Pos) -> int:
    return 1 if tab[p[0]][p[1]] == JOKER else 0


def evaluar_estado(state: GameState, max_player: int, weights: Tuple[float, float, float, float, float]) -> float:
    me = max_player
    opp = 1 - me

    my_moves = len(generar_movimientos_validos(state, me))
    opp_moves = len(generar_movimientos_validos(state, opp))

    my_area = reachable_area(state.board, state.pos[me], blocked={state.pos[opp]})
    opp_area = reachable_area(state.board, state.pos[opp], blocked={state.pos[me]})

    my_j = esta_en_joker(state.board, state.pos[me])
    opp_j = esta_en_joker(state.board, state.pos[opp])

    my_nei = vecinos_no_colapsados(state.board, state.pos[me])
    opp_nei = vecinos_no_colapsados(state.board, state.pos[opp])

    h1 = (my_moves - opp_moves) / MAX_MOB
    h2 = (my_area - opp_area) / MAX_REACH

    h3 = (MAX_MOB - opp_moves) / MAX_MOB
    h3 = 2 * h3 - 1

    h4 = (my_j - opp_j) / MAX_JOK
    h5 = (my_nei - opp_nei) / MAX_NEI

    w1, w2, w3, w4, w5 = weights
    denom = (abs(w1) + abs(w2) + abs(w3) + abs(w4) + abs(w5)) or 1.0
    score = (w1 * h1 + w2 * h2 + w3 * h3 + w4 * h4 + w5 * h5) / denom

    return max(-1.0, min(1.0, score))


# =========================
# MINIMAX (ALPHA-BETA + IDS)
# =========================
def terminal_value(state: GameState, max_player: int) -> Optional[float]:
    p = state.to_move
    if not hay_movimientos_posibles(state, p):
        winner = 1 - p
        return 1.0 if winner == max_player else -1.0
    return None


def minimax_ab(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    max_player: int,
    weights: Tuple[float, float, float, float, float],
    deadline: float,
    stats: "SearchStats",
) -> float:
    if time.perf_counter() >= deadline:
        raise TimeoutError

    stats.nodes += 1
    stats.max_depth = max(stats.max_depth, depth)

    tv = terminal_value(state, max_player)
    if tv is not None:
        return tv

    if depth == 0:
        return evaluar_estado(state, max_player, weights)

    p = state.to_move
    moves = generar_movimientos_validos(state, p)

    def move_key(m: Move) -> float:
        nxt = aplicar_movimiento(state, m)
        return evaluar_estado(nxt, max_player, weights)

    moves.sort(key=move_key, reverse=(p == max_player))

    if p == max_player:
        value = -math.inf
        for m in moves:
            child = aplicar_movimiento(state, m)
            v = minimax_ab(child, depth - 1, alpha, beta, max_player, weights, deadline, stats)
            value = max(value, v)
            alpha = max(alpha, value)
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return value
    else:
        value = math.inf
        for m in moves:
            child = aplicar_movimiento(state, m)
            v = minimax_ab(child, depth - 1, alpha, beta, max_player, weights, deadline, stats)
            value = min(value, v)
            beta = min(beta, value)
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return value


def elegir_movimiento_minimax(
    state: GameState,
    max_time_s: float,
    weights: Tuple[float, float, float, float, float],
) -> Tuple[Move, SearchStats]:
    start_t = time.perf_counter()
    deadline = start_t + max_time_s
    p = state.to_move
    moves = generar_movimientos_validos(state, p)
    if not moves:
        raise RuntimeError("No hay movimientos (debería haberse detectado antes).")

    best_move = random.choice(moves)
    best_stats = SearchStats()

    depth = 1
    while True:
        if time.perf_counter() >= deadline:
            break

        stats = SearchStats()
        local_best_move = best_move
        local_best_val = -math.inf

        def root_key(m: Move) -> float:
            nxt = aplicar_movimiento(state, m)
            return evaluar_estado(nxt, p, weights)

        root_moves = moves[:]
        root_moves.sort(key=root_key, reverse=True)

        try:
            for m in root_moves:
                if time.perf_counter() >= deadline:
                    raise TimeoutError
                child = aplicar_movimiento(state, m)
                v = minimax_ab(child, depth - 1, -math.inf, math.inf, p, weights, deadline, stats)
                if v > local_best_val:
                    local_best_val = v
                    local_best_move = m

            best_move = local_best_move
            best_stats = stats
            depth += 1

        except TimeoutError:
            break

    best_stats.elapsed = time.perf_counter() - start_t
    return best_move, best_stats


# =========================
# POLÍTICAS DE JUGADOR
# =========================
def parse_human_move_input(s: str) -> List[str]:
    """
    Permite ingresar direcciones en español separadas por espacios.
    Ejemplo:
    ARRIBA DERECHA ABAJO
    """
    s = s.strip().upper().replace(",", " ")
    parts = [p for p in s.split() if p]
    return parts

def pedir_movimiento_humano(state: GameState) -> Move:
    p = state.to_move
    ks = posibles_k(state, p)

    if ks == [1]:
        print("Debes mover EXACTAMENTE 1 paso (primer turno de este jugador).")
    else:
        r, c = state.pos[p]
        cell = state.board[r][c]
        if cell == JOKER:
            print("Estás en Joker. Elige k en {1,2,3,4} y el camino.")
        else:
            print(f"Debes mover EXACTAMENTE {int(cell)} pasos.")

    legal_moves = generar_movimientos_validos(state, p)
    legal_set = {m.steps for m in legal_moves}

    while True:
        if len(ks) > 1:
            try:
                k_in = int(input("Elige k (1-4): ").strip())
                if k_in not in ks:
                    print("k inválido.")
                    continue
            except ValueError:
                print("Entrada inválida.")
                continue
        else:
            k_in = ks[0]

        s = input("Camino (ej: ARRIBA DERECHA ABAJO): ").strip()
        dirs = parse_human_move_input(s)
        if len(dirs) != k_in:
            print(f"Debes ingresar exactamente {k_in} direcciones.")
            continue
        if any(d not in DIR_MAP for d in dirs):
            print("Usa solo: ARRIBA, ABAJO, IZQUIERDA, DERECHA.")
            continue

        start = state.pos[p]
        path: List[Pos] = []
        visited: Set[Pos] = {start}
        curr = start
        ok = True

        for d in dirs:
            curr = add_dir(curr, DIR_MAP[d])
            if state.board[curr[0]][curr[1]] == COLLAPSED:
                ok = False
                break
            if curr in visited:
                ok = False
                break
            visited.add(curr)
            path.append(curr)

        if not ok:
            print("Movimiento ilegal (X o repetición).")
            continue

        move = Move(steps=tuple(path), k=k_in)
        if move.steps not in legal_set:
            print("Movimiento ilegal (termina en inicio u oponente, o no existe como movimiento válido).")
            continue
        return move


def elegir_move_random(state: GameState) -> Move:
    moves = generar_movimientos_validos(state, state.to_move)
    return random.choice(moves)


def elegir_move_greedy(state: GameState, weights: Tuple[float, float, float, float, float]) -> Move:
    p = state.to_move
    moves = generar_movimientos_validos(state, p)
    best = moves[0]
    bestv = -math.inf
    for m in moves:
        child = aplicar_movimiento(state, m)
        v = evaluar_estado(child, p, weights)
        if v > bestv:
            bestv, best = v, m
    return best


def elegir_move_worst(state: GameState, weights: Tuple[float, float, float, float, float]) -> Move:
    p = state.to_move
    moves = generar_movimientos_validos(state, p)
    worst = moves[0]
    worstv = math.inf
    for m in moves:
        child = aplicar_movimiento(state, m)
        v = evaluar_estado(child, p, weights)
        if v < worstv:
            worstv, worst = v, m
    return worst


# =========================
# PARTIDA (RONDAS HASTA 16)
# =========================
@dataclass
class PlayerConfig:
    kind: str
    max_time: float = 3.0
    weights: Tuple[float, float, float, float, float] = (1, 1, 1, 1, 1)


@dataclass
class MatchReport:
    winner: int
    points_awarded: int
    nodes_expanded: int
    max_depth: int
    time_spent: float


def jugar_ronda(
    starting_player: int,
    cfgA: PlayerConfig,
    cfgB: PlayerConfig,
    show: bool = True,
) -> MatchReport:
    state = inicializar_ronda(starting_player)
    total_nodes = 0
    max_depth = 0
    total_time = 0.0

    if show:
        imprimir_tablero(state.board, state.pos[0], state.pos[1])
        print(f"Empieza: {colored_player(state.to_move)}")

    while True:
        p = state.to_move

        if not hay_movimientos_posibles(state, p):
            winner = 1 - p
            pts = contar_cartas_boca_arriba(state.board)
            if show:
                print(f"{colored_player(p)} no tiene movimientos válidos.")
                print(f"Gana {colored_player(winner)}")
                print(f"Puntos de la ronda: {pts} (cartas boca arriba)")
            return MatchReport(
                winner=winner,
                points_awarded=pts,
                nodes_expanded=total_nodes,
                max_depth=max_depth,
                time_spent=total_time,
            )

        if show:
            print(f"Turno de {colored_player(p)}")
            imprimir_tablero(state.board, state.pos[0], state.pos[1])

        cfg = cfgA if p == 0 else cfgB

        if cfg.kind == PlayerKind.HUMAN:
            move = pedir_movimiento_humano(state)
        elif cfg.kind == PlayerKind.RANDOM:
            move = elegir_move_random(state)
        elif cfg.kind == PlayerKind.GREEDY:
            move = elegir_move_greedy(state, cfg.weights)
        elif cfg.kind == PlayerKind.WORST:
            move = elegir_move_worst(state, cfg.weights)
        elif cfg.kind == PlayerKind.MINIMAX:
            move, stats = elegir_movimiento_minimax(state, cfg.max_time, cfg.weights)
            total_nodes += stats.nodes
            max_depth = max(max_depth, stats.max_depth)
            total_time += stats.elapsed
            if show:
                print(f"[IA] nodes={stats.nodes} depth={stats.max_depth} cutoffs={stats.cutoffs} time={stats.elapsed:.3f}s")
        else:
            raise ValueError("Tipo de jugador inválido.")

        state = aplicar_movimiento(state, move)

        if show:
            print(f"Movimiento de {colored_player(1 - state.to_move)}: k={move.k}, path={move.steps}")


def jugar_partida_hasta_16(cfgA: PlayerConfig, cfgB: PlayerConfig, show_rounds: bool = True) -> None:
    score = [0, 0]
    starting = PLAYER_A

    ronda = 1
    while score[0] < WIN_SCORE and score[1] < WIN_SCORE:
        if show_rounds:
            print("\n" + "=" * 40)
            print(f"RONDA {ronda} | Marcador: {colored_player(PLAYER_A)}={score[0]} {colored_player(PLAYER_B)}={score[1]}")
            print("=" * 40)

        rep = jugar_ronda(starting, cfgA, cfgB, show=show_rounds)
        score[rep.winner] += rep.points_awarded

        if show_rounds:
            print(f"Marcador actualizado: {colored_player(PLAYER_A)}={score[0]} {colored_player(PLAYER_B)}={score[1]}")
            print(f"Métricas ronda: winner={colored_player(rep.winner)} pts={rep.points_awarded} "
                  f"nodes={rep.nodes_expanded} depth={rep.max_depth} time={rep.time_spent:.3f}s")

        starting = 1 - starting
        ronda += 1

    winner = 0 if score[0] >= WIN_SCORE else 1
    print("\nPARTIDA TERMINADA")
    print(f"Ganador final: {colored_player(winner)}")
    print(f"Marcador final: {colored_player(PLAYER_A)}={score[0]} {colored_player(PLAYER_B)}={score[1]}")


# =========================
# BENCHMARK
# =========================
def benchmark(
    n_partidas: int,
    cfgA: PlayerConfig,
    cfgB: PlayerConfig,
) -> None:
    wins = [0, 0]
    total_nodes = 0
    total_depth = 0
    total_time = 0.0
    total_pts = [0, 0]

    starting = PLAYER_A
    for _ in range(n_partidas):
        rep = jugar_ronda(starting, cfgA, cfgB, show=False)
        wins[rep.winner] += 1
        total_pts[rep.winner] += rep.points_awarded
        total_nodes += rep.nodes_expanded
        total_depth = max(total_depth, rep.max_depth)
        total_time += rep.time_spent
        starting = 1 - starting

    print("\n=== BENCHMARK ===")
    print(f"Partidas: {n_partidas}")
    print(f"Victorias {colored_player(PLAYER_A)}: {wins[0]} | Puntos ganados por {colored_player(PLAYER_A)} (solo rondas ganadas): {total_pts[0]}")
    print(f"Victorias {colored_player(PLAYER_B)}: {wins[1]} | Puntos ganados por {colored_player(PLAYER_B)} (solo rondas ganadas): {total_pts[1]}")
    print(f"Nodos expandidos (total, solo Minimax): {total_nodes}")
    print(f"Profundidad máxima alcanzada (observada): {total_depth}")
    print(f"Tiempo total Minimax (suma decisiones): {total_time:.3f}s")


# =========================
# CLI
# =========================
def elegir_tipo(j: str) -> str:
    opts = [PlayerKind.HUMAN, PlayerKind.RANDOM, PlayerKind.GREEDY, PlayerKind.WORST, PlayerKind.MINIMAX]
    while True:
        print(f"Tipo para {j}: {opts}")
        t = input("> ").strip().lower()
        if t in opts:
            return t
        print("Opción inválida.")


def elegir_pesos() -> Tuple[float, float, float, float, float]:
    print("Pesos heurísticas (elige preset):")
    print("1) Equilibrada   (w1,w2,w3,w4,w5) = (1.0, 0.8, 1.2, 0.6, 0.7)")
    print("2) Agresiva      (w1,w2,w3,w4,w5) = (1.3, 0.6, 1.5, 0.4, 0.6)")
    while True:
        x = input("> ").strip()
        if x == "1":
            return (1.0, 0.8, 1.2, 0.6, 0.7)
        if x == "2":
            return (1.3, 0.6, 1.5, 0.4, 0.6)
        print("Elige 1 o 2.")


def main():
    random.seed()

    print("=== COLLAPSI (Consola) ===")
    print("1) Jugar partida completa (hasta 16 puntos)")
    print("2) Benchmark (N rondas rápidas)")

    mode = input("> ").strip()

    tA = elegir_tipo("Jugador A")
    tB = elegir_tipo("Jugador B")

    cfgA = PlayerConfig(kind=tA)
    cfgB = PlayerConfig(kind=tB)

    if cfgA.kind in (PlayerKind.MINIMAX, PlayerKind.GREEDY, PlayerKind.WORST):
        print("\nPesos para A:")
        cfgA.weights = elegir_pesos()
    if cfgB.kind in (PlayerKind.MINIMAX, PlayerKind.GREEDY, PlayerKind.WORST):
        print("\nPesos para B:")
        cfgB.weights = elegir_pesos()

    if cfgA.kind == PlayerKind.MINIMAX:
        cfgA.max_time = float(input("Max time IA A (seg): ").strip() or "3")
    if cfgB.kind == PlayerKind.MINIMAX:
        cfgB.max_time = float(input("Max time IA B (seg): ").strip() or "3")

    if mode == "1":
        jugar_partida_hasta_16(cfgA, cfgB, show_rounds=True)
    elif mode == "2":
        n = int(input("Cantidad de rondas (benchmark): ").strip() or "50")
        benchmark(n, cfgA, cfgB)
    else:
        print("Modo inválido.")


if __name__ == "__main__":
    main()