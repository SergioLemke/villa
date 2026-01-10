import pygame
import random
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

# =========================
# Config (simples e mexível)
# =========================
GRID_W, GRID_H = 40, 40
TILE_SIZE = 14

FPS = 60
YEAR_SECONDS = 0.5  # 1 ano = 10s

START_POP = 80
WORK_AGE = 16
REPRO_MIN_AGE = 18
REPRO_MAX_AGE = 40

MEET_RADIUS = 7              # raio para "conhecer"
PAIR_CHANCE_PER_YEAR = 0.51  # chance de formar par (se tiver alguém perto)
BIRTH_CHANCE_PER_YEAR = 0.8 # chance de ter filho (se coabitando e dentro da idade)
MAX_KIDS_PER_COUPLE = 7

MUTATION_CHANCE = 0.18
MUTATION_STEP = 1            # mutação ±1

# movimento
CHILD_STEPS = 3
ADULT_STEPS = 8
ELDER_AGE = 60
ELDER_STEPS = 1
BIAS_PROB = 0.80  # 80% escolhe melhor direção, 20% faz ruído

# =========================
# DNA / Profissões / Zonas
# =========================
STATS = ["str", "int", "dex", "cha", "vit"]

PROFESSIONS = {
    "Farmer": {
        "color": (60, 150, 70),
        "zone": (0, 0, GRID_W // 2, GRID_H // 2),
        "weights": {"str": 0.45, "vit": 0.35, "dex": 0.10, "int": 0.05, "cha": 0.05},
    },
    "Hunter": {
        "color": (180, 80, 80),
        "zone": (GRID_W // 2, 0, GRID_W, GRID_H // 2),
        "weights": {"dex": 0.45, "vit": 0.25, "str": 0.20, "int": 0.05, "cha": 0.05},
    },
    "Merchant": {
        "color": (200, 170, 70),
        "zone": (0, GRID_H // 2, GRID_W // 2, GRID_H),
        "weights": {"int": 0.40, "cha": 0.35, "dex": 0.10, "vit": 0.10, "str": 0.05},
    },
    "Smith": {
        "color": (120, 120, 140),
        "zone": (GRID_W // 2, GRID_H // 2, GRID_W, GRID_H),
        "weights": {"str": 0.35, "int": 0.25, "vit": 0.20, "dex": 0.15, "cha": 0.05},
    },
}

CHILD_COLOR = (110, 110, 110)
BG = (18, 18, 22)
GRID_LINE = (30, 30, 36)

HUD_HEIGHT = 190  # área embaixo pro texto

# ============
# Morte por idade
# ============
def base_death_risk(age: int) -> float:
    if age <= 5: return 0.02
    if age <= 17: return 0.003
    if age <= 39: return 0.006
    if age <= 59: return 0.015
    if age <= 79: return 0.04
    if age <= 99: return 0.12
    return 0.25

def adjusted_death_risk(age: int, vit: int) -> float:
    base = base_death_risk(age)
    factor = 1.0 - (vit / 200.0)  # vit 100 => risco pela metade
    risk = base * factor
    return max(0.0, min(risk, 0.95))

# ============
# Utilidades
# ============
def clamp(v, lo, hi): return max(lo, min(hi, v))

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def zone_center(zone):
    x0, y0, x1, y1 = zone
    return ((x0 + x1) // 2, (y0 + y1) // 2)

def best_direction_towards(x, y, tx, ty, occupied_check):
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    scored = []
    for dx, dy in dirs:
        nx, ny = x+dx, y+dy
        if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
            continue
        if occupied_check(nx, ny):
            continue
        dist = abs(nx-tx) + abs(ny-ty)
        scored.append((dist, (dx, dy)))
    scored.sort(key=lambda t: t[0])
    return [d for _, d in scored]

def random_mutation() -> int:
    if random.random() > MUTATION_CHANCE:
        return 0
    return random.choice([-MUTATION_STEP, MUTATION_STEP])

def dna_random_low() -> Dict[str,int]:
    return {k: random.randint(0, 5) for k in STATS}

def dna_child_from_parents(dna_a, dna_b) -> Dict[str,int]:
    child = {}
    for k in STATS:
        avg = int(round((dna_a[k] + dna_b[k]) / 2.0))
        m = random_mutation()
        child[k] = clamp(avg + m, 0, 100)
    return child

def aptitude_score(dna: Dict[str,int], prof_name: str) -> float:
    w = PROFESSIONS[prof_name]["weights"]
    return sum(dna[s] * w[s] for s in STATS)

def choose_profession(dna: Dict[str,int], parent_prof_a: Optional[str], parent_prof_b: Optional[str]) -> str:
    best = None
    best_score = -1e9
    for p in PROFESSIONS.keys():
        apt = aptitude_score(dna, p)
        bonus = 0.0
        if parent_prof_a == p: bonus += 8.0
        if parent_prof_b == p: bonus += 8.0
        score = 0.7 * apt + 0.3 * bonus
        if score > best_score:
            best_score = score
            best = p
    return best

# ============
# Pessoas
# ============
@dataclass
class Person:
    pid: int
    x: int
    y: int
    age: int
    dna: Dict[str,int]
    profession: Optional[str] = None
    partner_id: Optional[int] = None
    cohabiting: bool = False
    married: bool = False
    kids_count: int = 0
    mother_id: Optional[int] = None
    father_id: Optional[int] = None
    born_year: int = 0
    died_year: Optional[int] = None
    history: List[Tuple[int, Dict[str,int]]] = field(default_factory=list)

    def color(self):
        if self.profession is None:
            return CHILD_COLOR
        return PROFESSIONS[self.profession]["color"]

# ============
# Mundo
# ============
class World:
    def __init__(self):
        self.year = 0
        self.generation = 0
        self.people: Dict[int, Person] = {}
        self.next_id = 1

        self.births_last = 0
        self.deaths_last = 0

        # ocupação: (x,y) -> list de pids (0..2, e 2 só se casal)
        self.occ: Dict[Tuple[int,int], List[int]] = {}

        self.events: List[dict] = []

    def log(self, etype: str, data: dict):
        self.events.append({"year": self.year, "type": etype, **data})

    def save_to_file(self, path: str):
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "final_year": self.year,
            "final_generation": self.generation,
            "config": {
                "GRID_W": GRID_W, "GRID_H": GRID_H, "TILE_SIZE": TILE_SIZE,
                "YEAR_SECONDS": YEAR_SECONDS,
                "START_POP": START_POP,
                "WORK_AGE": WORK_AGE,
                "REPRO_MIN_AGE": REPRO_MIN_AGE,
                "REPRO_MAX_AGE": REPRO_MAX_AGE,
                "MEET_RADIUS": MEET_RADIUS,
                "PAIR_CHANCE_PER_YEAR": PAIR_CHANCE_PER_YEAR,
                "BIRTH_CHANCE_PER_YEAR": BIRTH_CHANCE_PER_YEAR,
                "MAX_KIDS_PER_COUPLE": MAX_KIDS_PER_COUPLE,
                "MUTATION_CHANCE": MUTATION_CHANCE,
                "MUTATION_STEP": MUTATION_STEP,
            },
            "events": self.events,
            "final_people": []
        }

        for p in self.people.values():
            payload["final_people"].append({
                "pid": p.pid,
                "age": p.age,
                "x": p.x, "y": p.y,
                "profession": p.profession,
                "partner_id": p.partner_id,
                "cohabiting": p.cohabiting,
                "married": p.married,
                "kids_count": p.kids_count,
                "born_year": p.born_year,
                "mother_id": p.mother_id,
                "father_id": p.father_id,
                "dna": p.dna,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def occupied_by_non_couple(self, x, y, incoming_pid=None, incoming_partner=None) -> bool:
        key = (x, y)
        if key not in self.occ or len(self.occ[key]) == 0:
            return False
        if len(self.occ[key]) >= 2:
            return True

        existing_pid = self.occ[key][0]
        if incoming_pid is None or incoming_partner is None:
            return True

        # pode entrar só se o existente for o parceiro
        return existing_pid != incoming_partner

    def place(self, p: Person, x: int, y: int):
        old = (p.x, p.y)
        if old in self.occ and p.pid in self.occ[old]:
            self.occ[old].remove(p.pid)
            if not self.occ[old]:
                del self.occ[old]

        p.x, p.y = x, y
        key = (x, y)
        self.occ.setdefault(key, []).append(p.pid)

    def add_person(self, x, y, age, dna, born_year, mother=None, father=None) -> Person:
        pid = self.next_id
        self.next_id += 1
        p = Person(
            pid=pid, x=x, y=y, age=age, dna=dna,
            mother_id=mother, father_id=father,
            born_year=born_year
        )
        p.history.append((born_year, dict(dna)))
        self.people[pid] = p
        self.occ.setdefault((x, y), []).append(pid)
        return p

    def random_empty_tile(self) -> Tuple[int,int]:
        for _ in range(8000):
            x = random.randrange(GRID_W)
            y = random.randrange(GRID_H)
            if (x,y) not in self.occ:
                return x,y
        for y in range(GRID_H):
            for x in range(GRID_W):
                if (x,y) not in self.occ:
                    return x,y
        return 0,0

    def init_population(self):
        for _ in range(START_POP):
            x,y = self.random_empty_tile()
            age = random.randint(0, 40)
            dna = dna_random_low()
            p = self.add_person(x,y,age,dna,born_year=0)
            if age >= WORK_AGE:
                p.profession = choose_profession(p.dna, None, None)
                self.log("profession_chosen", {
                    "pid": p.pid,
                    "profession": p.profession,
                    "pos": [p.x, p.y],
                    "dna": dict(p.dna),
                })

    def step_year(self):
        self.year += 1
        self.births_last = 0
        self.deaths_last = 0

        # 1) envelhece
        for p in self.people.values():
            p.age += 1

        # 2) escolhe profissão na idade certa
        for p in self.people.values():
            if p.profession is None and p.age >= WORK_AGE:
                parent_a = self.people.get(p.mother_id).profession if p.mother_id in self.people else None
                parent_b = self.people.get(p.father_id).profession if p.father_id in self.people else None
                p.profession = choose_profession(p.dna, parent_a, parent_b)
                self.log("profession_chosen", {
                    "pid": p.pid,
                    "profession": p.profession,
                    "pos": [p.x, p.y],
                    "dna": dict(p.dna),
                })

        # 3) mortes
        to_die = []
        for p in self.people.values():
            r = adjusted_death_risk(p.age, p.dna["vit"])
            if random.random() < r:
                to_die.append(p.pid)

        for pid in to_die:
            self.kill(pid)

        # 4) movimento
        self.move_all()

        # 5) formar pares
        self.form_pairs()

        # 6) coabitar
        self.try_cohabit_pairs()

        # 7) reproduzir
        self.try_births()

    def kill(self, pid: int):
        if pid not in self.people:
            return
        p = self.people[pid]
        p.died_year = self.year
        self.deaths_last += 1

        self.log("death", {
            "pid": p.pid,
            "age": p.age,
            "pos": [p.x, p.y],
            "partner_id": p.partner_id,
            "profession": p.profession,
        })

        # desfaz par
        if p.partner_id is not None and p.partner_id in self.people:
            partner = self.people[p.partner_id]
            partner.partner_id = None
            partner.cohabiting = False

        key = (p.x, p.y)
        if key in self.occ and pid in self.occ[key]:
            self.occ[key].remove(pid)
            if not self.occ[key]:
                del self.occ[key]

        del self.people[pid]

    def move_all(self):
        moved = set()

        # move casais (uma vez por par)
        for p in list(self.people.values()):
            if p.partner_id is None:
                continue
            if p.pid in moved:
                continue
            if p.partner_id not in self.people:
                p.partner_id = None
                p.cohabiting = False
                continue

            q = self.people[p.partner_id]
            moved.add(p.pid)
            moved.add(q.pid)

            if p.cohabiting and (p.x, p.y) == (q.x, q.y):
                steps = ADULT_STEPS if max(p.age, q.age) < ELDER_AGE else ELDER_STEPS
                tx, ty = self.target_for_couple(p, q)
                self.move_couple_unit(p, q, steps, tx, ty)
            else:
                self.move_person(p)
                self.move_person(q)

        # move solteiros
        for p in list(self.people.values()):
            if p.partner_id is None:
                self.move_person(p)

    def target_for_person(self, p: Person) -> Tuple[int,int]:
        if p.profession is None:
            return (p.x, p.y)
        return zone_center(PROFESSIONS[p.profession]["zone"])

    def target_for_couple(self, a: Person, b: Person) -> Tuple[int,int]:
        if a.profession is None and b.profession is None:
            return (a.x, a.y)
        if a.profession is None:
            return zone_center(PROFESSIONS[b.profession]["zone"])
        if b.profession is None:
            return zone_center(PROFESSIONS[a.profession]["zone"])

        sa = aptitude_score(a.dna, a.profession)
        sb = aptitude_score(b.dna, b.profession)
        lead = a if sa >= sb else b
        return zone_center(PROFESSIONS[lead.profession]["zone"])

    def steps_for_age(self, age: int) -> int:
        if age < WORK_AGE:
            return CHILD_STEPS
        if age >= ELDER_AGE:
            return ELDER_STEPS
        return ADULT_STEPS

    def move_person(self, p: Person):
        steps = self.steps_for_age(p.age)
        tx, ty = self.target_for_person(p)

        for _ in range(steps):
            # se está em par (mas não coabitando), às vezes puxa pro parceiro
            if p.partner_id is not None and p.partner_id in self.people and not p.cohabiting:
                partner = self.people[p.partner_id]
                if random.random() < 0.6:
                    tx, ty = partner.x, partner.y

            best_dirs = best_direction_towards(
                p.x, p.y, tx, ty,
                occupied_check=lambda nx, ny: self.occupied_by_non_couple(nx, ny)
            )
            if not best_dirs:
                return

            # viés forte com ruído
            if random.random() < BIAS_PROB:
                choices = best_dirs
            else:
                choices = best_dirs[:]
                random.shuffle(choices)

            moved_ok = False
            for dx, dy in choices:
                nx, ny = p.x + dx, p.y + dy
                if not self.occupied_by_non_couple(nx, ny):
                    self.place(p, nx, ny)
                    moved_ok = True
                    break
            if not moved_ok:
                return

    def move_couple_unit(self, a: Person, b: Person, steps: int, tx: int, ty: int):
        for _ in range(steps):
            best_dirs = best_direction_towards(
                a.x, a.y, tx, ty,
                occupied_check=lambda nx, ny: (nx, ny) in self.occ  # casal precisa tile vazio
            )
            if not best_dirs:
                return

            if random.random() < BIAS_PROB:
                choices = best_dirs
            else:
                choices = best_dirs[:]
                random.shuffle(choices)

            moved_ok = False
            for dx, dy in choices:
                nx, ny = a.x + dx, a.y + dy
                if (nx, ny) not in self.occ:
                    self.place(a, nx, ny)
                    self.place(b, nx, ny)
                    a.cohabiting = True
                    b.cohabiting = True
                    moved_ok = True
                    break
            if not moved_ok:
                return

    def form_pairs(self):
        singles = [p for p in self.people.values()
                   if p.partner_id is None and REPRO_MIN_AGE <= p.age <= REPRO_MAX_AGE]
        random.shuffle(singles)

        for p in singles:
            if p.partner_id is not None or p.pid not in self.people:
                continue

            candidates = []
            for q in self.people.values():
                if q.pid == p.pid:
                    continue
                if q.partner_id is not None:
                    continue
                if not (REPRO_MIN_AGE <= q.age <= REPRO_MAX_AGE):
                    continue
                if manhattan((p.x, p.y), (q.x, q.y)) <= MEET_RADIUS:
                    candidates.append(q)

            if not candidates:
                continue

            if random.random() < PAIR_CHANCE_PER_YEAR:
                q = random.choice(candidates)
                p.partner_id = q.pid
                q.partner_id = p.pid
                p.cohabiting = False
                q.cohabiting = False
                self.log("pair_formed", {
                    "a": p.pid, "b": q.pid,
                    "pos_a": [p.x, p.y],
                    "pos_b": [q.x, q.y],
                })

    def try_cohabit_pairs(self):
        visited = set()
        for p in list(self.people.values()):
            if p.partner_id is None or p.pid in visited:
                continue
            if p.partner_id not in self.people:
                p.partner_id = None
                p.cohabiting = False
                continue

            q = self.people[p.partner_id]
            visited.add(p.pid)
            visited.add(q.pid)

            if p.cohabiting and (p.x, p.y) == (q.x, q.y):
                continue

            if manhattan((p.x, p.y), (q.x, q.y)) == 1:
                if not self.occupied_by_non_couple(p.x, p.y, incoming_pid=q.pid, incoming_partner=p.pid):
                    self.place(q, p.x, p.y)
                    p.cohabiting = True
                    q.cohabiting = True
                    self.log("cohabitation", {"a": p.pid, "b": q.pid, "pos": [p.x, p.y]})
                elif not self.occupied_by_non_couple(q.x, q.y, incoming_pid=p.pid, incoming_partner=q.pid):
                    self.place(p, q.x, q.y)
                    p.cohabiting = True
                    q.cohabiting = True
                    self.log("cohabitation", {"a": p.pid, "b": q.pid, "pos": [q.x, q.y]})

    def try_births(self):
        visited = set()
        for p in list(self.people.values()):
            if p.partner_id is None or p.pid in visited:
                continue
            if p.partner_id not in self.people:
                continue
            q = self.people[p.partner_id]
            visited.add(p.pid)
            visited.add(q.pid)

            if not (p.cohabiting and q.cohabiting and (p.x, p.y) == (q.x, q.y)):
                continue

            if not (REPRO_MIN_AGE <= p.age <= REPRO_MAX_AGE and REPRO_MIN_AGE <= q.age <= REPRO_MAX_AGE):
                continue

            if p.kids_count >= MAX_KIDS_PER_COUPLE or q.kids_count >= MAX_KIDS_PER_COUPLE:
                continue

            if random.random() < BIRTH_CHANCE_PER_YEAR:
                adj = [(p.x+1,p.y),(p.x-1,p.y),(p.x,p.y+1),(p.x,p.y-1)]
                random.shuffle(adj)
                spot = None
                for (nx, ny) in adj:
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in self.occ:
                        spot = (nx, ny)
                        break
                if spot is None:
                    continue

                baby_dna = dna_child_from_parents(p.dna, q.dna)
                baby = self.add_person(
                    x=spot[0], y=spot[1], age=0, dna=baby_dna,
                    born_year=self.year,
                    mother=p.pid, father=q.pid
                )

                self.generation += 1
                self.births_last += 1
                p.kids_count += 1
                q.kids_count += 1

                if not (p.married and q.married):
                    p.married = True
                    q.married = True
                    self.log("marriage", {"a": p.pid, "b": q.pid, "pos": [p.x, p.y]})

                self.log("birth", {
                    "baby": baby.pid,
                    "mother": p.pid,
                    "father": q.pid,
                    "pos": [baby.x, baby.y],
                    "dna": dict(baby.dna),
                    "generation": self.generation,
                })

# ============
# Render
# ============
def draw_world(surface, world: World, font, selected_tile: Optional[Tuple[int,int]]):
    surface.fill(BG)

    # fundo das zonas
    for _, info in PROFESSIONS.items():
        x0,y0,x1,y1 = info["zone"]
        c = info["color"]
        zone_col = (c[0]//4, c[1]//4, c[2]//4)
        rect = pygame.Rect(x0*TILE_SIZE, y0*TILE_SIZE, (x1-x0)*TILE_SIZE, (y1-y0)*TILE_SIZE)
        pygame.draw.rect(surface, zone_col, rect)

    # grid
    for x in range(GRID_W+1):
        pygame.draw.line(surface, GRID_LINE, (x*TILE_SIZE, 0), (x*TILE_SIZE, GRID_H*TILE_SIZE))
    for y in range(GRID_H+1):
        pygame.draw.line(surface, GRID_LINE, (0, y*TILE_SIZE), (GRID_W*TILE_SIZE, y*TILE_SIZE))

    # pessoas
    for p in world.people.values():
        px = p.x*TILE_SIZE
        py = p.y*TILE_SIZE
        col = p.color()
        r = pygame.Rect(px+2, py+2, TILE_SIZE-4, TILE_SIZE-4)
        pygame.draw.rect(surface, col, r)

        if p.partner_id is not None and p.cohabiting and p.partner_id in world.people:
            pygame.draw.rect(surface, (240, 240, 240), r, 1)

    # destaque do tile selecionado
    if selected_tile is not None:
        tx, ty = selected_tile
        highlight = pygame.Rect(tx*TILE_SIZE, ty*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(surface, (255, 255, 255), highlight, 2)

    # HUD
    alive = len(world.people)
    hud_lines = [
        "Controles: [Espaco]=+1 ano | [R]=reset | [Q/Esc]=salvar e sair | Clique num tile p/ ver info",
        f"Ano: {world.year}  (1 ano={YEAR_SECONDS:.0f}s)   Geracao(bebe=+1): {world.generation}",
        f"Pop: {alive}   Nasc(ano): {world.births_last}   Mortes(ano): {world.deaths_last}",
        f"Pareamento: raio {MEET_RADIUS} | chance/ano {PAIR_CHANCE_PER_YEAR:.2f} | Filho/ano {BIRTH_CHANCE_PER_YEAR:.2f}",
    ]

    if alive > 0:
        avg = {k: 0.0 for k in STATS}
        for p in world.people.values():
            for k in STATS:
                avg[k] += p.dna[k]
        for k in STATS:
            avg[k] /= alive
        hud_lines.append("DNA medio: " + " ".join([f"{k}:{avg[k]:.1f}" for k in STATS]))

    if selected_tile is not None:
        ids = world.occ.get(selected_tile, [])
        hud_lines.append(f"Tile selecionado {selected_tile} | ocupantes: {len(ids)}")
        for pid in ids[:4]:
            p = world.people.get(pid)
            if not p:
                continue
            prof = p.profession if p.profession else "Crianca"
            partner = f" parceiro:{p.partner_id}" if p.partner_id else ""
            hud_lines.append(
                f" - ID {p.pid} | idade {p.age} | {prof}{partner} | dna " +
                " ".join([f"{k}:{p.dna[k]}" for k in STATS])
            )
        if len(ids) > 4:
            hud_lines.append(f" ... (+{len(ids)-4} no tile)")

    base_y = GRID_H*TILE_SIZE + 6
    for line in hud_lines:
        surf = font.render(line, True, (230,230,230))
        surface.blit(surf, (8, base_y))
        base_y += surf.get_height() + 4

def make_log_filename() -> str:
    return f"sim_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def main():
    pygame.init()

    # Canvas base (tamanho fixo do "mundo"), depois a gente escala pro tamanho da janela
    BASE_W = GRID_W * TILE_SIZE
    BASE_H = GRID_H * TILE_SIZE + HUD_HEIGHT
    base_surface = pygame.Surface((BASE_W, BASE_H))

    # Janela redimensionável
    screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
    pygame.display.set_caption("Village Evolution (MVP)")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    world = World()
    world.init_population()

    elapsed = 0.0
    running = True
    selected_tile: Optional[Tuple[int,int]] = None

    def save_and_quit():
        fname = make_log_filename()
        world.save_to_file(fname)
        print(f"[OK] Salvo: {fname}")

    while running:
        dt = clock.tick(FPS) / 1000.0
        elapsed += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_and_quit()
                running = False

            if event.type == pygame.VIDEORESIZE:
                # recria a janela com o novo tamanho
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                win_w, win_h = screen.get_size()

                # converte coordenadas da janela -> coordenadas do canvas base
                bx = int(mx * (BASE_W / max(1, win_w)))
                by = int(my * (BASE_H / max(1, win_h)))

                # só considera clique dentro do grid (não na HUD)
                if by < GRID_H * TILE_SIZE:
                    tx = bx // TILE_SIZE
                    ty = by // TILE_SIZE
                    if 0 <= tx < GRID_W and 0 <= ty < GRID_H:
                        selected_tile = (tx, ty)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    world.step_year()

                if event.key == pygame.K_r:
                    world = World()
                    world.init_population()
                    elapsed = 0.0
                    selected_tile = None

                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    save_and_quit()
                    running = False

        # avança 1 ano automaticamente
        if elapsed >= YEAR_SECONDS:
            elapsed -= YEAR_SECONDS
            world.step_year()

        # desenha sempre no canvas base
        draw_world(base_surface, world, font, selected_tile)

        # escala pro tamanho atual da janela
        win_w, win_h = screen.get_size()
        scaled = pygame.transform.smoothscale(base_surface, (win_w, win_h))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
