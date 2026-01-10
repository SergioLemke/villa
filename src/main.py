import pygame
import random
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# =========================
# Config (simples e mexível)
# =========================
GRID_W, GRID_H = 40, 40
TILE_SIZE = 14

FPS = 60
YEAR_SECONDS = 0.5  # 1 ano = 0.5s

START_POP = 80
WORK_AGE = 16
REPRO_MIN_AGE = 18
REPRO_MAX_AGE = 42

MEET_RADIUS = 5              # raio para "conhecer"
PAIR_CHANCE_PER_YEAR = 0.51  # chance de formar par (se tiver alguém perto)
BIRTH_CHANCE_PER_YEAR = 0.8  # chance de ter filho (se coabitando e dentro da idade)
MAX_KIDS_PER_COUPLE = 7

MUTATION_CHANCE = 0.22
MUTATION_STEP = 1            # mutação ±1

# movimento
CHILD_STEPS = 1
ADULT_STEPS = 4
ELDER_AGE = 60
ELDER_STEPS = 4

# casal buscando espaço: apenas “o suficiente”
SEEK_SPACE_TRIES = 30        # quantos candidatos avaliar por ano quando sem espaço
SEEK_SPACE_RADIUS = 2        # raio local pra buscar um lugar melhor (sem virar migração)
SEEK_SPACE_STEP_LIMIT = 1    # quantos passos o casal dá por ano quando travado sem espaço (1 = bem “humano”)

# zonas (profissão -> retângulo)
# (x0, y0, x1, y1) com x1/y1 exclusivos
PROFESSIONS = {
    "Farmer":   {"zone": (0, 0, 20, 20),  "color": (60, 200, 80)},
    "Smith":    {"zone": (20, 0, 40, 20), "color": (200, 80, 60)},
    "Merchant": {"zone": (0, 20, 20, 40), "color": (60, 120, 220)},
    "Hunter":   {"zone": (20, 20, 40, 40),"color": (220, 200, 60)},
}

# genes
DNA_BOUNDS = {
    "speed": (1, 10),
    "fert": (1, 10),
    "soc": (1, 10),
    "vit": (1, 10),
}

# pesos por profissão
PROF_WEIGHTS = {
    "Farmer":   {"speed": 0.7, "fert": 1.3, "soc": 0.9, "vit": 1.1},
    "Smith":    {"speed": 0.8, "fert": 0.9, "soc": 0.9, "vit": 1.4},
    "Merchant": {"speed": 0.9, "fert": 0.9, "soc": 1.4, "vit": 0.9},
    "Hunter":   {"speed": 1.3, "fert": 0.9, "soc": 0.8, "vit": 1.0},
}

BG = (16, 16, 18)
GRID_LINE = (32, 32, 36)
TEXT = (220, 220, 235)
SELECT = (255, 255, 255)


def clamp(v, a, b):
    return max(a, min(b, v))


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zone_center(z):
    x0, y0, x1, y1 = z
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def in_zone(x: int, y: int, zone: Tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = zone
    return (x0 <= x < x1) and (y0 <= y < y1)


def random_dna() -> Dict[str, int]:
    return {k: random.randint(lo, hi) for k, (lo, hi) in DNA_BOUNDS.items()}


def mutate_dna(dna: Dict[str, int]) -> Dict[str, int]:
    out = dict(dna)
    if random.random() < MUTATION_CHANCE:
        g = random.choice(list(out.keys()))
        lo, hi = DNA_BOUNDS[g]
        out[g] = clamp(out[g] + random.choice([-MUTATION_STEP, MUTATION_STEP]), lo, hi)
    return out


def dna_child_from_parents(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    child = {}
    for g in DNA_BOUNDS.keys():
        child[g] = random.choice([a[g], b[g]])
    return mutate_dna(child)


def aptitude_score(dna: Dict[str, int], prof: str) -> float:
    w = PROF_WEIGHTS[prof]
    s = 0.0
    for g in DNA_BOUNDS.keys():
        s += dna[g] * w[g]
    s += random.random() * 0.5
    return s


def choose_profession(dna: Dict[str, int]) -> str:
    best = None
    best_s = -1e9
    for p in PROFESSIONS.keys():
        apt = aptitude_score(dna, p)
        bonus = random.random() * 0.15
        score = apt + bonus
        if score > best_s:
            best_s = score
            best = p
    return best


@dataclass
class Person:
    pid: int
    x: int
    y: int
    age: int
    dna: Dict[str, int]
    profession: Optional[str] = None
    partner_id: Optional[int] = None
    cohabiting: bool = False
    married: bool = False
    kids_count: int = 0
    mother_id: Optional[int] = None
    father_id: Optional[int] = None
    born_year: int = 0
    died_year: Optional[int] = None


class World:
    def __init__(self):
        self.people: Dict[int, Person] = {}
        self.next_id = 1
        self.year = 0
        self.generation = 0

        self.births_last = 0
        self.deaths_last = 0

        # ocupação: (x,y) -> list de pids (0..2, e 2 só se casal)
        self.occ: Dict[Tuple[int, int], List[int]] = {}

        self.events: List[dict] = []

    def log(self, etype: str, data: dict):
        self.events.append({"year": self.year, "type": etype, **data})

    def save_to_file(self, path: str):
        payload = {
            "meta": {
                "grid_w": GRID_W,
                "grid_h": GRID_H,
                "start_pop": START_POP,
                "year_final": self.year,
            },
            "events": self.events,
            "final_people": [
                {
                    "pid": p.pid,
                    "age": p.age,
                    "born_year": p.born_year,
                    "died_year": p.died_year,
                    "mother": p.mother_id,
                    "father": p.father_id,
                    "profession": p.profession,
                    "partner": p.partner_id,
                    "cohabiting": p.cohabiting,
                    "married": p.married,
                    "kids_count": p.kids_count,
                    "pos": [p.x, p.y],
                    "dna": dict(p.dna),
                } for p in self.people.values()
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def add_person(self, x, y, age, dna, born_year, mother=None, father=None) -> Person:
        pid = self.next_id
        self.next_id += 1
        p = Person(
            pid=pid, x=x, y=y, age=age, dna=dna,
            profession=None,
            partner_id=None,
            cohabiting=False,
            married=False,
            kids_count=0,
            mother_id=mother, father_id=father,
            born_year=born_year,
            died_year=None
        )
        self.people[pid] = p
        self.occ.setdefault((x, y), []).append(pid)
        return p

    def init_population(self):
        for _ in range(START_POP):
            x = random.randrange(GRID_W)
            y = random.randrange(GRID_H)
            tries = 0
            while (x, y) in self.occ and tries < 50:
                x = random.randrange(GRID_W)
                y = random.randrange(GRID_H)
                tries += 1

            p = self.add_person(x=x, y=y, age=random.randint(0, 55), dna=random_dna(), born_year=0)
            if p.age >= WORK_AGE:
                p.profession = choose_profession(p.dna)
                self.log("profession_chosen", {"pid": p.pid, "profession": p.profession})

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

    def kill(self, pid: int):
        if pid not in self.people:
            return
        p = self.people[pid]
        p.died_year = self.year
        self.deaths_last += 1

        self.log("death", {
            "pid": p.pid,
            "age": p.age,
            "profession": p.profession,
            "pos": [p.x, p.y],
            "partner": p.partner_id,
        })

        # dissolve par
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

    # =========================
    # Espaço pra ter filhos (local)
    # =========================
    def has_free_adjacent(self, x: int, y: int) -> bool:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if (nx, ny) not in self.occ:
                    return True
        return False

    def preferred_zone_for_couple(self, a: Person, b: Person) -> Optional[Tuple[int, int, int, int]]:
        # se ambos têm a mesma profissão, prefere ficar no bairro dela
        if a.profession is not None and a.profession == b.profession:
            return PROFESSIONS[a.profession]["zone"]
        return None

    def zone_target_for_couple(self, a: Person, b: Person) -> Tuple[int, int]:
        # pra score suave quando profissões diferem
        if a.profession is None and b.profession is None:
            return (a.x, a.y)
        if a.profession is None:
            return zone_center(PROFESSIONS[b.profession]["zone"])
        if b.profession is None:
            return zone_center(PROFESSIONS[a.profession]["zone"])
        ax, ay = zone_center(PROFESSIONS[a.profession]["zone"])
        bx, by = zone_center(PROFESSIONS[b.profession]["zone"])
        return ((ax + bx) // 2, (ay + by) // 2)

    def choose_space_step_for_couple(self, a: Person, b: Person) -> Optional[Tuple[int, int]]:
        """
        Casal coabitando e sem espaço adjacente:
        escolhe 1 passo (4-dir) que melhora a chance de ter espaço, preferindo o mesmo bairro.
        """
        # se já tem espaço, nem mexe
        if self.has_free_adjacent(a.x, a.y):
            return None

        preferred_zone = self.preferred_zone_for_couple(a, b)
        ztx, zty = self.zone_target_for_couple(a, b)

        candidates = []
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(dirs)

        # primeiro: tenta passo direto em 4 direções
        for dx, dy in dirs:
            nx, ny = a.x + dx, a.y + dy
            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                continue
            if (nx, ny) in self.occ:
                continue

            space_ok = self.has_free_adjacent(nx, ny)
            zone_bonus = 0.0
            if preferred_zone is not None and in_zone(nx, ny, preferred_zone):
                zone_bonus += 3.0

            # preferir ficar perto do alvo de zona (quando profissões diferem)
            dist_zone = abs(nx - ztx) + abs(ny - zty)

            score = (10.0 if space_ok else 0.0) + zone_bonus - 0.2 * dist_zone
            candidates.append((score, nx, ny))

        if candidates:
            candidates.sort(reverse=True, key=lambda t: t[0])
            best_score, bx, by = candidates[0]
            # se nenhum passo melhora nada, ainda assim escolhe o melhorzinho
            return (bx, by)

        # se travou total, não faz nada (cidade lotada, paciência)
        return None

    # =========================
    # Movimento
    # =========================
    def move_person(self, p: Person):
        steps = CHILD_STEPS if p.age < WORK_AGE else (ELDER_STEPS if p.age >= ELDER_AGE else ADULT_STEPS)

        tx, ty = self.target_for_person(p)
        for _ in range(steps):
            dx = 0 if p.x == tx else (1 if tx > p.x else -1)
            dy = 0 if p.y == ty else (1 if ty > p.y else -1)

            if dx != 0 and dy != 0:
                if random.random() < 0.5:
                    nx, ny = p.x + dx, p.y
                else:
                    nx, ny = p.x, p.y + dy
            else:
                nx, ny = p.x + dx, p.y + dy

            if random.random() < 0.15:
                nx = p.x + random.choice([-1, 0, 1])
                ny = p.y + random.choice([-1, 0, 1])

            nx = clamp(nx, 0, GRID_W - 1)
            ny = clamp(ny, 0, GRID_H - 1)

            if (nx, ny) in self.occ:
                return
            self.place(p, nx, ny)

    def move_couple_unit(self, a: Person, b: Person, steps: int, tx: int, ty: int):
        """
        Casal ocupa o mesmo tile.
        Regra nova:
        - Se NÃO tem espaço adjacente pra nascer filho, dá poucos passos procurando espaço.
        - Se tem espaço, segue a lógica normal de movimento (zona).
        """
        # prioridade absoluta: achar espaço mínimo local
        if not self.has_free_adjacent(a.x, a.y):
            for _ in range(SEEK_SPACE_STEP_LIMIT):
                step = self.choose_space_step_for_couple(a, b)
                if step is None:
                    return
                nx, ny = step
                self.place(a, nx, ny)
                self.place(b, nx, ny)
            return

        # se já tem espaço, movimento normal
        for _ in range(steps):
            dx = 0 if a.x == tx else (1 if tx > a.x else -1)
            dy = 0 if a.y == ty else (1 if ty > a.y else -1)

            if dx != 0 and dy != 0:
                if random.random() < 0.5:
                    nx, ny = a.x + dx, a.y
                else:
                    nx, ny = a.x, a.y + dy
            else:
                nx, ny = a.x + dx, a.y + dy

            if random.random() < 0.15:
                nx = a.x + random.choice([-1, 0, 1])
                ny = a.y + random.choice([-1, 0, 1])

            nx = clamp(nx, 0, GRID_W - 1)
            ny = clamp(ny, 0, GRID_H - 1)

            if (nx, ny) in self.occ:
                return
            self.place(a, nx, ny)
            self.place(b, nx, ny)

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

    def target_for_person(self, p: Person) -> Tuple[int, int]:
        if p.profession is None:
            return (p.x, p.y)
        return zone_center(PROFESSIONS[p.profession]["zone"])

    def target_for_couple(self, a: Person, b: Person) -> Tuple[int, int]:
        if a.profession is None and b.profession is None:
            return (a.x, a.y)
        if a.profession is None:
            return zone_center(PROFESSIONS[b.profession]["zone"])
        if b.profession is None:
            return zone_center(PROFESSIONS[a.profession]["zone"])

        ax, ay = zone_center(PROFESSIONS[a.profession]["zone"])
        bx, by = zone_center(PROFESSIONS[b.profession]["zone"])
        return ((ax + bx) // 2, (ay + by) // 2)

    # =========================
    # Tick anual
    # =========================
    def tick_year(self):
        self.births_last = 0
        self.deaths_last = 0

        # 1) envelhecer
        for p in list(self.people.values()):
            p.age += 1

        # 2) escolher profissão quando atingir idade de trabalho
        for p in list(self.people.values()):
            if p.profession is None and p.age >= WORK_AGE:
                inherited = None
                if p.mother_id is not None and p.mother_id in self.people:
                    inherited = self.people[p.mother_id].profession
                if inherited is None and p.father_id is not None and p.father_id in self.people:
                    inherited = self.people[p.father_id].profession

                if inherited is not None and random.random() < 0.85:
                    p.profession = inherited
                else:
                    p.profession = choose_profession(p.dna)

                self.log("profession_chosen", {"pid": p.pid, "profession": p.profession})

        # 3) mortes (prob cresce com idade e dna vit)
        for p in list(self.people.values()):
            base = 0.002
            age_factor = max(0.0, (p.age - 35) / 80.0)
            vit = p.dna["vit"] / 10.0  # 0.1..1.0
            death_p = base + age_factor * (0.05) * (1.2 - vit)
            if random.random() < death_p:
                self.kill(p.pid)

        # 4) mover
        self.move_all()

        # 5) formar pares
        self.form_pairs()

        # 6) coabitar
        self.try_cohabit_pairs()

        # 7) reproduzir
        self.try_births()

        self.year += 1

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
                    soc = (p.dna["soc"] + q.dna["soc"]) / 20.0  # 0.1..1.0
                    candidates.append((q, soc))

            if not candidates:
                continue

            q, soc = random.choice(candidates)
            chance = PAIR_CHANCE_PER_YEAR * (0.6 + 0.8 * soc)
            if random.random() < chance:
                p.partner_id = q.pid
                q.partner_id = p.pid
                p.cohabiting = False
                q.cohabiting = False
                self.log("pair_formed", {"a": p.pid, "b": q.pid})

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

            # tenta aproximar se estiverem distantes
            if manhattan((p.x, p.y), (q.x, q.y)) > 1:
                continue

            # se estão adjacentes, tenta juntar no tile de p ou q
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

            fert = (p.dna["fert"] + q.dna["fert"]) / 20.0  # 0.1..1.0
            chance = BIRTH_CHANCE_PER_YEAR * (0.6 + 0.8 * fert)

            if p.kids_count >= MAX_KIDS_PER_COUPLE or q.kids_count >= MAX_KIDS_PER_COUPLE:
                continue

            if random.random() < chance:
                adj = [(p.x + 1, p.y), (p.x - 1, p.y), (p.x, p.y + 1), (p.x, p.y - 1)]
                random.shuffle(adj)
                spot = None
                for (nx, ny) in adj:
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in self.occ:
                        spot = (nx, ny)
                        break
                if spot is None:
                    # sem espaço local adjacente: sem filho esse ano
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
def draw_world(surface, world: World, font, selected_tile: Optional[Tuple[int, int]]):
    surface.fill(BG)

    # fundo das zonas
    for _, info in PROFESSIONS.items():
        x0, y0, x1, y1 = info["zone"]
        c = info["color"]
        zone_col = (c[0] // 4, c[1] // 4, c[2] // 4)
        rect = pygame.Rect(x0 * TILE_SIZE, y0 * TILE_SIZE, (x1 - x0) * TILE_SIZE, (y1 - y0) * TILE_SIZE)
        pygame.draw.rect(surface, zone_col, rect)

    # grid
    for x in range(GRID_W + 1):
        pygame.draw.line(surface, GRID_LINE, (x * TILE_SIZE, 0), (x * TILE_SIZE, GRID_H * TILE_SIZE), 1)
    for y in range(GRID_H + 1):
        pygame.draw.line(surface, GRID_LINE, (0, y * TILE_SIZE), (GRID_W * TILE_SIZE, y * TILE_SIZE), 1)

    # pessoas
    for p in world.people.values():
        col = (200, 200, 200)
        if p.profession is not None:
            col = PROFESSIONS[p.profession]["color"]

        cx = p.x * TILE_SIZE + TILE_SIZE // 2
        cy = p.y * TILE_SIZE + TILE_SIZE // 2
        r = 3 if p.age < WORK_AGE else 4
        pygame.draw.circle(surface, col, (cx, cy), r)

        if p.cohabiting:
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), r + 1, 1)

    # tile selecionado
    if selected_tile is not None:
        x, y = selected_tile
        rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(surface, SELECT, rect, 2)

    # HUD
    pop = len(world.people)
    txt = f"Ano: {world.year}  | Pop: {pop}  | Nasc: {world.births_last}  | Mortes: {world.deaths_last}"
    surf = font.render(txt, True, TEXT)
    surface.blit(surf, (8, 8))


def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * TILE_SIZE, GRID_H * TILE_SIZE))
    pygame.display.set_caption("Villa2 - Simulação Social")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    world = World()
    world.init_population()

    selected_tile = None
    running = True
    sim_time = 0.0

    while running:
        dt = clock.tick(FPS) / 1000.0
        sim_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                tx = mx // TILE_SIZE
                ty = my // TILE_SIZE
                if 0 <= tx < GRID_W and 0 <= ty < GRID_H:
                    selected_tile = (tx, ty)

        if sim_time >= YEAR_SECONDS:
            sim_time = 0.0
            world.tick_year()

        draw_world(screen, world, font, selected_tile)
        pygame.display.flip()

    # salva log
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"sim_log_{ts}.json"
    world.save_to_file(out)
    print("Log salvo em:", out)

    pygame.quit()


if __name__ == "__main__":
    main()
