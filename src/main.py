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

MEET_RADIUS = 3              # raio para "conhecer"
PAIR_CHANCE_PER_YEAR = 0.51  # chance de formar par (se tiver alguém perto)
BIRTH_CHANCE_PER_YEAR = 0.8 # chance de ter filho (se coabitando e dentro da idade)
MAX_KIDS_PER_COUPLE = 4

NEW_HOUSE_MIN_DIST = 10      # distância mínima (Manhattan) pra casal recém-coabitando criar nova 'casa'
NEW_HOUSE_TRIES = 300        # tentativas pra achar tile vazio adequado

MUTATION_CHANCE = 0.18
MUTATION_STEP = 1            # mutação ±1

# movimento
CHILD_STEPS = 1
ADULT_STEPS = 4
ELDER_AGE = 60
ELDER_STEPS = 4

# zonas (profissão -> retângulo)
# (x0, y0, x1, y1) com x1/y1 exclusivos
PROFESSIONS = {
    "Farmer":   {"zone": (0, 0, 20, 20), "color": (60, 200, 80)},
    "Smith":    {"zone": (20, 0, 40, 20), "color": (200, 80, 60)},
    "Merchant": {"zone": (0, 20, 20, 40), "color": (60, 120, 220)},
    "Hunter":   {"zone": (20, 20, 40, 40), "color": (220, 200, 60)},
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

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def zone_center(z):
    x0,y0,x1,y1 = z
    return ((x0+x1)//2, (y0+y1)//2)

def random_dna() -> Dict[str,int]:
    return {k: random.randint(lo, hi) for k, (lo,hi) in DNA_BOUNDS.items()}

def mutate_dna(dna: Dict[str,int]) -> Dict[str,int]:
    out = dict(dna)
    if random.random() < MUTATION_CHANCE:
        g = random.choice(list(out.keys()))
        lo, hi = DNA_BOUNDS[g]
        out[g] = clamp(out[g] + random.choice([-MUTATION_STEP, MUTATION_STEP]), lo, hi)
    return out

def dna_child_from_parents(a: Dict[str,int], b: Dict[str,int]) -> Dict[str,int]:
    child = {}
    for g in DNA_BOUNDS.keys():
        child[g] = random.choice([a[g], b[g]])
    return mutate_dna(child)

def aptitude_score(dna: Dict[str,int], prof: str) -> float:
    w = PROF_WEIGHTS[prof]
    # score ponderado simples
    s = 0.0
    for g in DNA_BOUNDS.keys():
        s += dna[g] * w[g]
    # bônus pequeno de aleatoriedade
    s += random.random() * 0.5
    return s

def choose_profession(dna: Dict[str,int]) -> str:
    best = None
    best_s = -1e9
    for p in PROFESSIONS.keys():
        apt = aptitude_score(dna, p)
        bonus = 0.0
        # incentiva diversidade muito levemente
        bonus += random.random() * 0.15
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
    dna: Dict[str,int]
    profession: Optional[str] = None
    partner_id: Optional[int] = None
    cohabiting: bool = False
    married: bool = False
    home_x: Optional[int] = None
    home_y: Optional[int] = None
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
        self.occ: Dict[Tuple[int,int], List[int]] = {}

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
                    "home": [p.home_x, p.home_y] if p.home_x is not None else None,
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
        self.occ.setdefault((x,y), []).append(pid)
        return p

    def init_population(self):
        for _ in range(START_POP):
            x = random.randrange(GRID_W)
            y = random.randrange(GRID_H)
            # se tile já tem alguém, tenta outros
            tries = 0
            while (x,y) in self.occ and tries < 50:
                x = random.randrange(GRID_W)
                y = random.randrange(GRID_H)
                tries += 1
            p = self.add_person(x=x, y=y, age=random.randint(0, 55), dna=random_dna(), born_year=0)
            # escolhe profissão ao atingir idade de trabalho
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

    def avg_parent_pos(self, a: Person, b: Person) -> Tuple[int,int]:
        """Tenta achar um 'centro' dos pais (se estiverem vivos). Se não der, usa a posição atual do casal."""
        pts = []
        for pid in (a.mother_id, a.father_id, b.mother_id, b.father_id):
            if pid is not None and pid in self.people:
                pp = self.people[pid]
                pts.append((pp.x, pp.y))
        if pts:
            ox = int(round(sum(x for x,_ in pts) / len(pts)))
            oy = int(round(sum(y for _,y in pts) / len(pts)))
            ox = max(0, min(GRID_W-1, ox))
            oy = max(0, min(GRID_H-1, oy))
            return (ox, oy)
        return (a.x, a.y)

    def find_empty_tile_far(self, origin: Tuple[int,int], min_dist: int, tries: int) -> Optional[Tuple[int,int]]:
        """Procura um tile vazio (sem ninguém) a pelo menos min_dist do origin."""
        best = None
        best_d = -1
        ox, oy = origin
        for _ in range(tries):
            x = random.randrange(GRID_W)
            y = random.randrange(GRID_H)
            if (x, y) in self.occ:
                continue
            d = abs(x-ox) + abs(y-oy)
            if d < min_dist:
                continue
            if d > best_d:
                best = (x, y)
                best_d = d
        return best

    def form_new_household(self, a: Person, b: Person):
        """Quando o casal começa a coabitar pela 1ª vez, cria uma 'casa' nova em um tile livre distante."""
        # já tem casa? não mexe.
        if a.home_x is not None or b.home_x is not None:
            return

        origin = self.avg_parent_pos(a, b)

        # tenta respeitar distância mínima, senão relaxa
        spot = self.find_empty_tile_far(origin, NEW_HOUSE_MIN_DIST, NEW_HOUSE_TRIES)
        if spot is None:
            spot = self.find_empty_tile_far(origin, 0, NEW_HOUSE_TRIES)

        if spot is None:
            return  # grid lotado: azar

        fx, fy = (a.x, a.y)
        tx, ty = spot

        # move os dois pro mesmo tile (casal)
        self.place(a, tx, ty)
        self.place(b, tx, ty)

        a.home_x, a.home_y = tx, ty
        b.home_x, b.home_y = tx, ty

        # loga como "migração" pra análise e debug
        self.log("migration", {"pid": a.pid, "from": [fx, fy], "to": [tx, ty], "reason": "new_household"})
        self.log("migration", {"pid": b.pid, "from": [fx, fy], "to": [tx, ty], "reason": "new_household"})
        self.log("household_formed", {"a": a.pid, "b": b.pid, "home": [tx, ty]})
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

    def move_person(self, p: Person):
        steps = CHILD_STEPS if p.age < WORK_AGE else (ELDER_STEPS if p.age >= ELDER_AGE else ADULT_STEPS)

        tx, ty = self.target_for_person(p)
        for _ in range(steps):
            dx = 0 if p.x == tx else (1 if tx > p.x else -1)
            dy = 0 if p.y == ty else (1 if ty > p.y else -1)

            # escolhe se anda em x ou y
            if dx != 0 and dy != 0:
                if random.random() < 0.5:
                    nx, ny = p.x + dx, p.y
                else:
                    nx, ny = p.x, p.y + dy
            else:
                nx, ny = p.x + dx, p.y + dy

            # ruído
            if random.random() < 0.15:
                nx = p.x + random.choice([-1,0,1])
                ny = p.y + random.choice([-1,0,1])

            nx = clamp(nx, 0, GRID_W-1)
            ny = clamp(ny, 0, GRID_H-1)

            if (nx, ny) in self.occ:
                return  # travou
            self.place(p, nx, ny)

    def move_couple_unit(self, a: Person, b: Person, steps: int, tx: int, ty: int):
        # casal ocupa mesmo tile; move como unidade
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
                nx = a.x + random.choice([-1,0,1])
                ny = a.y + random.choice([-1,0,1])

            nx = clamp(nx, 0, GRID_W-1)
            ny = clamp(ny, 0, GRID_H-1)

            # tile destino pode estar ocupado por 0 pessoas apenas (o casal não empilha com terceiros)
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
        # média dos centros
        ax, ay = zone_center(PROFESSIONS[a.profession]["zone"])
        bx, by = zone_center(PROFESSIONS[b.profession]["zone"])
        return ((ax+bx)//2, (ay+by)//2)

    def tick_year(self):
        self.births_last = 0
        self.deaths_last = 0

        # 1) envelhecer
        for p in list(self.people.values()):
            p.age += 1

        # 2) escolher profissão quando atingir idade de trabalho
        for p in list(self.people.values()):
            if p.profession is None and p.age >= WORK_AGE:
                # herança cultural: tenta herdar profissão dos pais se existir
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
                    # chance influenciada por dna "soc"
                    soc = (p.dna["soc"] + q.dna["soc"]) / 20.0  # 0.1..1.0
                    candidates.append((q, soc))

            if not candidates:
                continue

            q, soc = random.choice(candidates)
            chance = PAIR_CHANCE_PER_YEAR * (0.6 + 0.8*soc)
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
                    # casal recém-coabitando cria uma nova casa em um tile vazio (espalha a população)
                    self.form_new_household(p, q)
                elif not self.occupied_by_non_couple(q.x, q.y, incoming_pid=p.pid, incoming_partner=q.pid):
                    self.place(p, q.x, q.y)
                    p.cohabiting = True
                    q.cohabiting = True
                    self.log("cohabitation", {"a": p.pid, "b": q.pid, "pos": [q.x, q.y]})
                    # casal recém-coabitando cria uma nova casa em um tile vazio (espalha a população)
                    self.form_new_household(p, q)

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

            # chance extra baseada em fert
            fert = (p.dna["fert"] + q.dna["fert"]) / 20.0  # 0.1..1.0
            chance = BIRTH_CHANCE_PER_YEAR * (0.6 + 0.8*fert)

            if p.kids_count >= MAX_KIDS_PER_COUPLE or q.kids_count >= MAX_KIDS_PER_COUPLE:
                continue

            if random.random() < chance:
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
        pygame.draw.line(surface, GRID_LINE, (x*TILE_SIZE, 0), (x*TILE_SIZE, GRID_H*TILE_SIZE), 1)
    for y in range(GRID_H+1):
        pygame.draw.line(surface, GRID_LINE, (0, y*TILE_SIZE), (GRID_W*TILE_SIZE, y*TILE_SIZE), 1)

    # pessoas
    for p in world.people.values():
        col = (200, 200, 200)
        if p.profession is not None:
            col = PROFESSIONS[p.profession]["color"]
        # morto não aparece (já foi removido)
        cx = p.x*TILE_SIZE + TILE_SIZE//2
        cy = p.y*TILE_SIZE + TILE_SIZE//2
        r = 3 if p.age < WORK_AGE else 4
        pygame.draw.circle(surface, col, (cx, cy), r)

        # se coabitando, desenha um contorno
        if p.cohabiting:
            pygame.draw.circle(surface, (255,255,255), (cx, cy), r+1, 1)

    # tile selecionado
    if selected_tile is not None:
        x,y = selected_tile
        rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(surface, SELECT, rect, 2)

    # HUD
    pop = len(world.people)
    txt = f"Ano: {world.year}  | Pop: {pop}  | Nasc: {world.births_last}  | Mortes: {world.deaths_last}"
    surf = font.render(txt, True, TEXT)
    surface.blit(surf, (8, 8))

def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W*TILE_SIZE, GRID_H*TILE_SIZE))
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
