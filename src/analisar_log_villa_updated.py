from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path

import matplotlib.pyplot as plt


# =========================
# MODELOS
# =========================

@dataclass
class Person:
    id: int
    born_year: Optional[int] = None
    died_year: Optional[int] = None

    mother_id: Optional[int] = None
    father_id: Optional[int] = None

    partners: set[int] = field(default_factory=set)
    children: set[int] = field(default_factory=set)

    professions_by_year: Dict[int, str] = field(default_factory=dict)
    final_profession: Optional[str] = None

    positions_by_year: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # Dados finais (se o JSON tiver final_people)
    dna: Optional[Dict[str, int]] = None
    kids_count: Optional[int] = None
    final_partner: Optional[int] = None

    def is_alive_at(self, year: int) -> bool:
        if self.born_year is not None and year < self.born_year:
            return False
        if self.died_year is not None and year >= self.died_year:
            return False
        return True


# =========================
# 1) ADAPTADOR DO LOG
# =========================

def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_events(doc: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("events", "log", "history", "timeline"):
        if key in doc and isinstance(doc[key], list):
            return doc[key]
    if isinstance(doc, list):
        return doc
    raise ValueError("Não achei a lista de eventos. Ajusta iter_events() pro teu JSON.")


def ev_type(ev: Dict[str, Any]) -> str:
    return ev.get("type") or ev.get("event") or ev.get("name") or ""


def ev_year(ev: Dict[str, Any]) -> int:
    y = ev.get("year")
    if y is None:
        y = ev.get("t")
    if y is None:
        y = ev.get("time")
    if y is None:
        raise ValueError(f"Evento sem ano/tempo: {ev}")
    return int(y)


def ev_data(ev: Dict[str, Any]) -> Dict[str, Any]:
    d = ev.get("data")
    if isinstance(d, dict):
        return d
    return ev


# ---- Getters (adaptados pro teu formato)

def get_person_id(d: Dict[str, Any]) -> Optional[int]:
    # cobre birth (baby), death (pid), profissão (pid), etc.
    for k in (
        "pid",  # <<< TEU LOG USA ISSO
        "baby", "child_id", "child",
        "person_id", "person", "who",
        "dead", "victim", "agent_id",
        "id"
    ):
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except Exception:
                pass



def get_parents(d: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    m = d.get("mother_id", d.get("mother"))
    f = d.get("father_id", d.get("father"))
    return (int(m) if m is not None else None, int(f) if f is not None else None)


def get_partner_pair(d: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    for a_k, b_k in (
        ("a", "b"),
        ("p1", "p2"),
        ("id1", "id2"),
        ("person1_id", "person2_id"),
        ("partner1", "partner2"),
        ("husband", "wife"),
    ):
        if a_k in d and b_k in d:
            return int(d[a_k]), int(d[b_k])

    if "pair" in d and isinstance(d["pair"], list) and len(d["pair"]) == 2:
        return int(d["pair"][0]), int(d["pair"][1])

    return None, None


def get_profession(d: Dict[str, Any]) -> Optional[str]:
    for k in ("profession", "job", "role", "new_profession", "chosen_profession"):
        if k in d and d[k] is not None:
            return str(d[k])
    return None


def get_position(d: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    # teu birth tem "pos": [x,y]
    if "x" in d and "y" in d:
        return int(d["x"]), int(d["y"])
    if "pos" in d and isinstance(d["pos"], list) and len(d["pos"]) == 2:
        return int(d["pos"][0]), int(d["pos"][1])
    return None


# =========================
# 2) CONSTRUÇÃO DO ÍNDICE
# =========================

def get_or_create(people: Dict[int, Person], pid: int) -> Person:
    if pid not in people:
        people[pid] = Person(id=pid)
    return people[pid]


def build_people_index(doc: Dict[str, Any]) -> Tuple[Dict[int, Person], Dict[str, Any]]:
    events = list(iter_events(doc))
    people: Dict[int, Person] = {}

    meta = {
        "first_year": None,
        "last_year": None,
        "event_counts": Counter(),
    }

    for ev in events:
        et = ev_type(ev)
        y = ev_year(ev)
        d = ev_data(ev)

        meta["event_counts"][et] += 1
        meta["first_year"] = y if meta["first_year"] is None else min(meta["first_year"], y)
        meta["last_year"] = y if meta["last_year"] is None else max(meta["last_year"], y)

        if et == "birth":
            # teu birth: {"baby": id, "mother": id, "father": id, "pos":[x,y]}
            baby = d.get("baby")
            if baby is None:
                # fallback genérico
                baby = get_person_id(d)
            if baby is None:
                continue

            child_id = int(baby)
            child = get_or_create(people, child_id)
            child.born_year = y if child.born_year is None else min(child.born_year, y)

            m, f = get_parents(d)
            child.mother_id, child.father_id = m, f

            if m is not None:
                mom = get_or_create(people, m)
                mom.children.add(child_id)
            if f is not None:
                dad = get_or_create(people, f)
                dad.children.add(child_id)

            # se quiser registrar posição de nascimento como "posição naquele ano"
            pos = get_position(d)
            if pos is not None:
                child.positions_by_year[y] = pos

        elif et == "death":
            pid = get_person_id(d)
            if pid is None:
                continue
            p = get_or_create(people, pid)
            p.died_year = y if p.died_year is None else min(p.died_year, y)

        elif et in ("pair_formed", "cohabitation", "marriage"):
            a, b = get_partner_pair(d)
            if a is None or b is None:
                continue
            pa = get_or_create(people, a)
            pb = get_or_create(people, b)
            pa.partners.add(b)
            pb.partners.add(a)

        elif et == "profession_chosen":
            pid = get_person_id(d)
            prof = get_profession(d)
            if pid is None or prof is None:
                continue
            p = get_or_create(people, pid)
            p.professions_by_year[y] = str(prof)
            p.final_profession = str(prof)

        elif et in ("migration", "move", "moved"):
            pid = get_person_id(d)
            pos = get_position(d)
            if pid is None or pos is None:
                continue
            p = get_or_create(people, pid)
            p.positions_by_year[y] = pos

        else:
            pass

    # Finaliza profissão final se tiver histórico mas não tiver final
    for p in people.values():
        if p.final_profession is None and p.professions_by_year:
            last_y = max(p.professions_by_year.keys())
            p.final_profession = p.professions_by_year[last_y]


    # --- Enriquecimento com final_people (se existir) ---
    # O log de eventos nem sempre carrega DNA/kids_count/born_year de todo mundo (ex: população inicial).
    # Se o JSON tiver "final_people", usamos como fonte de verdade para esses campos finais.
    final_list = doc.get("final_people")
    if isinstance(final_list, list):
        for fp in final_list:
            try:
                pid = fp.get("pid", fp.get("id"))
                if pid is None:
                    continue
                pid = int(pid)
            except Exception:
                continue

            p = get_or_create(people, pid)

            by = fp.get("born_year")
            dy = fp.get("died_year")
            if by is not None:
                try:
                    p.born_year = int(by)
                except Exception:
                    pass
            if dy is not None:
                try:
                    p.died_year = int(dy)
                except Exception:
                    pass

            # pais
            m = fp.get("mother", fp.get("mother_id"))
            f = fp.get("father", fp.get("father_id"))
            try:
                p.mother_id = int(m) if m is not None else p.mother_id
            except Exception:
                pass
            try:
                p.father_id = int(f) if f is not None else p.father_id
            except Exception:
                pass

            # profissão/estado final
            prof = fp.get("profession")
            if prof is not None:
                p.final_profession = str(prof)

            partner = fp.get("partner")
            if partner is not None:
                try:
                    p.final_partner = int(partner)
                except Exception:
                    pass

            kc = fp.get("kids_count")
            if kc is not None:
                try:
                    p.kids_count = int(kc)
                except Exception:
                    pass

            dna = fp.get("dna")
            if isinstance(dna, dict):
                # normaliza para int quando possível
                out = {}
                for k, v in dna.items():
                    try:
                        out[str(k)] = int(v)
                    except Exception:
                        continue
                p.dna = out

            # posição final (se existir)
            pos = fp.get("pos")
            if isinstance(pos, list) and len(pos) == 2:
                try:
                    last_y = meta["last_year"] if meta["last_year"] is not None else 0
                    p.positions_by_year[last_y] = (int(pos[0]), int(pos[1]))
                except Exception:
                    pass

    return people, meta


# =========================
# 3) ANÁLISES
# =========================

def profession_mobility(people: Dict[int, Person]) -> Dict[str, Any]:
    total_children = 0
    comparable = 0
    moved = 0
    same_as_parent = 0
    unknown_parents = 0

    transitions = Counter()  # (parent_prof -> child_prof)

    for p in people.values():
        if p.born_year is None:
            continue
        total_children += 1

        child_prof = p.final_profession
        if p.mother_id is None and p.father_id is None:
            unknown_parents += 1
            continue

        parent_profs = []
        for parent_id in (p.mother_id, p.father_id):
            if parent_id is None:
                continue
            parent = people.get(parent_id)
            if parent and parent.final_profession:
                parent_profs.append(parent.final_profession)

        if not parent_profs or child_prof is None:
            continue

        comparable += 1
        if child_prof in parent_profs:
            same_as_parent += 1
        else:
            moved += 1

        for pp in parent_profs:
            transitions[(pp, child_prof)] += 1

    rate = (moved / comparable) if comparable else 0.0
    return {
        "children_born": total_children,
        "comparable": comparable,
        "unknown_parents": unknown_parents,
        "moved_diff_from_parents": moved,
        "same_as_parent": same_as_parent,
        "mobility_rate": rate,
        "top_transitions": transitions.most_common(20),
    }


def migrations_stats(people: Dict[int, Person]) -> Dict[str, Any]:
    total_moves = 0
    movers = 0
    moves_per_person = []
    moves_by_year = Counter()

    for p in people.values():
        if not p.positions_by_year:
            continue

        years_sorted = sorted(p.positions_by_year.keys())
        last_pos = None
        count = 0

        for y in years_sorted:
            pos = p.positions_by_year[y]
            if last_pos is None:
                last_pos = pos
                continue
            if pos != last_pos:
                count += 1
                total_moves += 1
                moves_by_year[y] += 1
                last_pos = pos

        if count > 0:
            movers += 1
            moves_per_person.append(count)

    avg = (sum(moves_per_person) / len(moves_per_person)) if moves_per_person else 0.0
    return {
        "total_migrations": total_moves,
        "people_who_migrated": movers,
        "avg_migrations_per_mover": avg,
        "top_years": moves_by_year.most_common(15),
    }


def descendants_of_couple(
    people: Dict[int, Person],
    mother: Optional[int],
    father: Optional[int],
) -> set[int]:
    if mother is None or father is None:
        return set()
    if mother not in people or father not in people:
        return set()

    # filhos em comum
    start = people[mother].children & people[father].children
    if not start:
        return set()

    visited = set()
    queue = list(start)

    while queue:
        pid = queue.pop()
        if pid in visited:
            continue
        visited.add(pid)

        p = people.get(pid)
        if p:
            queue.extend(p.children)

    return visited


def peak_alive(
    people: Dict[int, Person],
    members: set[int],
    first_year: int,
    last_year: int,
) -> tuple[int, Optional[int]]:
    peak = 0
    peak_year = None
    for y in range(first_year, last_year + 1):
        alive = sum(1 for m in members if people[m].is_alive_at(y))
        if alive > peak:
            peak = alive
            peak_year = y
    return peak, peak_year


def family_lineages(people: Dict[int, Person], last_year: int) -> Dict[str, Any]:
    extinct = 0
    alive = 0

    # todos os casais que aparecem como pais
    couples = set()
    for p in people.values():
        if p.mother_id is not None and p.father_id is not None:
            couples.add((p.mother_id, p.father_id))

    results = []

    for mother, father in couples:
        members = descendants_of_couple(people, mother, father)
        if not members:
            continue

        born_years = [
            people[m].born_year
            for m in members
            if people[m].born_year is not None
        ]
        start_year = min(born_years) if born_years else None

        last_alive_year = None
        alive_at_end = False

        for m in members:
            pm = people[m]
            if pm.is_alive_at(last_year):
                alive_at_end = True

            end = (pm.died_year - 1) if pm.died_year is not None else last_year
            last_alive_year = end if last_alive_year is None else max(last_alive_year, end)

        if alive_at_end:
            alive += 1
        else:
            extinct += 1

        duration = None
        if start_year is not None and last_alive_year is not None:
            duration = max(0, last_alive_year - start_year)

        peak, peak_year = peak_alive(people, members, start_year or 0, last_year)

        results.append({
            "family_key": (mother, father),
            "members_total_ever": len(members),      # 👈 total histórico (desde o início)
            "born_count": len(born_years),
            "start_year": start_year,
            "last_alive_year": last_alive_year,
            "duration_years": duration,
            "alive_at_end": alive_at_end,
            "peak_alive": peak,
            "peak_alive_year": peak_year,
        })

    by_duration = sorted(
        [r for r in results if r["duration_years"] is not None],
        key=lambda x: x["duration_years"],
        reverse=True,
    )

    by_size = sorted(
        results,
        key=lambda x: x["members_total_ever"],
        reverse=True,
    )

    return {
        "total_lineages": len(results),
        "extinct_lineages": extinct,
        "alive_lineages": alive,
        "top10_longest": by_duration[:10],
        "top10_biggest": by_size[:10],
        "all_lineages": results,
    }



# =========================
# 3.5) SÉRIES TEMPORAIS + GRÁFICOS
# =========================

def build_yearly_state_series(doc: Dict[str, Any], people: Dict[int, Person], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstrói estado ano-a-ano a partir dos eventos (e, quando existir, final_people).
    Importante: não existe "divórcio" no modelo atual, então:
      - parceria pode acabar só por morte
      - pode haver novo par após viuvez (a reconstrução abaixo suporta)
    """
    events = list(iter_events(doc))
    first_year = int(meta.get("first_year") or 0)
    last_year = int(meta.get("last_year") or 0)

    # agrupa eventos por ano, mantendo ordem original (porque a vida não tem rollback)
    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        y = ev_year(ev)
        by_year[y].append(ev)

    # born_year vindo do índice (já enriquecido com final_people)
    born_year: Dict[int, int] = {}
    for pid, p in people.items():
        if p.born_year is not None:
            born_year[pid] = int(p.born_year)

    # estado dinâmico
    alive: Dict[int, bool] = {pid: True for pid in born_year.keys()}
    partner: Dict[int, Optional[int]] = {pid: None for pid in born_year.keys()}

    # aplica mortos antes do início (caso exista)
    for pid, p in people.items():
        if p.died_year is not None and p.died_year <= first_year:
            alive[pid] = False

    # séries
    years = list(range(first_year, last_year + 1))
    alive_by_year = []
    singles_by_year = []
    singles_rate_by_year = []
    cross_pairs_by_year = []
    cross_pairs_rate_by_year = []
    avg_first_marriage_age_by_year = []

    # primeiro casamento por pessoa
    first_marriage_year: Dict[int, int] = {}

    for y in years:
        cross_pairs = 0
        total_pairs = 0
        marriage_ages_this_year = []

        for ev in by_year.get(y, []):
            et = ev_type(ev)
            d = ev_data(ev)

            if et == "birth":
                baby = d.get("baby", get_person_id(d))
                if baby is None:
                    continue
                baby = int(baby)
                born_year[baby] = y
                alive[baby] = True
                partner.setdefault(baby, None)

            elif et == "death":
                pid = get_person_id(d)
                if pid is None:
                    continue
                pid = int(pid)
                alive[pid] = False
                pr = partner.get(pid)
                if pr is not None:
                    partner[pr] = None
                partner[pid] = None

            elif et == "pair_formed":
                a, b = get_partner_pair(d)
                if a is None or b is None:
                    continue
                a, b = int(a), int(b)

                total_pairs += 1
                if bool(d.get("cross_zone")):
                    cross_pairs += 1

                if alive.get(a, True) and alive.get(b, True):
                    partner[a] = b
                    partner[b] = a

            elif et == "marriage":
                a, b = get_partner_pair(d)
                if a is None or b is None:
                    continue
                a, b = int(a), int(b)
                for pid in (a, b):
                    if pid not in first_marriage_year:
                        first_marriage_year[pid] = y
                        by = born_year.get(pid)
                        if by is not None:
                            marriage_ages_this_year.append(y - by)

        alive_count = sum(
            1 for pid, ok in alive.items()
            if ok and (born_year.get(pid, y + 1) <= y)
        )
        alive_by_year.append(alive_count)

        cfg = doc.get("config", {}) if isinstance(doc.get("config"), dict) else {}
        repro_min = int(cfg.get("REPRO_MIN_AGE", 18))
        repro_max = int(cfg.get("REPRO_MAX_AGE", 40))

        singles = 0
        for pid, ok in alive.items():
            if not ok:
                continue
            by = born_year.get(pid)
            if by is None or by > y:
                continue
            age = y - by
            if not (repro_min <= age <= repro_max):
                continue
            if partner.get(pid) is None:
                singles += 1

        singles_by_year.append(singles)
        singles_rate_by_year.append((singles / alive_count) if alive_count else 0.0)
        cross_pairs_by_year.append(cross_pairs)
        cross_pairs_rate_by_year.append((cross_pairs / total_pairs) if total_pairs else 0.0)
        avg_first_marriage_age_by_year.append(
            (sum(marriage_ages_this_year) / len(marriage_ages_this_year)) if marriage_ages_this_year else 0.0
        )

    return {
        "years": years,
        "alive": alive_by_year,
        "singles": singles_by_year,
        "singles_rate": singles_rate_by_year,
        "cross_pairs": cross_pairs_by_year,
        "cross_pairs_rate": cross_pairs_rate_by_year,
        "avg_first_marriage_age": avg_first_marriage_age_by_year,
        "first_marriage_year": first_marriage_year,
    }


def plot_new_metrics(doc: Dict[str, Any], people: Dict[int, Person], meta: Dict[str, Any]) -> None:
    series = build_yearly_state_series(doc, people, meta)
    years = series["years"]

    plt.figure()
    plt.plot(years, series["alive"], label="Vivos")
    plt.plot(years, series["singles"], label="Solteiros (janela fértil)")
    plt.title("Vivos vs solteiros (janela fértil)")
    plt.xlabel("Ano")
    plt.ylabel("Pessoas")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(years, series["singles_rate"])
    plt.title("% solteiros (janela fértil)")
    plt.xlabel("Ano")
    plt.ylabel("Proporção")
    plt.grid(True)

    plt.figure()
    plt.plot(years, series["cross_pairs_rate"])
    plt.title("Taxa de pares formados fora do bairro (cross_zone)")
    plt.xlabel("Ano")
    plt.ylabel("Proporção")
    plt.grid(True)

    plt.figure()
    plt.plot(years, series["avg_first_marriage_age"])
    plt.title("Idade média no 1º casamento (por ano)")
    plt.xlabel("Ano")
    plt.ylabel("Idade (anos)")
    plt.grid(True)

    couples_kids = Counter()
    for ev in iter_events(doc):
        if ev_type(ev) == "birth":
            d = ev_data(ev)
            m = d.get("mother")
            f = d.get("father")
            if m is None or f is None:
                continue
            try:
                key = tuple(sorted((int(m), int(f))))
            except Exception:
                continue
            couples_kids[key] += 1

    if couples_kids:
        plt.figure()
        plt.hist(list(couples_kids.values()), bins=range(0, max(couples_kids.values()) + 2))
        plt.title("Distribuição: filhos por casal")
        plt.xlabel("Filhos")
        plt.ylabel("Número de casais")
        plt.grid(True)

    genes = ["speed", "fert", "soc", "vit"]
    xs = {g: [] for g in genes}
    ys = []

    for p in people.values():
        if not p.dna or not isinstance(p.dna, dict):
            continue

        kids = None
        if p.kids_count is not None:
            kids = p.kids_count
        elif p.children:
            kids = len(p.children)

        if kids is None:
            continue

        ok = True
        for g in genes:
            if g not in p.dna:
                ok = False
                break
        if not ok:
            continue

        ys.append(kids)
        for g in genes:
            xs[g].append(p.dna[g])

    def pearson(a: List[float], b: List[float]) -> float:
        n = len(a)
        if n < 2:
            return 0.0
        ma = sum(a) / n
        mb = sum(b) / n
        num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
        da = sum((ai - ma) ** 2 for ai in a)
        db = sum((bi - mb) ** 2 for bi in b)
        if da <= 0 or db <= 0:
            return 0.0
        return num / ((da ** 0.5) * (db ** 0.5))

    if ys:
        for g in genes:
            r = pearson(xs[g], ys)
            plt.figure()
            plt.scatter(xs[g], ys)
            plt.title(f"DNA vs filhos: {g} (r={r:.2f})")
            plt.xlabel(g)
            plt.ylabel("Filhos (kids_count)")
            plt.grid(True)

    plt.show()

# =========================
# 4) CONSULTA POR ID + ÁRVORE ASCII
# =========================

def person_summary(people: Dict[int, Person], pid: int, last_year: Optional[int] = None) -> str:
    p = people.get(pid)
    if not p:
        return f"Pessoa {pid} não existe no índice."

    alive = None
    if last_year is not None:
        alive = p.is_alive_at(last_year)

    lines = []
    lines.append(f"Pessoa {p.id}")
    lines.append(f"  Nasc.: {p.born_year} | Morte: {p.died_year} | Viva no fim: {alive}")
    lines.append(f"  Pais: mãe={p.mother_id} pai={p.father_id}")
    lines.append(f"  Parceiros: {sorted(p.partners) if p.partners else []}")
    lines.append(f"  Filhos: {sorted(p.children) if p.children else []}")
    lines.append(f"  Profissão final: {p.final_profession}")
    if p.professions_by_year:
        last5 = sorted(p.professions_by_year.items())[-5:]
        lines.append(f"  Profissões (últimos 5 registros): {last5}")
    if p.positions_by_year:
        last5p = sorted(p.positions_by_year.items())[-5:]
        lines.append(f"  Posições (últimos 5 registros): {last5p}")
    return "\n".join(lines)


def ascii_tree(people: Dict[int, Person], root_id: int, depth: int = 3) -> str:
    if root_id not in people:
        return f"(ID {root_id} não existe)"

    def fmt(pid: Optional[int]) -> str:
        if pid is None or pid not in people:
            return str(pid)
        p = people[pid]
        prof = p.final_profession or "?"
        return f"{pid}({prof})"

    def build_desc(pid: int, d: int, prefix: str = "") -> List[str]:
        if d <= 0:
            return []
        p = people[pid]
        kids = sorted(list(p.children))
        out = []
        for i, k in enumerate(kids):
            last = (i == len(kids) - 1)
            branch = "└── " if last else "├── "
            out.append(prefix + branch + fmt(k))
            out.extend(build_desc(k, d - 1, prefix + ("    " if last else "│   ")))
        return out

    p = people[root_id]
    lines = []
    lines.append(f"Raiz: {fmt(root_id)}")
    lines.append("Pais:")
    lines.append(f"  mãe: {fmt(p.mother_id)}")
    lines.append(f"  pai: {fmt(p.father_id)}")
    lines.append("Filhos/descendentes:")
    lines.extend(build_desc(root_id, depth))
    return "\n".join(lines)


# =========================
# 5) EXPORT CSV
# =========================

def export_people_csv(people: Dict[int, Person], out_path: str | Path) -> None:
    out_path = Path(out_path)
    lines = ["id,born_year,died_year,mother_id,father_id,final_profession,partners,children"]
    for p in sorted(people.values(), key=lambda x: x.id):
        partners = ";".join(map(str, sorted(p.partners)))
        children = ";".join(map(str, sorted(p.children)))
        lines.append(
            f"{p.id},{p.born_year},{p.died_year},{p.mother_id},{p.father_id},"
            f"{(p.final_profession or '')},{partners},{children}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# 6) CLI
# =========================

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Analisador do log (famílias, migração, mobilidade, árvore).")
    ap.add_argument("json_path", help="Caminho do sim_log_*.json")
    ap.add_argument("--person", type=int, default=None, help="Mostra resumo de uma pessoa por ID")
    ap.add_argument("--tree", type=int, default=None, help="Mostra árvore ASCII a partir de um ID")
    ap.add_argument("--depth", type=int, default=3, help="Profundidade da árvore ASCII (default=3)")
    ap.add_argument("--export_csv", default=None, help="Exporta CSV com tabela de pessoas")
    args = ap.parse_args()

    doc = load_json(args.json_path)
    people, meta = build_people_index(doc)
    last_year = meta["last_year"] if meta["last_year"] is not None else 0

    print("\n=== META ===")
    print(f"Ano inicial: {meta['first_year']} | Ano final: {meta['last_year']}")
    print("Eventos por tipo (top 20):")
    for k, v in meta["event_counts"].most_common(20):
        print(f"  {k}: {v}")

    print("\n=== POPULAÇÃO INDEXADA ===")
    print(f"Pessoas no índice: {len(people)}")

    # debug útil (porque o mundo mente)
    died = sum(1 for p in people.values() if p.died_year is not None)
    profs = sum(1 for p in people.values() if p.final_profession is not None)
    print(f"[DEBUG] Pessoas com died_year preenchido: {died}")
    print(f"[DEBUG] Pessoas com profissão final: {profs}")

    mob = profession_mobility(people)
    print("\n=== MOBILIDADE DE PROFISSÃO (filhos vs pais) ===")
    print(f"Crianças nascidas: {mob['children_born']}")
    print(f"Comparáveis (pais+prof conhecidos): {mob['comparable']}")
    print(f"Sem pais conhecidos: {mob['unknown_parents']}")
    print(f"Diferente dos pais: {mob['moved_diff_from_parents']} | Igual a algum dos pais: {mob['same_as_parent']}")
    print(f"Taxa de mobilidade: {mob['mobility_rate']:.3f}")
    print("Top transições (pai->filho):")
    for (pp, cp), n in mob["top_transitions"]:
        print(f"  {pp} -> {cp}: {n}")

    mig = migrations_stats(people)
    print("\n=== MIGRAÇÕES ===")
    print(f"Total migrações: {mig['total_migrations']}")
    print(f"Pessoas que migraram: {mig['people_who_migrated']}")
    print(f"Média de migrações por migrante: {mig['avg_migrations_per_mover']:.2f}")
    print("Anos com mais migrações:")
    for y, n in mig["top_years"]:
        print(f"  Ano {y}: {n}")

    fam = family_lineages(people, last_year=last_year)
    print("\n=== FAMÍLIAS / LINHAGENS (descendência real do casal fundador) ===")
    print(f"Total linhagens: {fam['total_lineages']}")
    print(f"Ativas no final: {fam['alive_lineages']} | Extintas: {fam['extinct_lineages']}")

    print("\nTop 10 mais duradouras:")
    for r in fam["top10_longest"]:
        print(
            f"  {r['family_key']} | duração={r['duration_years']} | "
            f"total_ever={r['members_total_ever']} | pico={r['peak_alive']} "
            f"(ano {r['peak_alive_year']}) | viva_no_fim={r['alive_at_end']}"
        )

    print("\nTop 10 maiores (histórico):")
    for r in fam["top10_biggest"]:
        print(
            f"  {r['family_key']} | total_ever={r['members_total_ever']} | "
            f"duração={r['duration_years']} | viva_no_fim={r['alive_at_end']}"
        )


    # =========================
    # GRÁFICOS (novas métricas)
    # =========================
    try:
        plot_new_metrics(doc, people, meta)
    except Exception as ex:
        # Se teu matplotlib estiver em backend "Agg" ou sem GUI, ele não abre janela.
        # Pelo menos não vamos falhar silenciosamente.
        print("\n[AVISO] Não consegui abrir os gráficos (backend/headless?).")
        print("        Erro:", ex)
        print("        Dica: no Windows, tenta:  python -m pip install pyqt6  (ou instala o Python com Tk).")

    if args.export_csv:
        export_people_csv(people, args.export_csv)
        print(f"\nCSV exportado em: {args.export_csv}")

    if args.person is not None:
        print("\n=== PESSOA (RESUMO) ===")
        print(person_summary(people, args.person, last_year=last_year))

    if args.tree is not None:
        print("\n=== ÁRVORE ASCII ===")
        print(ascii_tree(people, args.tree, depth=args.depth))


if __name__ == "__main__":
    main()
