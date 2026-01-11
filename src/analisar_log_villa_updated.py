from __future__ import annotations

import json
import math
from statistics import mean, median
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
    return None


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
            pa = get_or_create(people, int(a))
            pb = get_or_create(people, int(b))
            pa.partners.add(int(b))
            pb.partners.add(int(a))

        elif et == "profession_chosen":
            pid = get_person_id(d)
            prof = get_profession(d)
            if pid is None or prof is None:
                continue
            p = get_or_create(people, int(pid))
            p.professions_by_year[y] = str(prof)
            p.final_profession = str(prof)

        elif et in ("migration", "move", "moved"):
            pid = get_person_id(d)
            pos = get_position(d)
            if pid is None or pos is None:
                continue
            p = get_or_create(people, int(pid))
            p.positions_by_year[y] = pos

    # Finaliza profissão final se tiver histórico mas não tiver final
    for p in people.values():
        if p.final_profession is None and p.professions_by_year:
            last_y = max(p.professions_by_year.keys())
            p.final_profession = p.professions_by_year[last_y]

    # --- Enriquecimento com final_people (se existir) ---
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
                out = {}
                for k, v in dna.items():
                    try:
                        out[str(k)] = int(v)
                    except Exception:
                        continue
                p.dna = out

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
            "members_total_ever": len(members),
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
    events = list(iter_events(doc))
    first_year = int(meta.get("first_year") or 0)
    last_year = int(meta.get("last_year") or 0)

    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        y = ev_year(ev)
        by_year[y].append(ev)

    born_year: Dict[int, int] = {}
    for pid, p in people.items():
        if p.born_year is not None:
            born_year[pid] = int(p.born_year)

    alive: Dict[int, bool] = {pid: True for pid in born_year.keys()}
    partner: Dict[int, Optional[int]] = {pid: None for pid in born_year.keys()}

    for pid, p in people.items():
        if p.died_year is not None and p.died_year <= first_year:
            alive[pid] = False

    years = list(range(first_year, last_year + 1))
    alive_by_year = []
    singles_by_year = []
    singles_rate_by_year = []
    cross_pairs_by_year = []
    cross_pairs_rate_by_year = []
    avg_first_marriage_age_by_year = []

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

    # ===== Extras (idade fértil, DNA no tempo, profissões e fitness) =====
    try:
        plot_births_by_age_bins(doc, people, meta)
        plot_genes_by_birth_cohort(people, meta)
        plot_profession_chosen_per_year(doc, meta)
        plot_dna_by_profession(people)
        plot_fitness_by_profession(people, meta)
    except Exception as ex:
        print("[AVISO] Falhou ao gerar gráficos extras:", ex)

    plt.show()



# =========================
# 3.7) EXTRAS: NATALIDADE POR FAIXA ETÁRIA + DNA NO TEMPO + PROFISSÕES + FITNESS
# =========================

def _age_bin(age: int, bins: List[Tuple[int, int]]) -> Optional[str]:
    for a0, a1 in bins:
        if a0 <= age <= a1:
            return f"{a0}-{a1}"
    return None


def plot_births_by_age_bins(doc: Dict[str, Any], people: Dict[int, Person], meta: Dict[str, Any]) -> None:
    """Nascimentos por ano por faixa etária da mãe e do pai (usa born_year do índice)."""
    bins = [(18, 22), (23, 27), (28, 32), (33, 37), (38, 42), (43, 49)]

    mother_counts: Dict[int, Counter] = defaultdict(Counter)
    father_counts: Dict[int, Counter] = defaultdict(Counter)

    for ev in iter_events(doc):
        if ev_type(ev) != "birth":
            continue
        y = ev_year(ev)
        d = ev_data(ev)

        m = d.get("mother")
        f = d.get("father")
        if m is None or f is None:
            continue

        try:
            m = int(m)
            f = int(f)
        except Exception:
            continue

        pm = people.get(m)
        pf = people.get(f)
        if not pm or not pf or pm.born_year is None or pf.born_year is None:
            continue

        am = y - int(pm.born_year)
        af = y - int(pf.born_year)

        bm = _age_bin(am, bins)
        bf = _age_bin(af, bins)
        if bm:
            mother_counts[y][bm] += 1
        if bf:
            father_counts[y][bf] += 1

    first_year = int(meta.get("first_year") or 0)
    last_year = int(meta.get("last_year") or first_year)
    years = list(range(first_year, last_year + 1))
    labels = [f"{a}-{b}" for a, b in bins]

    plt.figure()
    for lab in labels:
        ys = [mother_counts[y][lab] for y in years]
        plt.plot(years, ys, label=f"Mãe {lab}")
    plt.title("Nascimentos por ano por faixa etária (mãe)")
    plt.xlabel("Ano")
    plt.ylabel("Nascimentos/ano")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for lab in labels:
        ys = [father_counts[y][lab] for y in years]
        plt.plot(years, ys, label=f"Pai {lab}")
    plt.title("Nascimentos por ano por faixa etária (pai)")
    plt.xlabel("Ano")
    plt.ylabel("Nascimentos/ano")
    plt.grid(True)
    plt.legend()


def plot_genes_by_birth_cohort(people: Dict[int, Person], meta: Dict[str, Any]) -> None:
    """DNA médio por coorte de nascimento (usa DNA disponível em final_people).
    Observação: se o DNA só existe no dump final, isso mede só quem tem DNA no final (viés de sobrevivência).
    """
    genes = ["speed", "fert", "soc", "vit"]
    BIN = 20  # anos por coorte

    cohorts: Dict[int, List[Dict[str, int]]] = defaultdict(list)
    for p in people.values():
        if p.born_year is None or not p.dna:
            continue
        if any(g not in p.dna for g in genes):
            continue
        c0 = (int(p.born_year) // BIN) * BIN
        cohorts[c0].append(p.dna)

    if not cohorts:
        print("[AVISO] Sem DNA suficiente pra plotar genes por coorte (final_people não trouxe dna?).")
        return

    xs = sorted(cohorts.keys())

    for g in genes:
        ys = []
        for c0 in xs:
            vals = [dna[g] for dna in cohorts[c0] if g in dna]
            ys.append(sum(vals) / len(vals) if vals else 0.0)

        plt.figure()
        plt.plot(xs, ys)
        plt.title(f"DNA médio por coorte de nascimento ({BIN} anos): {g}")
        plt.xlabel("Ano (início da coorte)")
        plt.ylabel("Média do gene")
        plt.grid(True)

    plt.figure()
    plt.plot(xs, [len(cohorts[c0]) for c0 in xs])
    plt.title(f"Amostra com DNA por coorte ({BIN} anos)")
    plt.xlabel("Ano (início da coorte)")
    plt.ylabel("N pessoas (com DNA)")
    plt.grid(True)


def plot_profession_chosen_per_year(doc: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Fluxo: quantas escolhas de profissão por ano (profession_chosen)."""
    by_year_prof: Dict[int, Counter] = defaultdict(Counter)

    for ev in iter_events(doc):
        if ev_type(ev) != "profession_chosen":
            continue
        y = ev_year(ev)
        d = ev_data(ev)
        prof = get_profession(d)
        if not prof:
            continue
        by_year_prof[y][prof] += 1

    first_year = int(meta.get("first_year") or 0)
    last_year = int(meta.get("last_year") or first_year)
    years = list(range(first_year, last_year + 1))
    all_profs = sorted({p for y in by_year_prof for p in by_year_prof[y].keys()})

    if not all_profs:
        print("[AVISO] Não achei nenhum evento profession_chosen pra plotar.")
        return

    plt.figure()
    for prof in all_profs:
        ys = [by_year_prof[y][prof] for y in years]
        plt.plot(years, ys, label=prof)

    plt.title("Profissão escolhida por ano (fluxo)")
    plt.xlabel("Ano")
    plt.ylabel("Escolhas/ano")
    plt.grid(True)
    plt.legend()


def plot_dna_by_profession(people: Dict[int, Person]) -> None:
    """DNA médio por profissão final (usa DNA disponível em final_people)."""
    genes = ["speed", "fert", "soc", "vit"]
    buckets: Dict[str, List[Dict[str, int]]] = defaultdict(list)

    for p in people.values():
        if not p.final_profession or not p.dna:
            continue
        if any(g not in p.dna for g in genes):
            continue
        buckets[p.final_profession].append(p.dna)

    if not buckets:
        print("[AVISO] Sem dados suficientes de DNA+profissão (talvez final_people não tem dna?).")
        return

    profs = sorted(buckets.keys())

    for g in genes:
        means = []
        ns = []
        for prof in profs:
            vals = [dna[g] for dna in buckets[prof] if g in dna]
            means.append(sum(vals) / len(vals) if vals else 0.0)
            ns.append(len(vals))

        plt.figure()
        plt.plot(range(len(profs)), means, marker="o")
        plt.title(f"DNA médio por profissão: {g}")
        plt.xlabel("Profissão (índice)")
        plt.ylabel("Média do gene")
        plt.grid(True)

        print(f"\nDNA médio por profissão ({g}):")
        for i, prof in enumerate(profs):
            print(f"  {i:02d} {prof}: {means[i]:.3f} (n={ns[i]})")


# ---------- FITNESS (reprodução + sobrevivência) ----------

def _safe_mean(xs: List[Optional[float]]) -> float:
    xs2 = [x for x in xs if x is not None]
    return mean(xs2) if xs2 else 0.0


def _safe_median(xs: List[Optional[float]]) -> float:
    xs2 = [x for x in xs if x is not None]
    return median(xs2) if xs2 else 0.0


def _person_kids(p: Person) -> int:
    if p.kids_count is not None:
        return int(p.kids_count)
    if p.children:
        return len(p.children)
    return 0


def _person_lifespan(p: Person, last_year: int) -> Optional[int]:
    if p.born_year is None:
        return None
    by = int(p.born_year)
    if p.died_year is not None:
        return max(0, int(p.died_year) - by)
    return max(0, int(last_year) - by)


def compute_fitness_tables(people: Dict[int, Person], last_year: int):
    """Retorna:
      - per_prof: dict[prof] -> listas de kids / lifespan / fitness
      - per_person_rows: ranking por pessoa
    """
    rows = []
    for p in people.values():
        prof = p.final_profession
        if not prof:
            continue
        kids = _person_kids(p)
        life = _person_lifespan(p, last_year)
        if life is None:
            continue
        rows.append((p.id, prof, kids, life))

    if not rows:
        return {}, []

    kids_all = [r[2] for r in rows]
    life_all = [r[3] for r in rows]

    k_min, k_max = min(kids_all), max(kids_all)
    l_min, l_max = min(life_all), max(life_all)

    def norm(x, mn, mx):
        if mx <= mn:
            return 0.0
        return (x - mn) / (mx - mn)

    # pesos (ajusta se quiser)
    w_kids = 0.65
    w_life = 0.35

    per_prof = {}
    per_person_rows = []

    for pid, prof, kids, life in rows:
        nk = norm(kids, k_min, k_max)
        nl = norm(life, l_min, l_max)
        fitness = w_kids * nk + w_life * nl  # [0..1]

        per_person_rows.append({
            "pid": pid,
            "prof": prof,
            "kids": kids,
            "lifespan": life,
            "fitness": fitness,
        })

        if prof not in per_prof:
            per_prof[prof] = {"kids": [], "lifespan": [], "fitness": []}
        per_prof[prof]["kids"].append(kids)
        per_prof[prof]["lifespan"].append(life)
        per_prof[prof]["fitness"].append(fitness)

    return per_prof, per_person_rows


def print_fitness_report(people: Dict[int, Person], meta: Dict[str, Any], top_n: int = 15) -> None:
    last_year = int(meta.get("last_year") or 0)
    per_prof, per_person = compute_fitness_tables(people, last_year)

    if not per_prof:
        print("\n[AVISO] Não consegui montar fitness (faltou born_year/died_year?).")
        return

    print("\n=== FITNESS (reprodução + longevidade) POR PROFISSÃO ===")
    prof_rows = []
    for prof, d in per_prof.items():
        prof_rows.append({
            "prof": prof,
            "n": len(d["fitness"]),
            "kids_mean": _safe_mean(d["kids"]),
            "kids_median": _safe_median(d["kids"]),
            "life_mean": _safe_mean(d["lifespan"]),
            "life_median": _safe_median(d["lifespan"]),
            "fitness_mean": _safe_mean(d["fitness"]),
            "fitness_median": _safe_median(d["fitness"]),
        })

    prof_rows.sort(key=lambda r: r["fitness_mean"], reverse=True)
    for r in prof_rows:
        print(
            f"  {r['prof']}: n={r['n']} | kids μ={r['kids_mean']:.2f} (med={r['kids_median']:.0f}) | "
            f"vida μ={r['life_mean']:.2f} (med={r['life_median']:.0f}) | "
            f"fitness μ={r['fitness_mean']:.3f} (med={r['fitness_median']:.3f})"
        )

    per_person.sort(key=lambda x: x["fitness"], reverse=True)
    print(f"\n=== TOP {top_n} INDIVÍDUOS POR FITNESS ===")
    for r in per_person[:top_n]:
        print(f"  ID {r['pid']} | {r['prof']:<8} | kids={r['kids']:<2} | vida={r['lifespan']:<3} | fitness={r['fitness']:.3f}")


def plot_fitness_by_profession(people: Dict[int, Person], meta: Dict[str, Any]) -> None:
    last_year = int(meta.get("last_year") or 0)
    per_prof, _ = compute_fitness_tables(people, last_year)
    if not per_prof:
        return

    profs = sorted(per_prof.keys())
    kids_mean = [_safe_mean(per_prof[p]["kids"]) for p in profs]
    life_mean = [_safe_mean(per_prof[p]["lifespan"]) for p in profs]
    fit_mean = [_safe_mean(per_prof[p]["fitness"]) for p in profs]

    plt.figure()
    plt.plot(range(len(profs)), kids_mean, marker="o")
    plt.title("Média de filhos por profissão")
    plt.xlabel("Profissão (índice)")
    plt.ylabel("Filhos (média)")
    plt.grid(True)
    print("\n[ÍNDICES PROFISSÃO] (pra ler os gráficos)")
    for i, p in enumerate(profs):
        print(f"  {i:02d} -> {p}")

    plt.figure()
    plt.plot(range(len(profs)), life_mean, marker="o")
    plt.title("Longevidade média por profissão")
    plt.xlabel("Profissão (índice)")
    plt.ylabel("Anos (média)")
    plt.grid(True)

    plt.figure()
    plt.plot(range(len(profs)), fit_mean, marker="o")
    plt.title("Fitness composto médio por profissão")
    plt.xlabel("Profissão (índice)")
    plt.ylabel("Fitness (0..1)")
    plt.grid(True)


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

    print_fitness_report(people, meta, top_n=15)

    # =========================
    # GRÁFICOS (novas métricas)
    # =========================
    try:
        plot_new_metrics(doc, people, meta)
    except Exception as ex:
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
