import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import math
import statistics
import matplotlib.pyplot as plt

# -------------------------
# Load
# -------------------------
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def get_events(doc):
    return doc.get("events", [])

def get_final_people(doc):
    return doc.get("final_people", [])

def get_start_pop(doc, final_people):
    # prioridade: meta.start_pop (teu main.py salva isso)
    meta = doc.get("meta", {})
    if isinstance(meta, dict) and isinstance(meta.get("start_pop"), int):
        return meta["start_pop"]

    # fallback: config.START_POP (se existir em algum log antigo)
    cfg = doc.get("config", {})
    if isinstance(cfg, dict) and isinstance(cfg.get("START_POP"), int):
        return cfg["START_POP"]

    # fallback: inferir por final_people vivos no ano 0
    alive0 = 0
    for p in final_people:
        born = p.get("born_year", 0)
        died = p.get("died_year", None)
        if born <= 0 and (died is None or died > 0):
            alive0 += 1
    return alive0

def get_max_year(doc, events, final_people):
    # usa meta.year_final se existir, senão max dos eventos, senão max de died/born
    meta = doc.get("meta", {})
    if isinstance(meta, dict) and isinstance(meta.get("year_final"), int):
        return meta["year_final"]

    if events:
        return max(int(e.get("year", 0)) for e in events)

    mx = 0
    for p in final_people:
        mx = max(mx, int(p.get("born_year", 0)))
        dy = p.get("died_year", None)
        if dy is not None:
            mx = max(mx, int(dy))
    return mx

# -------------------------
# Stats helpers
# -------------------------
def pearson_r(xs, ys):
    if len(xs) < 3:
        return float("nan")
    try:
        return statistics.correlation(xs, ys)
    except Exception:
        return float("nan")

def gini(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    vals = sorted(vals)
    s = sum(vals)
    if s == 0:
        return 0.0
    n = len(vals)
    cum = 0
    for i, v in enumerate(vals, 1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n

# -------------------------
# Reconstruct population
# -------------------------
def build_births_deaths(events, max_y):
    births = [0]*(max_y+1)
    deaths = [0]*(max_y+1)

    for e in events:
        y = int(e.get("year", 0))
        if y < 0 or y > max_y:
            continue
        t = e.get("type")
        if t == "birth":
            births[y] += 1
        elif t == "death":
            deaths[y] += 1

    return births, deaths

def build_population_series(start_pop, births, deaths):
    max_y = len(births)-1
    pop = [0]*(max_y+1)
    pop[0] = start_pop
    for y in range(1, max_y+1):
        pop[y] = pop[y-1] + births[y] - deaths[y]
    return pop

def growth_rate(pop):
    # (pop[y]-pop[y-1]) / pop[y-1]
    gr = [0.0]*len(pop)
    for y in range(1, len(pop)):
        prev = pop[y-1]
        gr[y] = (pop[y]-prev)/prev if prev else 0.0
    return gr

# -------------------------
# Partner timeline reconstruction
# -------------------------
def build_partner_timeline(events, max_y):
    """
    Retorna partners_by_year[y] = dict(pid -> partner_pid or None)
    Assumimos:
      - pair_formed cria vínculo naquele ano
      - death encerra o vínculo naquele ano
      - pode haver recasamento depois (porque pair_formed pode ocorrer de novo)
    """
    # eventos por ano
    by_year = defaultdict(list)
    for e in events:
        y = int(e.get("year", 0))
        if 0 <= y <= max_y:
            by_year[y].append(e)

    current_partner = {}  # pid -> partner or None (apenas para quem já apareceu)
    partners_by_year = []

    for y in range(max_y+1):
        # aplica eventos do ano
        for e in by_year.get(y, []):
            t = e.get("type")
            if t == "pair_formed":
                a = e.get("a"); b = e.get("b")
                if isinstance(a, int) and isinstance(b, int):
                    current_partner[a] = b
                    current_partner[b] = a
            elif t == "death":
                pid = e.get("pid")
                if isinstance(pid, int):
                    # se ele tinha parceiro, solte o parceiro
                    partner = current_partner.get(pid, None)
                    if partner is not None:
                        current_partner[partner] = None
                    current_partner[pid] = None

        # snapshot
        partners_by_year.append(dict(current_partner))

    return partners_by_year

# -------------------------
# Alive & ages per year (from final_people)
# -------------------------
def alive_and_age(final_people, year):
    alive = []
    for p in final_people:
        born = int(p.get("born_year", 0))
        died = p.get("died_year", None)
        if year < born:
            continue
        if died is not None and year >= int(died):
            continue
        age = year - born
        alive.append((p, age))
    return alive

# -------------------------
# Metrics
# -------------------------
def singles_metrics(final_people, partners_by_year, max_y, repro_min=18, repro_max=42):
    singles = [0]*(max_y+1)
    fertile = [0]*(max_y+1)

    for y in range(max_y+1):
        alive = alive_and_age(final_people, y)
        pmap = partners_by_year[y] if y < len(partners_by_year) else {}

        for p, age in alive:
            if not (repro_min <= age <= repro_max):
                continue
            fertile[y] += 1
            pid = p.get("pid")
            partner = pmap.get(pid, None)
            if partner is None:
                singles[y] += 1

    rate = [ (singles[y]/fertile[y] if fertile[y] else 0.0) for y in range(max_y+1) ]
    return singles, fertile, rate

def cross_zone_rate(events, max_y):
    total_pairs = [0]*(max_y+1)
    cross_pairs = [0]*(max_y+1)

    for e in events:
        if e.get("type") != "pair_formed":
            continue
        y = int(e.get("year", 0))
        if 0 <= y <= max_y:
            total_pairs[y] += 1
            if e.get("cross_zone") is True:
                cross_pairs[y] += 1

    rate = [ (cross_pairs[y]/total_pairs[y] if total_pairs[y] else 0.0) for y in range(max_y+1) ]
    return rate, total_pairs, cross_pairs

def age_first_marriage(events, final_people):
    born_year = {p["pid"]: int(p.get("born_year", 0)) for p in final_people if "pid" in p}
    ages_by_year = defaultdict(list)

    for e in events:
        if e.get("type") != "marriage":
            continue
        y = int(e.get("year", 0))
        a = e.get("a"); b = e.get("b")
        if isinstance(a, int) and a in born_year:
            ages_by_year[y].append(y - born_year[a])
        if isinstance(b, int) and b in born_year:
            ages_by_year[y].append(y - born_year[b])

    # série com média por ano (zeros quando não tem)
    return ages_by_year

def kids_by_couple(events):
    couples = Counter()
    for e in events:
        if e.get("type") == "birth":
            m = e.get("mother"); f = e.get("father")
            if isinstance(m, int) and isinstance(f, int):
                couples[tuple(sorted((m, f)))] += 1
    return couples

def pipeline_times(events):
    """
    Mede delays:
      pair_formed -> cohabitation
      cohabitation -> birth (primeiro filho)
    """
    pair_year = {}
    cohab_year = {}
    first_birth_year = {}

    def key(a, b):
        return tuple(sorted((a, b)))

    for e in sorted(events, key=lambda x: int(x.get("year", 0))):
        y = int(e.get("year", 0))
        t = e.get("type")
        if t in ("pair_formed", "cohabitation", "marriage"):
            a = e.get("a"); b = e.get("b")
            if not (isinstance(a, int) and isinstance(b, int)):
                continue
            k = key(a, b)
            if t == "pair_formed" and k not in pair_year:
                pair_year[k] = y
            elif t == "cohabitation" and k not in cohab_year:
                cohab_year[k] = y
        elif t == "birth":
            m = e.get("mother"); f = e.get("father")
            if not (isinstance(m, int) and isinstance(f, int)):
                continue
            k = key(m, f)
            if k not in first_birth_year:
                first_birth_year[k] = y

    dt_pair_to_cohab = []
    dt_cohab_to_birth = []
    sterile_pairs = 0

    for k, py in pair_year.items():
        cy = cohab_year.get(k, None)
        by = first_birth_year.get(k, None)

        if cy is not None:
            dt_pair_to_cohab.append(cy - py)

        if cy is not None and by is not None:
            dt_cohab_to_birth.append(by - cy)

        # "estéril" = teve par, mas nunca teve nascimento
        if by is None:
            sterile_pairs += 1

    return dt_pair_to_cohab, dt_cohab_to_birth, sterile_pairs, len(pair_year)

def dna_vs_kids(final_people):
    # kids_count está em final_people
    xs = defaultdict(list)
    ys = []
    for p in final_people:
        kc = p.get("kids_count", None)
        dna = p.get("dna", None)
        if kc is None or not isinstance(dna, dict):
            continue
        ys.append(kc)
        for g in ("speed", "fert", "soc", "vit"):
            if g in dna and isinstance(dna[g], (int, float)):
                xs[g].append(float(dna[g]))
    # cuidado: listas podem ficar desalinhadas se algum gene faltar, mas no teu log geralmente vem completo
    # então aqui assumimos que vem completo.
    return xs, ys

def births_dna_series(events, max_y):
    series = {g: [None]*(max_y+1) for g in ("speed", "fert", "soc", "vit")}
    vals_by_year = {g: defaultdict(list) for g in ("speed","fert","soc","vit")}

    for e in events:
        if e.get("type") != "birth":
            continue
        y = int(e.get("year", 0))
        dna = e.get("dna")
        if not isinstance(dna, dict):
            continue
        for g in ("speed","fert","soc","vit"):
            if g in dna and isinstance(dna[g], (int, float)):
                vals_by_year[g][y].append(float(dna[g]))

    for g in ("speed","fert","soc","vit"):
        for y in range(max_y+1):
            vs = vals_by_year[g].get(y, [])
            if vs:
                series[g][y] = sum(vs)/len(vs)

    return series

def gene_variance_alive(final_people, max_y):
    var = {g: [0.0]*(max_y+1) for g in ("speed","fert","soc","vit")}
    for y in range(max_y+1):
        alive = alive_and_age(final_people, y)
        for g in ("speed","fert","soc","vit"):
            vs = []
            for p, _age in alive:
                dna = p.get("dna", {})
                if isinstance(dna, dict) and g in dna:
                    vs.append(float(dna[g]))
            var[g][y] = statistics.pvariance(vs) if len(vs) >= 2 else 0.0
    return var

def age_pyramid(final_people, year, bins=(0,10,20,30,40,50,60,70,80,200)):
    alive = alive_and_age(final_people, year)
    ages = [age for _p, age in alive]
    counts = [0]*(len(bins)-1)
    for a in ages:
        for i in range(len(bins)-1):
            if bins[i] <= a < bins[i+1]:
                counts[i] += 1
                break
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    return labels, counts

# -------------------------
# Plotting
# -------------------------
def maybe_save(fig_id, out_dir):
    if not out_dir:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(fig_id)
    plt.savefig(out / f"plot_{fig_id}.png", dpi=160, bbox_inches="tight")

def plot_population(pop, births, deaths, gr, out_dir=None):
    x = list(range(len(pop)))

    plt.figure()
    plt.plot(x, pop, label="População (reconstruída)")
    plt.title("População")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.grid(True)
    plt.legend()
    maybe_save(plt.gcf().number, out_dir)

    plt.figure()
    plt.plot(x, births, label="Nascimentos")
    plt.plot(x, deaths, label="Mortes")
    plt.title("Nascimentos e mortes por ano")
    plt.xlabel("Ano")
    plt.ylabel("Eventos/ano")
    plt.grid(True)
    plt.legend()
    maybe_save(plt.gcf().number, out_dir)

    plt.figure()
    plt.plot(x, gr, label="Crescimento (Δpop/pop)")
    plt.title("Taxa de crescimento anual")
    plt.xlabel("Ano")
    plt.ylabel("Proporção")
    plt.grid(True)
    plt.legend()
    maybe_save(plt.gcf().number, out_dir)

def plot_social(x, singles, rate, cross_rate, marr_age_by_year, out_dir=None):
    plt.figure()
    plt.plot(x, singles, label="Solteiros (janela fértil)")
    plt.title("Solteiros férteis ao longo do tempo")
    plt.xlabel("Ano")
    plt.ylabel("Pessoas")
    plt.grid(True)
    plt.legend()
    maybe_save(plt.gcf().number, out_dir)

    plt.figure()
    plt.plot(x, rate)
    plt.title("% solteiros (janela fértil)")
    plt.xlabel("Ano")
    plt.ylabel("Proporção")
    plt.grid(True)
    maybe_save(plt.gcf().number, out_dir)

    plt.figure()
    plt.plot(x, cross_rate)
    plt.title("Taxa de pares formados fora do bairro (cross_zone)")
    plt.xlabel("Ano")
    plt.ylabel("Proporção")
    plt.grid(True)
    maybe_save(plt.gcf().number, out_dir)

    # idade média no casamento (anos sem casamento ficam como 0 por visualização)
    yvals = [0.0]*(max(x)+1 if x else 0)
    for y, ages in marr_age_by_year.items():
        if ages:
            yvals[y] = sum(ages)/len(ages)

    plt.figure()
    plt.plot(range(len(yvals)), yvals)
    plt.title("Idade média no 1º casamento (por ano)")
    plt.xlabel("Ano")
    plt.ylabel("Idade (anos)")
    plt.grid(True)
    maybe_save(plt.gcf().number, out_dir)

def plot_reproduction(kids_per_couple, pipeline1, pipeline2, sterile_pairs, total_pairs, out_dir=None):
    kids = kids_per_couple
    if kids:
        plt.figure()
        plt.hist(kids, bins=range(0, max(kids)+2))
        plt.title("Distribuição de filhos por casal")
        plt.xlabel("Filhos")
        plt.ylabel("Casais")
        plt.grid(True)
        maybe_save(plt.gcf().number, out_dir)

        # R0 aproximado (filhos por casal /2 por pessoa, só como “termômetro”)
        avg_k = sum(kids)/len(kids)
        r0 = avg_k / 2.0
        print(f"[REPRO] filhos/casal médio={avg_k:.2f} | R0~{r0:.2f}")

        print(f"[REPRO] casais com par formado: {total_pairs} | estéreis (sem filho): {sterile_pairs} ({(sterile_pairs/total_pairs*100 if total_pairs else 0):.1f}%)")
        print(f"[REPRO] Gini de filhos por casal: {gini(kids):.3f}")

    if pipeline1:
        plt.figure()
        plt.hist(pipeline1, bins=min(20, len(set(pipeline1))))
        plt.title("Delay: pair_formed → cohabitation")
        plt.xlabel("Anos")
        plt.ylabel("Casais")
        plt.grid(True)
        maybe_save(plt.gcf().number, out_dir)

    if pipeline2:
        plt.figure()
        plt.hist(pipeline2, bins=min(20, len(set(pipeline2))))
        plt.title("Delay: cohabitation → 1º filho")
        plt.xlabel("Anos")
        plt.ylabel("Casais")
        plt.grid(True)
        maybe_save(plt.gcf().number, out_dir)

def plot_genetics(final_people, events, max_y, out_dir=None):
    # DNA vs filhos (scatter)
    xs, ys = dna_vs_kids(final_people)
    for g in ("speed","fert","soc","vit"):
        if g not in xs or len(xs[g]) < 3 or len(ys) < 3:
            continue
        r = pearson_r(xs[g], ys)
        plt.figure()
        plt.scatter(xs[g], ys)
        plt.title(f"DNA vs filhos: {g} (r={r:.2f})")
        plt.xlabel(g)
        plt.ylabel("Filhos (kids_count)")
        plt.grid(True)
        maybe_save(plt.gcf().number, out_dir)

    # Média dos genes nos nascidos por ano
    birth_avg = births_dna_series(events, max_y)
    x = list(range(max_y+1))
    for g in ("speed","fert","soc","vit"):
        ys2 = [birth_avg[g][i] if birth_avg[g][i] is not None else float("nan") for i in x]
        plt.figure()
        plt.plot(x, ys2)
        plt.title(f"Média do gene nos nascidos: {g}")
        plt.xlabel("Ano")
        plt.ylabel(g)
        plt.grid(True)
        maybe_save(plt.gcf().number, out_dir)

    # Variância genética nos vivos (diversidade)
    var = gene_variance_alive(final_people, max_y)
    for g in ("speed","fert","soc","vit"):
        plt.figure()
        plt.plot(x, var[g])
        plt.title(f"Variância genética nos vivos: {g}")
        plt.xlabel("Ano")
        plt.ylabel("Variância")
        plt.grid(True)
        maybe_save(plt.gcf().number, out_dir)

def plot_age_pyramids(final_people, years, out_dir=None):
    for y in years:
        labels, counts = age_pyramid(final_people, y)
        plt.figure()
        plt.bar(labels, counts)
        plt.title(f"Pirâmide etária (ano {y})")
        plt.xlabel("Faixa etária")
        plt.ylabel("Pessoas")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y")
        maybe_save(plt.gcf().number, out_dir)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Analisador unificado da Villa (ALL v2).")
    ap.add_argument("json_path", help="sim_log_*.json")
    ap.add_argument("--no-show", action="store_true", help="Não abre janelas; útil com --out.")
    ap.add_argument("--out", default=None, help="Pasta para salvar PNGs.")
    ap.add_argument("--repro-min", type=int, default=18)
    ap.add_argument("--repro-max", type=int, default=42)
    ap.add_argument("--pyramids", default="0,50,100,150,200", help="Anos para pirâmide etária (CSV).")
    args = ap.parse_args()

    doc = load_json(args.json_path)
    events = get_events(doc)
    final_people = get_final_people(doc)

    max_y = get_max_year(doc, events, final_people)
    start_pop = get_start_pop(doc, final_people)

    births, deaths = build_births_deaths(events, max_y)
    pop = build_population_series(start_pop, births, deaths)
    gr = growth_rate(pop)

    partners_by_year = build_partner_timeline(events, max_y)
    singles, fertile, single_rate = singles_metrics(
        final_people, partners_by_year, max_y,
        repro_min=args.repro_min, repro_max=args.repro_max
    )

    cross_rate, total_pairs, cross_pairs = cross_zone_rate(events, max_y)
    marr_age_by_year = age_first_marriage(events, final_people)

    couples_kids = kids_by_couple(events)
    kids_list = list(couples_kids.values())

    dt_pair_cohab, dt_cohab_birth, sterile_pairs, total_pairs_formed = pipeline_times(events)

    # prints úteis
    print("\n=== META ===")
    meta = doc.get("meta", {})
    print(f"Ano final (detectado): {max_y} | START_POP usado: {start_pop}")
    print(f"Eventos: {len(events)} | Pessoas finais: {len(final_people)}")

    # PLOTS
    x = list(range(max_y+1))
    plot_population(pop, births, deaths, gr, out_dir=args.out)
    plot_social(x, singles, single_rate, cross_rate, marr_age_by_year, out_dir=args.out)
    plot_reproduction(kids_list, dt_pair_cohab, dt_cohab_birth, sterile_pairs, total_pairs_formed, out_dir=args.out)
    plot_genetics(final_people, events, max_y, out_dir=args.out)

    # pirâmides
    try:
        years = [int(s.strip()) for s in args.pyramids.split(",") if s.strip()]
        years = [y for y in years if 0 <= y <= max_y]
        if years:
            plot_age_pyramids(final_people, years, out_dir=args.out)
    except Exception:
        pass

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
