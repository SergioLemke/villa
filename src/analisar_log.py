import json
import sys
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


def resolve_input_path(path_str: str) -> Path:
    """
    Resolve o caminho do JSON de forma robusta:
    - se for absoluto: usa direto
    - se for relativo: assume que é relativo à raiz do projeto (pai da pasta src)
    """
    p = Path(path_str)

    if p.is_absolute():
        return p

    # __file__ = .../villa2/src/analisar_log.py
    script_dir = Path(__file__).resolve().parent      # .../villa2/src
    project_root = script_dir.parent                  # .../villa2

    return project_root / p


def load_json(path_str: str) -> dict:
    p = resolve_input_path(path_str)

    if not p.exists():
        # tenta mais um fallback: se o user passou tipo "src\alguma_coisa.json"
        # e ficou duplicado, normaliza
        p2 = Path(path_str).resolve()
        if p2.exists():
            p = p2
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(v, default=None):
    return v if isinstance(v, int) else default


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python src\\analisar_log.py sim_log_20260110_025110.json")
        print("  python src\\analisar_log.py D:\\codando\\villa2\\sim_log_20260110_025110.json")
        sys.exit(1)

    input_path = sys.argv[1]
    data = load_json(input_path)

    events = data.get("events", [])
    final_people = data.get("final_people", [])
    cfg = data.get("config", {})

    final_year = safe_int(data.get("final_year"), None)
    final_gen = safe_int(data.get("final_generation"), None)

    # --- eventos por tipo e ano ---
    by_type = Counter(e.get("type") for e in events)

    years = [e.get("year", 0) for e in events if isinstance(e.get("year", None), int)]
    max_year_in_events = max(years) if years else 0
    max_year = final_year if isinstance(final_year, int) else max_year_in_events

    births_by_year = [0] * (max_year + 1)
    deaths_by_year = [0] * (max_year + 1)
    pairs_by_year  = [0] * (max_year + 1)
    cohab_by_year  = [0] * (max_year + 1)
    marr_by_year   = [0] * (max_year + 1)

    for e in events:
        y = e.get("year", 0)
        if not isinstance(y, int) or y < 0 or y > max_year:
            continue
        t = e.get("type")
        if t == "birth":
            births_by_year[y] += 1
        elif t == "death":
            deaths_by_year[y] += 1
        elif t == "pair_formed":
            pairs_by_year[y] += 1
        elif t == "cohabitation":
            cohab_by_year[y] += 1
        elif t == "marriage":
            marr_by_year[y] += 1

    # --- população reconstruída ---
    start_pop = cfg.get("START_POP", None)
    if not isinstance(start_pop, int):
        # fallback: estima por "nascimentos no ano 0"? (geralmente 0)
        start_pop = 0

    pop = [0] * (max_year + 1)
    pop[0] = start_pop
    for y in range(1, max_year + 1):
        pop[y] = pop[y-1] + births_by_year[y] - deaths_by_year[y]

    # --- DNA final ---
    stats = ["str", "int", "dex", "cha", "vit"]
    dna_avg = {k: 0.0 for k in stats}
    dna_vals = {k: [] for k in stats}

    for p in final_people:
        dna = p.get("dna", {})
        for k in stats:
            v = dna.get(k, None)
            if isinstance(v, (int, float)):
                dna_avg[k] += float(v)
                dna_vals[k].append(float(v))

    n_final = len(final_people)
    if n_final > 0:
        for k in stats:
            dna_avg[k] /= n_final

    # --- profissões finais ---
    prof_counter = Counter(p.get("profession", None) for p in final_people)
    prof_counter = Counter({("Crianca" if k is None else k): v for k, v in prof_counter.items()})

    # --- linhagens (pais/casais com mais filhos) ---
    parent_kids = Counter()
    couples_kids = Counter()
    for e in events:
        if e.get("type") == "birth":
            m = e.get("mother")
            f = e.get("father")
            if isinstance(m, int):
                parent_kids[m] += 1
            if isinstance(f, int):
                parent_kids[f] += 1
            if isinstance(m, int) and isinstance(f, int):
                key = tuple(sorted((m, f)))
                couples_kids[key] += 1

    top_parents = parent_kids.most_common(10)
    top_couples = couples_kids.most_common(10)

    # --- imprime resumo ---
    resolved = resolve_input_path(input_path)
    print("\n=== RESUMO DO LOG ===")
    print(f"Entrada: {input_path}")
    print(f"Resolvido: {resolved}")
    print(f"Ano final: {final_year} | Geracao final: {final_gen}")
    print(f"Eventos totais: {len(events)}")

    print("\nEventos por tipo:")
    for t, c in by_type.most_common():
        print(f"  - {t}: {c}")

    print("\nPopulação:")
    print(f"  START_POP (config): {start_pop}")
    print(f"  Final (no JSON): {n_final}")
    if max_year >= 0:
        print(f"  Estimativa reconstruída no ano {max_year}: {pop[max_year]}")

    print("\nDNA final (média):")
    for k in stats:
        print(f"  {k}: {dna_avg[k]:.2f}")

    print("\nProfissões finais:")
    for k, v in prof_counter.most_common():
        print(f"  {k}: {v}")

    if top_parents:
        print("\nTop 10 IDs por número de filhos (mãe+pai contam):")
        for pid, c in top_parents:
            print(f"  ID {pid}: {c} filhos")

    if top_couples:
        print("\nTop 10 casais por número de filhos:")
        for (a, b), c in top_couples:
            print(f"  Casal ({a}, {b}): {c} filhos")

    # --- gráficos ---
    x = list(range(max_year + 1))

    # 1) População
    plt.figure()
    plt.plot(x, pop)
    plt.title("População (reconstruída)")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.grid(True)

    # 2) Nascimentos e mortes por ano
    plt.figure()
    plt.plot(x, births_by_year, label="Nascimentos")
    plt.plot(x, deaths_by_year, label="Mortes")
    plt.title("Nascimentos e mortes por ano")
    plt.xlabel("Ano")
    plt.ylabel("Eventos/ano")
    plt.grid(True)
    plt.legend()

    # 3) Eventos sociais
    plt.figure()
    plt.plot(x, pairs_by_year, label="Pares formados")
    plt.plot(x, cohab_by_year, label="Coabitações")
    plt.plot(x, marr_by_year, label="Casamentos")
    plt.title("Eventos sociais por ano")
    plt.xlabel("Ano")
    plt.ylabel("Eventos/ano")
    plt.grid(True)
    plt.legend()

    # 4) Distribuição DNA final
    for k in stats:
        if not dna_vals[k]:
            continue
        plt.figure()
        plt.hist(dna_vals[k], bins=20)
        plt.title(f"Distribuição final de {k}")
        plt.xlabel(k)
        plt.ylabel("Contagem")
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
