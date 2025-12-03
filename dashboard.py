import os
import io
import json
from openai import OpenAI
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, inspect
import re

# ----------------------------
# Config
# ----------------------------


st.set_page_config(page_title="Project Updates", layout="wide", initial_sidebar_state="expanded")

# One CSS block to handle everything

# Use Postgres in prod via env DATABASE_URL; fallback to local SQLite for dev
DATABASE_URL = (
    st.secrets.get("DATABASE_URL")          # Streamlit Cloud
    or os.getenv("DATABASE_URL")            # local/CLI
)


engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


# Seed file (your test inputs). In prod, you can clear this env var.
SEED_INPUTS_PATH = os.getenv("SEED_INPUTS_PATH", "")


# Exactly match the columns from your Excel (order matters)
EXPECTED_INPUT_COLS = [
    "Bid Name","Bid Category","Stage","Engineer","Bid Owner","Mechanical Contractor",
    "Bidder Owner","Must-Close","Location","Projected Total","Address","Project Name"
]

ARTICLE_COLS = [
    "Article Title", "Article Date", "Groundbreaking Year", "Completion Year", "Scraped Date", "Article Link",
    "Article Summary", "Milestone Mentions"
]

# ----------------------------
# Schema helpers
# ----------------------------

JOIN_DELIM = " + "
SPLIT_PATTERN = r"\s*\+\s*"   # split ONLY on ';' or '+' (never commas)

# ---------- ARCHIVE TABLES & HELPERS ----------

# ========= Chatbot helpers (single-table, safe SQL) =========


from sqlalchemy import MetaData, Table

def _common_cols(src: str, dst: str) -> list[str]:
    src_cols = get_table_columns(src) or []
    dst_cols = get_table_columns(dst) or []
    return [c for c in src_cols if c in dst_cols]

def _move_rows_oneway(src: str, dst: str, key_cols: list[str], rows_df: pd.DataFrame) -> int:
    """
    Move selected rows from `src` -> `dst` using key_cols.
    - INSERT ... SELECT ... WHERE keys match
    - AND NOT EXISTS in dst (prevents dupes)
    - then DELETE from src
    Works on Postgres and SQLite (no ON CONFLICT).
    """
    if rows_df.empty:
        return 0

    cols = _common_cols(src, dst)
    if not cols:
        return 0

    # Build key predicate with stable bind names (no spaces/hyphens)
    key_pred_src_alias = " AND ".join([f'a.{quote_ident(k)} = :k{i}' for i, k in enumerate(key_cols)])
    key_pred_src_table = " AND ".join([f'{quote_ident(k)} = :k{i}' for i, k in enumerate(key_cols)])
    not_exists_join = " AND ".join([f'r.{quote_ident(k)} = a.{quote_ident(k)}' for k in key_cols])

    insert_sql = text(f"""
        INSERT INTO {dst} ({", ".join(quote_ident(c) for c in cols)})
        SELECT {", ".join("a." + quote_ident(c) for c in cols)}
        FROM {src} AS a
        WHERE {key_pred_src_alias}
          AND NOT EXISTS (
                SELECT 1 FROM {dst} AS r
                WHERE {not_exists_join}
          )
    """)
    delete_sql = text(f"""
        DELETE FROM {src}
        WHERE {key_pred_src_table}
    """)

    moved = 0
    with engine.begin() as conn:
        for _, r in rows_df.iterrows():
            binds = {f'k{i}': (None if pd.isna(r.get(k)) else str(r.get(k))) for i, k in enumerate(key_cols)}
            conn.execute(insert_sql, binds)
            conn.execute(delete_sql,  binds)
            moved += 1
    return moved

# Keys you already use elsewhere for stable identity
RESULTS_KEYS = ["Project Name", "Address", "Article Link"]
GENERAL_KEYS = ["Project Name", "Article Link"]

def archive_results(rows_df: pd.DataFrame) -> int:
    return _move_rows_oneway("results", "archived_results", RESULTS_KEYS, rows_df)



def _insert_rows_sql(table: str, df: pd.DataFrame) -> int:
    """Append df rows into `table` using SQLAlchemy Core insert."""
    if df.empty:
        return 0

    meta = MetaData()
    tbl = Table(table, meta, autoload_with=engine)

    # Only send columns that exist in the destination table
    dest_cols = [c.name for c in tbl.columns if c.name in df.columns]
    if not dest_cols:
        return 0

    # Convert NaNs/empty strings to None so DB accepts them as NULL
    records = (
        df[dest_cols]
        .applymap(lambda v: None if (pd.isna(v) or v == "") else v)
        .to_dict(orient="records")
    )

    with engine.begin() as conn:
        if records:
            conn.execute(tbl.insert(), records)
    return len(records)


def ensure_archived_tables():
    insp = inspect(engine)

    def make_like(source_table: str, archive_table: str):
        src_cols = get_table_columns(source_table) or []

        if insp.has_table(archive_table):
            if src_cols:
                existing = get_table_columns(archive_table) or []
                to_add = [c for c in src_cols if c not in existing]
                if to_add:
                    with engine.begin() as conn:
                        for c in to_add:
                            conn.execute(text(f'ALTER TABLE {archive_table} ADD COLUMN "{c}" TEXT'))
                        if "archived_at" not in existing:
                            conn.execute(text(f'ALTER TABLE {archive_table} ADD COLUMN archived_at TIMESTAMP'))
            return

        if src_cols:
            cols_sql = ", ".join(f'"{c}" TEXT' for c in src_cols)
            ddl = f"CREATE TABLE {archive_table} ({cols_sql}, archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        else:
            ddl = f"CREATE TABLE {archive_table} (archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        with engine.begin() as conn:
            conn.execute(text(ddl))

    # Results archive unchanged
    make_like("results", "archived_results")
    # Mirror the *scored* general table
    make_like("general_internal_scored", "archived_general")



def _safe_insert_df(df: pd.DataFrame, table: str):
    """Insert df into table, selecting only columns that exist in dest."""
    if df.empty:
        return 0
    dest_cols = get_table_columns(table) or list(df.columns)
    subset = [c for c in df.columns if c in dest_cols]
    if not subset:
        return 0
    df[subset].to_sql(table, con=engine, if_exists="append", index=False, method="multi")
    return len(df)


# ---- Archive from MAIN -> ARCHIVE ----
def archive_results_rows(rows_df: pd.DataFrame):
    """Move rows from results -> archived_results (by Project Name, Address, Article Link)."""
    if rows_df.empty:
        return 0
    moved = 0
    with engine.begin() as conn:
        # Insert first
        moved += _safe_insert_df(rows_df.copy(), "archived_results")
        # Then delete from main
        for _, r in rows_df.iterrows():
            conn.execute(
                text('DELETE FROM results WHERE "Project Name"=:pn AND "Address"=:addr AND "Article Link"=:al'),
                {"pn": str(r.get("Project Name","")), "addr": str(r.get("Address","")), "al": str(r.get("Article Link",""))}
            )
    return moved

def archive_general_scored_rows(rows_df: pd.DataFrame):
    """Move rows from general_internal_scored -> archived_general (by Project Name, Article Link)."""
    if rows_df.empty:
        return 0
    moved = 0
    with engine.begin() as conn:
        moved += _safe_insert_df(rows_df.copy(), "archived_general")
        for _, r in rows_df.iterrows():
            conn.execute(
                text('DELETE FROM general_internal_scored WHERE "Project Name"=:pn AND "Article Link"=:al'),
                {"pn": str(r.get("Project Name","")), "al": str(r.get("Article Link",""))}
            )
    return moved




# ---- Permanently delete FROM ARCHIVE ----
def delete_archived_results_rows(rows_df: pd.DataFrame):
    if rows_df.empty:
        return 0
    n = 0
    with engine.begin() as conn:
        for _, r in rows_df.iterrows():
            conn.execute(
                text('DELETE FROM archived_results WHERE "Project Name"=:pn AND "Address"=:addr AND "Article Link"=:al'),
                {"pn": str(r.get("Project Name","")), "addr": str(r.get("Address","")), "al": str(r.get("Article Link",""))}
            )
            n += 1
    return n

def delete_archived_general_rows(rows_df: pd.DataFrame):
    if rows_df.empty:
        return 0
    n = 0
    with engine.begin() as conn:
        for _, r in rows_df.iterrows():
            conn.execute(
                text('DELETE FROM archived_general WHERE "Project Name"=:pn AND "Article Link"=:al'),
                {"pn": str(r.get("Project Name","")), "al": str(r.get("Article Link",""))}
            )
            n += 1
    return n



def split_for_dedupe(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    parts = re.split(SPLIT_PATTERN, s)
    parts = [p.strip() for p in parts if p.strip()]
    # if no real split happened, keep the whole cell as one token
    return parts if len(parts) > 1 else [s]

def reorder_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    new_order = []

    # 1) Bid Name
    if "Bid Name" in cols:
        new_order.append("Bid Name")

    # 2) Article Link (if exists)
    if "Article Link" in cols:
        new_order.append("Article Link")

    # 3) Remaining article columns (except Article Link)
    for c in ARTICLE_COLS:
        if c in cols and c not in new_order:
            new_order.append(c)

    # 4) Remaining input cols
    for c in EXPECTED_INPUT_COLS:
        if c not in new_order and c in cols:
            new_order.append(c)

    # 5) Anything else
    for c in cols:
        if c not in new_order:
            new_order.append(c)

    return df[new_order]

def join_unique(series: pd.Series, delim=JOIN_DELIM) -> str:
    seen, ordered = set(), []
    for v in series.tolist():
        for tok in split_for_dedupe(v):
            if tok not in seen:
                seen.add(tok)
                ordered.append(tok)
    return delim.join(ordered)

def consolidate_by_bid_name(df: pd.DataFrame, key_col="Bid Name") -> pd.DataFrame:
    df = df[EXPECTED_INPUT_COLS].copy()

    def agg_group(g: pd.DataFrame) -> pd.Series:
        out = {key_col: g.iloc[0][key_col]}
        for col in EXPECTED_INPUT_COLS:
            if col == key_col:
                continue
            out[col] = join_unique(g[col])  # will join with " + "
        return pd.Series(out, index=EXPECTED_INPUT_COLS)

    return (
        df.groupby(key_col, sort=False, dropna=False)
          .apply(agg_group)
          .reset_index(drop=True)
    )

def filter_required_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with non-empty Project Name AND Address."""
    df = df.copy()
    for col in ["Project Name", "Address"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].fillna("").astype(str).str.strip()
    keep = (df["Project Name"] != "") & (df["Address"] != "")
    return df[keep]



def quote_ident(col: str) -> str:
    # Double-quote SQL identifiers and escape internal quotes
    return '"' + col.replace('"', '""') + '"'

def get_table_columns(table: str):
    insp = inspect(engine)
    if not insp.has_table(table):
        return None
    cols = [c["name"] for c in insp.get_columns(table)]
    return cols

# Years helper (place near tokens_for_filter, etc.)
YEAR_RE = re.compile(r'\b(?:19|20)\d{2}\b')

def years_for_filter(series: pd.Series) -> list[str]:
    """Collect distinct 4-digit years from a column that may contain text."""
    yrs = set()
    for cell in series.dropna().astype(str):
        yrs.update(YEAR_RE.findall(cell))
    # return sorted newest->oldest; change to sorted(yrs) if you prefer asc
    return sorted(yrs, key=lambda x: int(x), reverse=True)


def tokens_for_filter(series: pd.Series):
    tokens = set()
    for cell in series.dropna().astype(str):
        for t in re.split(SPLIT_PATTERN, cell):
            t = t.strip()
            if t:
                tokens.add(t)
    return sorted(tokens)

def recreate_inputs_table(expected_cols):
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS inputs;")
        # Build CREATE TABLE with quoted identifiers (TEXT) + created_at
        cols_sql = ",\n".join(f"{quote_ident(c)} TEXT" for c in expected_cols)
        ddl = f"""
        CREATE TABLE inputs (
            {cols_sql},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        conn.execute(text(ddl))
        # Optional helper index on first two cols if present
        if len(expected_cols) >= 2:
            idx = f'CREATE INDEX IF NOT EXISTS idx_inputs_key ON inputs({quote_ident(expected_cols[0])}, {quote_ident(expected_cols[1])})'
            conn.execute(text(idx))

def ensure_results_table(expected_input_cols):
    """
    Additive migration for `results`:
    - Create if missing
    - ADD any missing columns
    - Never drop/recreate in prod
    """
    desired_cols = expected_input_cols + ARTICLE_COLS
    existing = get_table_columns("results")

    def q(name: str) -> str:
        return '"' + name.replace('"','""') + '"'

    if existing is None:
        with engine.begin() as conn:
            cols_sql = ",\n".join(f"{q(c)} TEXT" for c in desired_cols)
            ddl = f"CREATE TABLE results ({cols_sql})"
            conn.execute(text(ddl))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_results_scraped ON results({q('Scraped Date')})"))
        return

    # ADD any missing columns; do NOT drop if there are extras or order differs
    # to_add = [c for c in desired_cols if c not in existing]
    # if to_add:
    #     with engine.begin() as conn:
    #         for c in to_add:
    #             conn.execute(text(f"ALTER TABLE results ADD COLUMN {q(c)} TEXT"))
    #         conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_results_scraped ON results({q('Scraped Date')})"))


def ensure_inputs_table(expected_cols):
    existing = get_table_columns("inputs")
    if existing is None:
        recreate_inputs_table(expected_cols)
        return

    # inputs has created_at at the end; ignore it during comparison
    existing_no_meta = [c for c in existing if c != "created_at"]
    if existing_no_meta != expected_cols:  # compare order and names
        recreate_inputs_table(expected_cols)

def seed_inputs_if_empty():
    # Only seed once if table has no rows and seed file exists
    try:
        count = pd.read_sql("SELECT COUNT(*) AS n FROM inputs", engine)
        if int(count.iloc[0]["n"]) > 0:
            return
        if not SEED_INPUTS_PATH or not os.path.exists(SEED_INPUTS_PATH):
            return
        df_seed = pd.read_excel(SEED_INPUTS_PATH)
        if list(df_seed.columns) != EXPECTED_INPUT_COLS:
            st.warning("Seed file found, but its columns do not exactly match EXPECTED_INPUT_COLS. Skipping seeding.")
            return
        clean = df_seed[EXPECTED_INPUT_COLS].dropna(how="all").drop_duplicates()
        # Append rows
        clean.to_sql("inputs", con=engine, if_exists="append", index=False, method="multi")
        st.success(f"Inputs seeded from {SEED_INPUTS_PATH}.")
    except Exception as e:
        st.error(f"Failed to seed inputs: {e}")

def ensure_general_internal_table():
    """
    Additive migration for `general_internal`:
    - Create if missing
    - ADD any missing columns
    - Never drop/recreate
    """
    desired_cols = [
        "Project Name", "Architect", "Developer", "Possible Engineer",
        "Location", "Groundbreaking Year", "Completion Year",
        "Article Title", "Article Date", "Scraped Date",
        "Article Link", "Article Summary",
        "Milestone Mentions", "Planned Mentions",
        "Lead Score",
    ]

    existing = get_table_columns("general_internal")

    def q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    if existing is None:
        with engine.begin() as conn:
            cols_sql = ", ".join(f"{q(c)} TEXT" for c in desired_cols)
            conn.execute(text(f"CREATE TABLE general_internal ({cols_sql})"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_general_internal_scraped ON general_internal({q('Scraped Date')})"))
        return

    # Add any missing columns
    to_add = [c for c in desired_cols if c not in existing]
    if to_add:
        with engine.begin() as conn:
            for c in to_add:
                conn.execute(text(f"ALTER TABLE general_internal ADD COLUMN {q(c)} TEXT"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_general_internal_scraped ON general_internal({q('Scraped Date')})"))


@st.cache_data(ttl=60)
def load_general():
    cols = [
        "Project Name","Developer","Architect","Possible Engineer","Location", "Address","Territory",
        "Groundbreaking Year","Completion Year",
        "Article Title","Article Date","Scraped Date",
        "Article Link","Article Summary","Milestone Mentions","Planned Mentions",
        "Lead Score",
      
        "Qualified","Justification"
    ]

    q = f"""
    SELECT {', '.join(quote_ident(c) for c in cols if c in (get_table_columns('general_internal_scored') or []))}
    FROM general_internal_scored
    WHERE "Qualified" = 'Yes'
    ORDER BY COALESCE(NULLIF("Scraped Date", ''), '1900-01-01')::date DESC,
             "Article Date" DESC NULLS LAST
    """
    df = pd.read_sql(q, engine)
    if "Scraped Date" in df.columns:
        df["Scraped Date"] = pd.to_datetime(df["Scraped Date"], errors="coerce")

    if "Groundbreaking Year" in df.columns: 
        df = df[~df["Groundbreaking Year"].astype(str).str.contains(r"\b2025\b", na=False)]

   
    return df


def reorder_general(df: pd.DataFrame) -> pd.DataFrame:
    priority = [
        "Project Name", "Lead Score",
        "Architect", "Developer", "Possible Engineer",
        "Location", "Address",
        "Groundbreaking Year","Completion Year",
        "Article Title","Article Date","Scraped Date",
        "Article Link","Article Summary",
        "Milestone Mentions","Planned Mentions", "Topic"
    ]

    order = [c for c in priority if c in df.columns] + [c for c in df.columns if c not in priority]
    return df[order]


# Bootstrap/migrate schema every start

ensure_general_internal_table()
ensure_inputs_table(EXPECTED_INPUT_COLS)
ensure_results_table(EXPECTED_INPUT_COLS)
ensure_archived_tables()
seed_inputs_if_empty()

# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_data(ttl=60)
def load_inputs():
    return pd.read_sql('SELECT * FROM inputs ORDER BY "created_at" DESC', engine)

@st.cache_data(ttl=60)
def load_archived_results():
    # Try to select same columns as 'results' to keep UI consistent
    cols = get_table_columns("archived_results") or []
    if not cols:
        return pd.DataFrame()
    q = f'SELECT {", ".join(quote_ident(c) for c in cols)} FROM archived_results ORDER BY {quote_ident("archived_at")} DESC NULLS LAST'
    return pd.read_sql(q, engine)

@st.cache_data(ttl=60)
def load_archived_general():
    cols = get_table_columns("archived_general") or []
    if not cols:
        return pd.DataFrame()
    # archived_at may exist last; handle gracefully
    order_col = "archived_at" if "archived_at" in cols else cols[0]
    q = f'SELECT {", ".join(quote_ident(c) for c in cols)} FROM archived_general ORDER BY {quote_ident(order_col)} DESC NULLS LAST'
    return pd.read_sql(q, engine)


@st.cache_data(ttl=60)
def load_results():
    # Select dynamic: inputs columns + article fields
    all_cols = EXPECTED_INPUT_COLS + ARTICLE_COLS
    select_list = ", ".join(quote_ident(c) for c in all_cols if c not in ("created_at",))
    # ORDER BY scraped_date desc if present
    q = f"SELECT {select_list} FROM results ORDER BY {quote_ident('Scraped Date')} DESC NULLS LAST"
    df = pd.read_sql(q, engine)
    # Parse known date-ish columns
    if "Scraped Date" in df.columns:
        df["Scraped Date"] = pd.to_datetime(df["Scraped Date"], errors="coerce")
    return df

def replace_inputs_from_excel(file) -> int:
    try:
        df_up = pd.read_excel(file, dtype=str, keep_default_na=False)

        if list(df_up.columns) != EXPECTED_INPUT_COLS:
            raise ValueError("Column names/order must match the example exactly.")

        before = len(df_up)
        df_up = filter_required_rows(df_up)
        dropped = before - len(df_up)

        clean = consolidate_by_bid_name(df_up)

        with engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM inputs;")
        clean.to_sql("inputs", con=engine, if_exists="append", index=False, method="multi")

        load_inputs.clear()
        if dropped > 0:
            st.info(f"Dropped {dropped} row(s) with blank Project Name and/or Address.")
        return len(clean)
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return 0



# ----------------------------
# UI
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Open Bids Dashboard", "General Dashboard", "Inputs", "Archived"]
)



if page == "Open Bids Dashboard":
    st.title("Open Bids Dashboard")
    results = load_results()
    if results.empty:
        st.info("No results yet. When your scraper writes to the `results` table, they will appear here.")
    else:
        # Optional filters: if "Project Name" exists
        row1a, row1b = st.columns([2, 2])
        row2a, row2b = st.columns([2, 3])
        row3a, row3b = st.columns([2, 2])


        with row1a:
            projects = sorted(results.get("Project Name", pd.Series()).dropna().astype(str).unique())
            sel_projects = st.multiselect("Project Name", projects)

        with row1b:
            owner_choices = sorted(results.get("Bid Owner", pd.Series()).dropna().astype(str).str.strip().unique())
            sel_bid_owner = st.multiselect("Bid Owner", owner_choices)

        with row2a:
            bidder_owner_choices = tokens_for_filter(results.get("Bidder Owner", pd.Series()))
            sel_bidder_owner = st.multiselect("Bidder Owner", bidder_owner_choices)
        
      

        def _parse_range(val):
            """Return (start, end) where each is a date or None, regardless of what Streamlit returns."""
            if val is None:
                return None, None
            if isinstance(val, (list, tuple)):
                start = val[0] if len(val) > 0 else None
                end   = val[1] if len(val) > 1 else None
                return start, end
            # Some Streamlit versions return a single date while user is picking
            return val, None
        with row2b:
            warn_placeholder = st.empty()
            if "Scraped Date" in results.columns and results["Scraped Date"].notna().any():
                min_d = results["Scraped Date"].min().date()
                max_d = results["Scraped Date"].max().date()
                scr_range = st.date_input(
                    "Scraped Date Range",
                    value=(min_d, max_d),
                    key="scraped_range",
                )
            else:
                scr_range = None
        
        with row3a:
            gb_choices = years_for_filter(results.get("Groundbreaking Year", pd.Series()))
            sel_gb_years = st.multiselect("Groundbreaking Year", gb_choices)

        with row3b:
            comp_choices = years_for_filter(results.get("Completion Year", pd.Series()))
            sel_comp_years = st.multiselect("Completion Year", comp_choices)

        view = results.copy()
        if sel_projects:
            view = view[view["Project Name"].astype(str).isin(sel_projects)]
        if sel_bid_owner:
            view = view[view["Bid Owner"].astype(str).str.strip().isin(sel_bid_owner)]
        if sel_bidder_owner:
            exploded = view.assign(
                _token=view["Bidder Owner"].fillna("").astype(str).str.split(SPLIT_PATTERN, regex=True)
            ).explode("_token")
            exploded["_token"] = exploded["_token"].str.strip()
            exploded = exploded[exploded["_token"].isin(sel_bidder_owner)]
            view = exploded.drop(columns="_token").drop_duplicates()
        start_d, end_d = _parse_range(scr_range)

        if sel_gb_years:
            gb_col = view.get("Groundbreaking Year", pd.Series(index=view.index, dtype=str)).fillna("").astype(str)
            mask_gb = gb_col.apply(lambda s: bool(set(sel_gb_years) & set(YEAR_RE.findall(s))))
            view = view[mask_gb]

        if sel_comp_years:
            comp_col = view.get("Completion Year", pd.Series(index=view.index, dtype=str)).fillna("").astype(str)
            mask_comp = comp_col.apply(lambda s: bool(set(sel_comp_years) & set(YEAR_RE.findall(s))))
            view = view[mask_comp]

        if start_d and end_d:
            start = pd.to_datetime(start_d)
            end   = pd.to_datetime(end_d) + pd.Timedelta(days=1)  # inclusive end
            view = view[(view["Scraped Date"] >= start) & (view["Scraped Date"] < end)]
        elif start_d or end_d:
            warn_placeholder.info("Select both a start and an end date to apply the filter.")

        st.subheader("Latest Results")
        def _fix_url(u):
            if pd.isna(u) or not str(u).strip():
                return ""
            u = str(u).strip()
            return u if u.startswith(("http://","https://")) else "https://" + u

        view = view.copy()
        if "Article Link" in view.columns:
            view["Article Link"] = view["Article Link"].map(_fix_url)
        
        view = reorder_for_dashboard(view)

        import numpy as np

        THIS_YEAR = pd.Timestamp.today().year  # e.g., 2025
        gb_ser = view.get("Groundbreaking Year")
        gb_mask = gb_ser.astype(str).str.contains(fr"\b{THIS_YEAR}\b", na=False) if gb_ser is not None else pd.Series(False, index=view.index)

        view_for_edit = view.copy()
        view_for_edit["__archive__"] = False
        view_for_edit["⭐ Groundbreaking This Year"] = np.where(gb_mask, "⭐", "")

        # Bring the flag + archive checkbox to the front
        lead_cols = ["__archive__", "⭐ Groundbreaking This Year"]
        view_for_edit = view_for_edit.reindex(columns=lead_cols + [c for c in view_for_edit.columns if c not in lead_cols])

        edited = st.data_editor(
            view_for_edit,
            use_container_width=True,
            height=600,
            column_config={
                "__archive__": st.column_config.CheckboxColumn("Archive?", help="Move this row to Archive"),
                "⭐ Groundbreaking This Year": st.column_config.TextColumn(
                    "⭐ Groundbreaking This Year", help=f"Groundbreaking Year == {THIS_YEAR}"
                ),
                "Article Link": st.column_config.LinkColumn("Article Link", display_text="Open"),
                "Article Summary": st.column_config.TextColumn(width="large"),
                "Milestone Mentions": st.column_config.TextColumn(width="large"),
                "Planned Mentions": st.column_config.TextColumn(width="large"),
                "Developer": st.column_config.TextColumn("Developer"),
            },
            disabled=[c for c in view_for_edit.columns if c != "__archive__"],
            hide_index=True,
        )

        # <-- Build the selection for archiving
        to_archive_df = edited[edited["__archive__"]].drop(columns="__archive__", errors="ignore")

        col_del, _ = st.columns([1, 3])
        with col_del:
            if st.button("Archive selected rows", type="primary"):
                count = archive_results_rows(to_archive_df)
                if count > 0:
                    load_results.clear()
                    load_archived_results.clear()
                    st.rerun()
                else:
                    st.info("No rows selected.")



        # Download filtered CSV (without the checkbox column)
        buf = io.StringIO()
        view.to_csv(buf, index=False)
        st.download_button("Download filtered CSV", buf.getvalue(), "project_updates.csv", "text/csv")


elif page == "General Dashboard":
    st.title("South Florida — General Developments")

    df = load_general()
    if df.empty:
        st.info("No results yet. Run general_scraper.py to populate the 'general' table.")
    else:
        col1, col2, col3 = st.columns([2,2,3])

        with col1:
            projects = sorted(df["Project Name"].dropna().astype(str).unique())
            sel_proj = st.multiselect("Project Name", projects)

        with col2:
            # Lead Score range (1–5). Works even if the column is text.
            lead_min, lead_max = 1, 4
            if "Lead Score" in df.columns:
                # coerce to int safely
                lead_series = pd.to_numeric(df["Lead Score"], errors="coerce").fillna(0).astype(int)
                lead_min = max(1, int(lead_series.min())) if not lead_series.empty else 1
                lead_max = min(4, int(lead_series.max())) if not lead_series.empty else 4
            lead_range = st.slider("Lead Score", min_value=1, max_value=4, value=(lead_min, lead_max))

        with col3:
            # scraped date range
            if df["Scraped Date"].notna().any():
                min_d = df["Scraped Date"].min().date()
                max_d = df["Scraped Date"].max().date()
                d_rng = st.date_input("Scraped Date Range", value=(min_d, max_d))
            else:
                d_rng = None


        # Year filters row (unchanged)
        rowL1, rowT1, rowY2 = st.columns([2, 2, 3])
        with rowL1:
            loc_choices = tokens_for_filter(df.get("Location", pd.Series()))
            sel_loc = st.multiselect("Location", loc_choices, key="loc_general")
        with rowT1:
            TERR_ORDER = [
                "Dade / Monroe",
                "Broward / Palm Beach / Martin / St. Lucie",
                "Greater Tampa / Fort Myers",
                "Orlando",
            ]
            terr_series = df.get("Territory", pd.Series(dtype=str)).fillna("").astype(str)

            # collect unique tokens, respecting your " + " splitter if ever multi-valued
            terr_tokens = set()
            for cell in terr_series:
                for t in re.split(SPLIT_PATTERN, cell):
                    t = t.strip()
                    if t:
                        terr_tokens.add(t)

            # show known territories first if present; otherwise fall back to full known list
            terr_choices = [t for t in TERR_ORDER if t in terr_tokens] or TERR_ORDER
            sel_territory = st.multiselect("Territory", terr_choices, key="territory_general")

        with rowY2:
            comp_choices_g = years_for_filter(df.get("Completion Year", pd.Series()))
            sel_comp_years_g = st.multiselect("Completion Year", comp_choices_g, key="comp_years_general")

        # ----- Apply filters -----
        view = df.copy()

        if sel_proj:
            view = view[view["Project Name"].astype(str).isin(sel_proj)]

        if "Lead Score" in view.columns:
            _ls = pd.to_numeric(view["Lead Score"], errors="coerce").fillna(0).astype(int)
            view = view[(_ls >= lead_range[0]) & (_ls <= lead_range[1])]

        if isinstance(d_rng, (list, tuple)) and all(d_rng):
            start = pd.to_datetime(d_rng[0])
            end   = pd.to_datetime(d_rng[1]) + pd.Timedelta(days=1)
            view = view[(view["Scraped Date"] >= start) & (view["Scraped Date"] < end)]

        # NEW: Location filter (handles multi-value cells separated by " + ")
        if sel_loc:
            exploded_loc = view.assign(
                _loc=view["Location"].fillna("").astype(str).str.split(SPLIT_PATTERN, regex=True)
            ).explode("_loc")
            exploded_loc["_loc"] = exploded_loc["_loc"].str.strip()
            exploded_loc = exploded_loc[exploded_loc["_loc"].isin(sel_loc)]
            view = exploded_loc.drop(columns="_loc").drop_duplicates()

        # Year filters (unchanged)
        if sel_territory:
            exploded_terr = view.assign(
                _terr=view["Territory"].fillna("").astype(str).str.split(SPLIT_PATTERN, regex=True)
            ).explode("_terr")
            exploded_terr["_terr"] = exploded_terr["_terr"].str.strip()
            exploded_terr = exploded_terr[exploded_terr["_terr"].isin(sel_territory)]
            view = exploded_terr.drop(columns="_terr").drop_duplicates()


        if sel_comp_years_g:
            comp_col = view.get("Completion Year", pd.Series(index=view.index, dtype=str)).fillna("").astype(str)
            mask_comp = comp_col.apply(lambda s: bool(set(sel_comp_years_g) & set(YEAR_RE.findall(s))))
            view = view[mask_comp]

        def _fix_url(u):
            if pd.isna(u) or not str(u).strip(): return ""
            u = str(u).strip()
            return u if u.startswith(("http://","https://")) else "https://" + u

        view["Article Link"] = view["Article Link"].map(_fix_url)
        view = reorder_general(view)

        display_order = [
            "Project Name", "Article Link", "Lead Score",
            "Architect","Developer","Possible Engineer","Location", "Territory",
            "Article Title","Article Date","Scraped Date",
            "Article Summary","Milestone Mentions","Planned Mentions", "Justification"
        ]

        view = view.reindex(
            columns=[c for c in display_order if c in view.columns] +
                    [c for c in view.columns if c not in display_order]
        )

        view_for_edit = view.copy()
        view_for_edit["__archive__"] = False

        lead_cols = ["__archive__"]
        view_for_edit = view_for_edit.reindex(
            columns=lead_cols + [c for c in view_for_edit.columns if c not in lead_cols]
        )

        edited = st.data_editor(
            view_for_edit,
            use_container_width=True,
            height=600,
            column_config={
                "__archive__": st.column_config.CheckboxColumn("Archive?", help="Move this row to Archive"),
                "Article Link": st.column_config.LinkColumn("Article Link", display_text="Open"),
                "Article Summary": st.column_config.TextColumn(width="large"),
                "Milestone Mentions": st.column_config.TextColumn(width="large"),
            },
            disabled=[c for c in view_for_edit.columns if c != "__archive__"],
            hide_index=True,
        )


        to_archive_df = edited[edited["__archive__"]].drop(columns="__archive__", errors="ignore")

        col_del, _ = st.columns([1, 3])
        with col_del:
            if st.button("Archive selected rows", type="primary", key="archive_general"):
                count = archive_general_scored_rows(to_archive_df)
                if count > 0:
                    load_general.clear()
                    load_archived_general.clear()
                    st.rerun()
                else:
                    st.info("No rows selected.")

        # Download (without the checkbox column)
        buf = io.StringIO()
        view.to_csv(buf, index=False)
        st.download_button("Download filtered CSV", buf.getvalue(), "general_sfl.csv", "text/csv")



elif page == "Inputs":
    st.title("Open Bids — Inputs")

    st.caption("Expected columns (must match exactly):")
    st.code("\n".join(EXPECTED_INPUT_COLS))

    inputs_df = load_inputs()
    st.subheader("Current inputs (used by the scraper)")
    st.dataframe(inputs_df, use_container_width=True, height=420)

    # Download current inputs (without created_at)
    # if not inputs_df.empty:
    #     out = io.StringIO()
    #     inputs_df[[c for c in inputs_df.columns if c != "created_at"]].to_csv(out, index=False)
    #     st.download_button("Download inputs CSV", out.getvalue(), "inputs.csv", "text/csv")

    st.divider()
    st.subheader("Replace inputs via Excel upload")
    st.caption("Upload an .xlsx/.xls file in the **exact same format** as the example.")

    up = st.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx", "xls"])
    col_a, col_b = st.columns([1, 3])

    if up:
        try:
            preview = pd.read_excel(up, dtype=str, keep_default_na=False)
            if list(preview.columns) == EXPECTED_INPUT_COLS:
                filtered = filter_required_rows(preview)
                after = consolidate_by_bid_name(filtered)
                dropped = len(preview) - len(filtered)
                if dropped > 0:
                    st.info(f"Dropped {dropped} row(s) with blank Project Name and/or Address before consolidation.")
                st.write(f"Preview (Number of open bids after consolidation: {len(after)})")
                st.dataframe(after.head(10), use_container_width=True)
            else:
                st.info("Fix column headers to match exactly before consolidation.")

        except Exception as e:
            st.error(f"Could not read Excel: {e}")


        with col_a:
            if st.button("Replace Inputs (Drop & Load)", type="primary"):
                count = replace_inputs_from_excel(up)
                if count > 0:
                    st.success(f"Replaced inputs with {count} rows.")
                    st.rerun()

    st.info("This action replaces **all** rows in `inputs` with the uploaded file.")

elif page == "Archived":
    st.title("Archived Items")

    # ---- Archived Open Bids (results) ----
    st.subheader("Archived — Open Bids")
    arch_res = load_archived_results()
    if arch_res.empty:
        st.info("No archived Open Bids rows.")
    else:
        view_res = arch_res.copy()

        # Make links clickable if present
        if "Article Link" in view_res.columns:
            def _fix_url(u):
                if pd.isna(u) or not str(u).strip():
                    return ""
                u = str(u).strip()
                return u if u.startswith(("http://","https://")) else "https://" + u
            view_res["Article Link"] = view_res["Article Link"].map(_fix_url)

        # Only allow permanent deletion (no restore)
        view_res["__delete__"] = False

        edited_res = st.data_editor(
            view_res,
            use_container_width=True,
            height=420,
            column_config={
                "__delete__": st.column_config.CheckboxColumn("Delete permanently?"),
                "Article Link": st.column_config.LinkColumn("Article Link", display_text="Open"),
            },
            disabled=[c for c in view_res.columns if c != "__delete__"],
            hide_index=True,
        )

        sel_delete_res = edited_res[edited_res["__delete__"]].drop(columns=["__delete__"], errors="ignore")

        if st.button("Delete Selected - Open Bids (permanent)", type="secondary"):
            n = delete_archived_results_rows(sel_delete_res)
            if n > 0:
                load_archived_results.clear()
                st.rerun()
            else:
                st.info("No rows selected to delete.")

    st.divider()

    # ---- Archived General ----
    st.subheader("Archived — General")
    arch_gen = load_archived_general()
    if arch_gen.empty:
        st.info("No archived General rows.")
    else:
        view_gen = arch_gen.copy()

        if "Article Link" in view_gen.columns:
            def _fix_url(u):
                if pd.isna(u) or not str(u).strip():
                    return ""
                u = str(u).strip()
                return u if u.startswith(("http://","https://")) else "https://" + u
            view_gen["Article Link"] = view_gen["Article Link"].map(_fix_url)

        view_gen["__delete__"] = False

        edited_gen = st.data_editor(
            view_gen,
            use_container_width=True,
            height=420,
            column_config={
                "__delete__": st.column_config.CheckboxColumn("Delete permanently?"),
                "Article Link": st.column_config.LinkColumn("Article Link", display_text="Open"),
                "Article Summary": st.column_config.TextColumn(width="large"),
                "Milestone Mentions": st.column_config.TextColumn(width="large"),
            },
            disabled=[c for c in view_gen.columns if c != "__delete__"],
            hide_index=True,
        )

        sel_delete_gen = edited_gen[edited_gen["__delete__"]].drop(columns=["__delete__"], errors="ignore")

        if st.button("Delete Selected - General (permanent)", type="secondary"):
            n = delete_archived_general_rows(sel_delete_gen)
            if n > 0:
                load_archived_general.clear()
                st.rerun()
            else:
                st.info("No rows selected to delete.")

