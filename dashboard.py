import os
import io
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, inspect
import re

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Project Updates", layout="wide")

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
    "Bidder Owner","Must-Close","Location","Projected Total","Address","Project Name",
    "Plan & Specs/ Job Info"
]

ARTICLE_COLS = ["Article Title", "Article Date", "Scraped Date", "Article Link"]

# ----------------------------
# Schema helpers
# ----------------------------

JOIN_DELIM = " + "
SPLIT_PATTERN = r"\s*\+\s*"   # split ONLY on ';' or '+' (never commas)

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


def quote_ident(col: str) -> str:
    # Double-quote SQL identifiers and escape internal quotes
    return '"' + col.replace('"', '""') + '"'

def get_table_columns(table: str):
    insp = inspect(engine)
    if not insp.has_table(table):
        return None
    cols = [c["name"] for c in insp.get_columns(table)]
    return cols

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
    # results = all input cols + article fields
    desired_cols = expected_input_cols + ARTICLE_COLS
    existing = get_table_columns("results")

    if existing is None:
        # Create results fresh
        with engine.begin() as conn:
            cols_sql = ",\n".join(f"{quote_ident(c)} TEXT" for c in desired_cols)
            ddl = f"CREATE TABLE results ({cols_sql})"
            conn.execute(text(ddl))
            # Index on scraped_date for ordering
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_results_scraped ON results({quote_ident('Scraped Date')})"))
        return

    # If columns mismatch, recreate (simple, predictable)
    existing_set = set(existing)
    desired_set = set(desired_cols)
    if existing_set != desired_set or len(existing) != len(desired_cols):
        with engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE IF EXISTS results;")
        with engine.begin() as conn:
            cols_sql = ",\n".join(f"{quote_ident(c)} TEXT" for c in desired_cols)
            ddl = f"CREATE TABLE results ({cols_sql})"
            conn.execute(text(ddl))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_results_scraped ON results({quote_ident('Scraped Date')})"))

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

# Bootstrap/migrate schema every start
ensure_inputs_table(EXPECTED_INPUT_COLS)
ensure_results_table(EXPECTED_INPUT_COLS)
seed_inputs_if_empty()

# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_data(ttl=60)
def load_inputs():
    return pd.read_sql('SELECT * FROM inputs ORDER BY "created_at" DESC', engine)

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
    """Replace entire inputs table with consolidated rows by Bid Name."""
    try:
        df_up = pd.read_excel(file, dtype=str)
        if list(df_up.columns) != EXPECTED_INPUT_COLS:
            raise ValueError("Column names/order must match the example exactly.")

        # consolidate to one row per Bid Name
        clean = consolidate_by_bid_name(df_up)

        with engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM inputs;")
        clean.to_sql("inputs", con=engine, if_exists="append", index=False, method="multi")

        load_inputs.clear()  # bust cache
        return len(clean)
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return 0


# ----------------------------
# UI
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Open Bids"])

if page == "Dashboard":
    st.title("Open Bids Dashboard")
    results = load_results()
    if results.empty:
        st.info("No results yet. When your scraper writes to the `results` table, they will appear here.")
    else:
        # Optional filters: if "Project Name" exists
        row1a, row1b = st.columns([2, 2])
        row2a, row2b = st.columns([2, 3])

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

        st.dataframe(
            view,
            use_container_width=True,
            height=600,
            column_config={
                "Article Link": st.column_config.LinkColumn(
                    "Article Link",
                    help="Open article in a new tab",
                    display_text="Click to open"
                )
            }
        )

        # Download filtered CSV
        buf = io.StringIO()
        view.to_csv(buf, index=False)
        st.download_button("Download filtered CSV", buf.getvalue(), "project_updates.csv", "text/csv")

elif page == "Open Bids":
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
            preview = pd.read_excel(up, dtype=str)
            

            if list(preview.columns) == EXPECTED_INPUT_COLS:
                after = consolidate_by_bid_name(preview)
                st.write(f"Preview (Number of open bids: {len(after)}):")
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
