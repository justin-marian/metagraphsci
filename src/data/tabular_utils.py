import ast
from pathlib import Path
from typing import Any
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from constants import DEFAULT_YEAR, REQUIRED_COLUMNS, UNKNOWN_TOKEN


def parse_authors(value: Any) -> list[str]:
    """Normalize the author field into a clean list."""
    # String Unpacking
    # External datasets represent lists of authors in wildly inconsistent ways:
    # pure Python lists, stringified lists ("['A', 'B']"), or delimited strings ("A;B" or "A|B").
    # Standardize everything into an actual Python `list[str]` so 
    # the downstream categorical encoder doesn't crash or create duplicate 
    # garbage embeddings for the same author.
    if isinstance(value, list): 
        return [str(i).strip() for i in value if str(i).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)): 
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list): 
                return [str(i).strip() for i in parsed if str(i).strip()]
        except (ValueError, SyntaxError):
            return []

    return [p.strip() for p in text.replace("|", ";").split(";") if p.strip()]


def prepare_documents(documents: pl.DataFrame, label_column: str = "label") -> tuple[pl.DataFrame, list[str] | None]:
    """Normalize a raw document table to the schema expected by the pipeline."""
    # Strict Boundary Checking
    # "Airlock" for all tabular data entering the project. By validating REQUIRED_COLUMNS 
    # and casting every field to its expected explicit primitive type (int, str, list),
    # guarantee that the PyTorch Dataset class will never encounter 
    # a `None` type or a malformed Series when trying to build tensors.
    df = documents.clone()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: 
        raise ValueError(f"Missing required columns: {missing}")

    # Polars optimization: Execute all casting and filling in parallel
    df = df.with_columns([
        pl.col("doc_id").cast(pl.Int64, strict=False),
        pl.col("title").fill_null("").cast(pl.String),
        pl.col("abstract").fill_null("").cast(pl.String),
        pl.col("venue").fill_null(UNKNOWN_TOKEN).cast(pl.String),
        pl.col("publisher").fill_null(UNKNOWN_TOKEN).cast(pl.String),
        pl.col("authors").map_elements(parse_authors, return_dtype=pl.List(pl.String)),
        pl.col("year").fill_null(DEFAULT_YEAR).cast(pl.Int64, strict=False)
    ])

    label_names: list[str] | None = None
    if label_column in df.columns:
        # Universal Label Coercion
        # Classification loss functions require labels to be continuous integers starting from 0.
        # If the dataset provides string labels (e.g., "Physics", "CS"), automatically map them
        # to integers and extract the mapping list (`label_names`) so evaluation reports can be human-readable.
        if df[label_column].dtype.is_numeric():
            df = df.with_columns(pl.col(label_column).cast(pl.Int64).alias("label"))
            valid_labels = sorted(df.drop_nulls("label")["label"].unique().to_list())
            label_names = [str(lbl) for lbl in valid_labels]
        else:
            classes = sorted(df.drop_nulls(label_column)[label_column].cast(pl.String).unique().to_list())
            mapping = {name: idx for idx, name in enumerate(classes)}
            df = df.with_columns(pl.col(label_column).cast(pl.String).replace(mapping).cast(pl.Int64).alias("label"))
            label_names = classes

    return df, label_names


def load_documents(path: str | Path, label_column: str = "label") -> tuple[pl.DataFrame, list[str] | None]:
    path = Path(path)
    if path.suffix == ".csv": 
        frame = pl.read_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        frame = pl.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return prepare_documents(frame, label_column=label_column)


def create_encoders(documents: pl.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Build categorical vocabularies for venue, publisher, and authors."""
    # 0-Indexed OOV Encoders
    # PyTorch layers need a vocabulary size. Extract all unique 
    # metadata strings and map them to integers starting at 1. Index 0 is exclusively 
    # reserved for `UNKNOWN_TOKEN` (or missing data). This allows the embedding matrix 
    # to cleanly learn a specific, safe representation for "Missing Field".
    v_unique = documents.drop_nulls("venue")["venue"].cast(pl.String).unique().sort().to_list()
    p_unique = documents.drop_nulls("publisher")["publisher"].cast(pl.String).unique().sort().to_list()
    
    v_enc = {name: idx + 1 for idx, name in enumerate(v_unique)}
    p_enc = {name: idx + 1 for idx, name in enumerate(p_unique)}
    
    # Explode list of authors natively in Polars to extract unique elements
    a_unique = documents.select(pl.col("authors").explode()).drop_nulls()["authors"].unique().sort().to_list()
    a_enc = {name: idx + 1 for idx, name in enumerate(a_unique)}
    
    for enc in (v_enc, p_enc, a_enc): 
        enc[UNKNOWN_TOKEN] = 0
    return v_enc, p_enc, a_enc


def split_documents(
    documents: pl.DataFrame, test_size: float, val_size: float,
    seed: int, strategy: str, time_column: str = "year"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create train, validation, and test splits."""
    if "label" not in documents.columns or documents["label"].is_null().any():
        raise ValueError("split_documents requires a fully labeled dataframe.")

    strat = strategy.lower()

    # Time-based vs. Random Evaluation
    # Graph datasets are highly susceptible to data leakage. If a random split is used 
    # on a continuously evolving graph, the model learns to predict paper topics using 
    # citations from the *future*, heavily inflating metrics.
    # The 'time' strategy forces the model to train entirely on older papers and predict 
    # the strictly newer test set, simulating real-world production constraints.
    if strat == "time":
        ranked = documents.sort([time_column, "doc_id"])
        n = ranked.height
        n_test, n_val = max(1, int(round(n * test_size))), max(1, int(round(n * val_size)))

        train_df = ranked.slice(0, max(0, n - n_test - n_val))
        val_df = ranked.slice(max(0, n - n_test - n_val), n_val)
        test_df = ranked.slice(n - n_test, n_test)
        
        if train_df.is_empty() or val_df.is_empty() or test_df.is_empty(): 
            raise ValueError("Time-based split produced an empty split.")
        return train_df, val_df, test_df

    if strat != "random": 
        raise ValueError(f"Unknown split strategy: {strat}")

    # Random strategy uses stratified sampling to ensure rare classes are balanced
    # We temporarily cast to Pandas solely to leverage Sklearn's exact, battle-tested stratification logic
    pd_docs = documents.to_pandas()
    train_pd, test_pd = train_test_split(pd_docs, test_size=test_size, random_state=seed, stratify=pd_docs["label"])
    train_pd, val_pd = train_test_split(train_pd, test_size=val_size / (1.0 - test_size), random_state=seed, stratify=train_pd["label"])
    
    return pl.from_pandas(train_pd), pl.from_pandas(val_pd), pl.from_pandas(test_pd)


def create_low_label_split(documents: pl.DataFrame, label_ratio: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the training set into labeled and unlabeled subsets."""
    # Simulating Semi-Supervised Data
    # To test the model's Pseudo-Labeling capabilities,  artificially mask out a large 
    # portion of the training set (leaving only 1/5/10/25% labeled). 
    # Crucially, enforce balanced masking (per-class sampling). If IT just masked blindly, 
    # minor classes might lose all their labels, collapsing the supervised training branch.
    rng = np.random.default_rng(seed)
    labeled_parts, unlabeled_parts = [], []

    for label in sorted(documents.drop_nulls("label")["label"].cast(pl.Int64).unique().to_list()):
        group = documents.filter(pl.col("label") == label)
        order = rng.permutation(group.height)
        keep_count = min(group.height, max(1, int(round(group.height * label_ratio))))
        
        labeled_parts.append(group[order[:keep_count]])
        unlabeled_parts.append(group[order[keep_count:]])

    labeled = pl.concat(labeled_parts, how="vertical").sample(fraction=1.0, seed=seed, shuffle=True)
    unlabeled = pl.concat(unlabeled_parts, how="vertical").sample(fraction=1.0, seed=seed, shuffle=True)
    
    unlabeled = unlabeled.with_columns(pl.lit(None).cast(pl.Int64).alias("label"))
    return labeled, unlabeled


def build_year_lookup(documents: pl.DataFrame) -> dict[int, int]:
    """Generates a fast O(1) dictionary mapping a document ID to its publication year."""
    # Instantly zips the contiguous underlying lists rather than slowly yielding rows
    return dict(zip(documents["doc_id"].to_list(), documents["year"].to_list()))
