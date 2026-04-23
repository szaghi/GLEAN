"""ID construction and validation for GLEAN.

ID patterns are defined in AGENTS.md v0.2 §2.3. Every ID is lowercase snake_case,
deterministic, and never renamed after creation — wiki pages and claims cite
them by string match, so an ID change is a breaking schema event.

Public API:
    - `slugify(text: str) -> str` — lowercase snake_case from arbitrary text
    - `source_id_for(type: SourceType, **kwargs) -> str` — deterministic per-type ID
    - `claim_id_for(year: int, source_slug: str, assertion_slug: str) -> str`
    - `notebook_id_for(date: date, slug: str) -> str`
    - `is_valid_<kind>_id(s: str) -> bool` — one per namespace
"""

from __future__ import annotations

import re
from datetime import date

from glean.enums import SourceType

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_SLUG_TRIM = re.compile(r"(^_+|_+$)")

# ID pattern regexes per AGENTS.md v0.2 §2.3.
# Each pattern anchored with ^ and $; callers validate via full match.
_PATTERNS: dict[SourceType, re.Pattern[str]] = {
    # paper: arxiv_<id_normalized>  OR  paper_<firstauthor>_<year>_<shortslug>
    SourceType.PAPER: re.compile(r"^(arxiv_[a-z0-9_.]+|paper_[a-z0-9]+_\d{4}_[a-z0-9_]+)$"),
    # preprint: arxiv_<id>
    SourceType.PREPRINT: re.compile(r"^arxiv_[a-z0-9_.]+$"),
    # repository: repo_<org>_<name>_<commit_short>
    SourceType.REPOSITORY: re.compile(r"^repo_[a-z0-9]+_[a-z0-9_]+_[a-f0-9]{6,12}$"),
    # dataset: data_<short_slug>_<year>
    SourceType.DATASET: re.compile(r"^data_[a-z0-9_]+_\d{4}$"),
    # simulation: sim_<YYYY_MM>_<short_slug>
    SourceType.SIMULATION: re.compile(r"^sim_\d{4}_\d{2}_[a-z0-9_]+$"),
    # talk: talk_<speaker_surname>_<venue>_<year>
    SourceType.TALK: re.compile(r"^talk_[a-z0-9]+_[a-z0-9_]+_\d{4}$"),
    # book: book_<firstauthor>_<year>_<shortslug>
    SourceType.BOOK: re.compile(r"^book_[a-z0-9]+_\d{4}_[a-z0-9_]+$"),
    # standard: std_<body>_<number>
    SourceType.STANDARD: re.compile(r"^std_[a-z0-9]+_[a-z0-9_]+$"),
    # personal_comm: comm_<YYYY_MM_DD>_<short_slug>
    SourceType.PERSONAL_COMM: re.compile(r"^comm_\d{4}_\d{2}_\d{2}_[a-z0-9_]+$"),
    # web_article: web_<domain>_<YYYY_MM_DD>_<short_slug>
    # Domain and slug both allow underscores (slugify("example.com") = "example_com").
    SourceType.WEB_ARTICLE: re.compile(r"^web_[a-z0-9_]+_\d{4}_\d{2}_\d{2}_[a-z0-9_]+$"),
    # notebook: note_<YYYY_MM_DD>_<short_slug>
    SourceType.NOTEBOOK: re.compile(r"^note_\d{4}_\d{2}_\d{2}_[a-z0-9_]+$"),
}

# claim IDs: claim_<year>_<source_slug>_<assertion_slug>.
# Source slug and assertion slug are lowercase snake_case; year is 4 digits.
_CLAIM_PATTERN = re.compile(r"^claim_\d{4}_[a-z0-9_]+$")

# wiki page IDs: <kind>_<short_slug> OR bare slug for well-known terms.
# We permit both forms; lint layer checks that the prefix (if present) matches
# the frontmatter `kind:` field.
_WIKI_PAGE_PATTERN = re.compile(r"^[a-z0-9_]+$")


def slugify(text: str) -> str:
    """Convert arbitrary text to a lowercase snake_case slug.

    Non-alphanumeric runs collapse to single underscores; leading and trailing
    underscores are stripped. Empty output raises `ValueError`.

    >>> slugify("Hello, World!")
    'hello_world'
    >>> slugify("  A--B--C  ")
    'a_b_c'
    """
    s = _SLUG_STRIP.sub("_", text.lower())
    s = _SLUG_TRIM.sub("", s)
    if not s:
        raise ValueError(f"text produced empty slug: {text!r}")
    return s


def source_id_for(source_type: SourceType, **kwargs: object) -> str:
    """Build a deterministic source ID per AGENTS.md v0.2 §2.3.

    Required kwargs vary by type. Common argument names:
        - `arxiv_id: str` (paper/preprint arXiv case)
        - `first_author: str` (paper/book)
        - `year: int` (paper/book/dataset/talk)
        - `slug: str | None` (short topic slug; required for most types)
        - `org: str`, `name: str`, `commit: str` (repository)
        - `year_month: str` like "2026_04" (simulation)
        - `speaker: str`, `venue: str` (talk)
        - `body: str`, `number: str` (standard)
        - `correspondents_slug: str`, `date_ymd: str` (personal_comm)
        - `domain: str`, `date_ymd: str` (web_article)
        - `date_ymd: str` (notebook)

    Raises `ValueError` for missing required kwargs or an ID that does not
    validate against the per-type regex.
    """
    built: str
    match source_type:
        case SourceType.PAPER:
            if "arxiv_id" in kwargs:
                arxiv_id = _require_str(kwargs, "arxiv_id")
                built = f"arxiv_{slugify(arxiv_id)}"
            else:
                first_author = slugify(_require_str(kwargs, "first_author"))
                year = _require_int(kwargs, "year")
                slug = slugify(_require_str(kwargs, "slug"))
                built = f"paper_{first_author}_{year:04d}_{slug}"
        case SourceType.PREPRINT:
            arxiv_id = _require_str(kwargs, "arxiv_id")
            built = f"arxiv_{slugify(arxiv_id)}"
        case SourceType.REPOSITORY:
            org = slugify(_require_str(kwargs, "org"))
            name = slugify(_require_str(kwargs, "name"))
            commit = _require_str(kwargs, "commit").lower()
            if not re.fullmatch(r"[a-f0-9]{6,12}", commit):
                raise ValueError(f"repository commit must be 6-12 hex chars; got {commit!r}")
            built = f"repo_{org}_{name}_{commit}"
        case SourceType.DATASET:
            slug = slugify(_require_str(kwargs, "slug"))
            year = _require_int(kwargs, "year")
            built = f"data_{slug}_{year:04d}"
        case SourceType.SIMULATION:
            year_month = _require_str(kwargs, "year_month")
            if not re.fullmatch(r"\d{4}_\d{2}", year_month):
                raise ValueError(f"simulation year_month must be YYYY_MM; got {year_month!r}")
            slug = slugify(_require_str(kwargs, "slug"))
            built = f"sim_{year_month}_{slug}"
        case SourceType.TALK:
            speaker = slugify(_require_str(kwargs, "speaker"))
            venue = slugify(_require_str(kwargs, "venue"))
            year = _require_int(kwargs, "year")
            built = f"talk_{speaker}_{venue}_{year:04d}"
        case SourceType.BOOK:
            first_author = slugify(_require_str(kwargs, "first_author"))
            year = _require_int(kwargs, "year")
            slug = slugify(_require_str(kwargs, "slug"))
            built = f"book_{first_author}_{year:04d}_{slug}"
        case SourceType.STANDARD:
            body = slugify(_require_str(kwargs, "body"))
            number = slugify(_require_str(kwargs, "number"))
            built = f"std_{body}_{number}"
        case SourceType.PERSONAL_COMM:
            date_ymd = _require_date_ymd(kwargs, "date_ymd")
            slug = slugify(_require_str(kwargs, "correspondents_slug"))
            built = f"comm_{date_ymd}_{slug}"
        case SourceType.WEB_ARTICLE:
            domain = slugify(_require_str(kwargs, "domain"))
            date_ymd = _require_date_ymd(kwargs, "date_ymd")
            slug = slugify(_require_str(kwargs, "slug"))
            built = f"web_{domain}_{date_ymd}_{slug}"
        case SourceType.NOTEBOOK:
            date_ymd = _require_date_ymd(kwargs, "date_ymd")
            slug = slugify(_require_str(kwargs, "slug"))
            built = f"note_{date_ymd}_{slug}"

    if not _PATTERNS[source_type].fullmatch(built):
        raise ValueError(
            f"constructed ID {built!r} does not match pattern for {source_type.value}; "
            f"check that slugs and other parts are lowercase snake_case"
        )
    return built


def claim_id_for(year: int, source_slug: str, assertion_slug: str) -> str:
    """Build a claim ID: `claim_<year>_<source_slug>_<assertion_slug>`.

    Both slugs are passed through `slugify`. Year is 4 digits.
    """
    if year < 1000 or year > 9999:
        raise ValueError(f"year must be 4 digits; got {year}")
    source = slugify(source_slug)
    assertion = slugify(assertion_slug)
    built = f"claim_{year:04d}_{source}_{assertion}"
    if not _CLAIM_PATTERN.fullmatch(built):
        raise ValueError(f"constructed claim ID {built!r} does not validate")
    return built


def notebook_id_for(entry_date: date, slug: str) -> str:
    """Build a notebook ID: `note_<YYYY_MM_DD>_<slug>`.

    Convenience wrapper around `source_id_for(SourceType.NOTEBOOK, ...)`.
    """
    return source_id_for(
        SourceType.NOTEBOOK,
        date_ymd=entry_date.strftime("%Y_%m_%d"),
        slug=slug,
    )


def is_valid_source_id(source_id: str, source_type: SourceType | None = None) -> bool:
    """Return True if `source_id` matches the pattern for its declared type.

    When `source_type` is `None`, return True if the ID matches ANY source-type
    pattern. This is useful for cross-layer validation (e.g., a claim's
    `source:` field) where the caller does not yet know the type.
    """
    if source_type is not None:
        return bool(_PATTERNS[source_type].fullmatch(source_id))
    return any(p.fullmatch(source_id) for p in _PATTERNS.values())


def is_valid_claim_id(claim_id: str) -> bool:
    """Return True if `claim_id` matches the claim ID pattern."""
    return bool(_CLAIM_PATTERN.fullmatch(claim_id))


def is_valid_wiki_page_id(page_id: str) -> bool:
    """Return True if `page_id` matches the wiki page ID pattern.

    Note: this does NOT check that an `entity_*` / `concept_*` / etc. prefix
    matches the page's frontmatter `kind:` field — that is a lint-layer check.
    """
    return bool(_WIKI_PAGE_PATTERN.fullmatch(page_id))


# --- internal helpers ---


def _require_str(kwargs: dict[str, object], key: str) -> str:
    v = kwargs.get(key)
    if not isinstance(v, str) or not v:
        raise ValueError(f"missing or non-string kwarg: {key!r}")
    return v


def _require_int(kwargs: dict[str, object], key: str) -> int:
    v = kwargs.get(key)
    if not isinstance(v, int):
        raise ValueError(f"missing or non-int kwarg: {key!r}")
    return v


def _require_date_ymd(kwargs: dict[str, object], key: str) -> str:
    v = kwargs.get(key)
    if not isinstance(v, str) or not re.fullmatch(r"\d{4}_\d{2}_\d{2}", v):
        raise ValueError(f"kwarg {key!r} must be a YYYY_MM_DD string; got {v!r}")
    return v
