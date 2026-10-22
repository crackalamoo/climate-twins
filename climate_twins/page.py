"""The web UI's page: index.html with a <head> describing the query in its URL, for link previews
(crawlers don't run the UI's JavaScript) and the first tab title. Names follow the UI's
(web/app.ts), which retitles the tab as the query changes."""

from html import escape

SITE_NAME = "Climate Twins"
SITE_DESCRIPTION = "Find places around the world whose climate is most like yours."  # index.html's intro
# How many matches the description names; it counts the rest.
NAMED_MATCHES = 3
IMAGE_SIZE = (1200, 630)  # the card's: Open Graph's recommended size, also X's large-image card


def possessive(name: str) -> str:
    return f"{name}'" if name.lower().endswith("s") else f"{name}'s"


def short_headline(place: dict) -> str:
    """The tab title's: "Portland's climate twins" in or near a city, else the label's first line as
    the UI shows it: "97 km NW of Cairns", "Western Australia"."""
    if place["tier"] in ("exact", "near"):
        return f"{possessive(place['name'])} climate twins"
    if place["tier"] == "offset":
        return f"{place['km']:,.0f} km {place['direction']} of {place['name']}"
    return place["name"]


def subject(place: dict) -> str:
    """Whose climate twins the page shows: "Portland" in or near a city, else "a spot 97 km NW of
    Cairns", "a spot in Western Australia"."""
    if place["tier"] in ("exact", "near"):
        return place["name"]
    return f"a spot {'' if place['tier'] == 'offset' else 'in '}{short_headline(place)}"


def headline(place: dict) -> str:
    """The preview's (og:title and the card's): the short one in or near a city, else saying what
    the page is: "Climate twins of a spot 97 km NW of Cairns"."""
    if place["tier"] in ("exact", "near"):
        return short_headline(place)
    return f"Climate twins of {subject(place)}"


def card_lines(place: dict) -> tuple[str, str]:
    """The preview card's kicker and headline: the headline is the place as the panel's first line
    names it (the UI's placeLines), the kicker what of it the card shows. A city's own name ("Climate
    twins of" / "Portland"), else a spot's ("... of a spot" / "115 km NW of Cairns", "in Iran", or
    "near Portland" for a point across a border from it)."""
    if place["tier"] in ("exact", "near") and not place["in_area"]:
        return "Climate twins of", place["name"]
    if place["tier"] == "offset":
        headline = short_headline(place)
    elif place["tier"] == "region":
        headline = f"in {place['name']}"
    else:
        headline = f"near {place['name']}"
    return "Climate twins of a spot", headline


def image_alt(place: dict | None) -> str:
    if place is None:
        return f"A world map titled {SITE_NAME}"
    return f"World map shading the places whose climate is most like {subject(place)}, with the closest numbered"


def short_name(place: dict | None) -> str | None:
    """A match as the description lists it: the city it's in or near, else its own region or
    country (the UI's shortPlaceName), then the country unless that's the name."""
    if place is None:
        return None
    if place["tier"] in ("exact", "near"):
        name, country = place["name"], place["country"]
    else:
        area = place["area"] or {}
        name, country = area.get("admin1") or area.get("country"), area.get("country")
    if not name:
        return None
    return f"{name}, {country}" if country and country != name else name


def description(place: dict, matches: list[dict | None]) -> str:
    """"The places on Earth whose climate is most like Portland, Oregon, United States: Wellington,
    New Zealand; …, and 17 more." `matches` are the matches' Place labels, best first."""
    names = [s for s in map(short_name, matches) if s][:NAMED_MATCHES]
    if not names:
        return SITE_DESCRIPTION
    where = place["label"] if place["tier"] == "exact" and not place["in_area"] else f"the spot {place['label']}"
    rest = len(matches) - len(names)
    return f"The places on Earth whose climate is most like {where}: {'; '.join(names)}{f', and {rest} more' if rest else ''}."


def render(template: str, url: str, image_url: str, place: dict | None = None, matches: list[dict | None] | None = None) -> str:
    """index.html with its <title> replaced by the page's title and preview tags: for the query's
    Place label and its matches' when given, else the site's. `image_url`: its card (preview.py)."""
    if place is None:
        title, og_title, desc = SITE_NAME, SITE_NAME, SITE_DESCRIPTION
    else:
        og_title = headline(place)
        title, desc = f"{short_headline(place)} · {SITE_NAME}", description(place, matches or [])
    tags = [
        f"<title>{escape(title, quote=False)}</title>",
        f'<meta name="description" content="{escape(desc)}">',
        f'<link rel="canonical" href="{escape(url)}">',
        '<meta property="og:type" content="website">',
        f'<meta property="og:site_name" content="{SITE_NAME}">',
        f'<meta property="og:title" content="{escape(og_title)}">',
        f'<meta property="og:description" content="{escape(desc)}">',
        f'<meta property="og:url" content="{escape(url)}">',
        f'<meta property="og:image" content="{escape(image_url)}">',
        f'<meta property="og:image:width" content="{IMAGE_SIZE[0]}">',
        f'<meta property="og:image:height" content="{IMAGE_SIZE[1]}">',
        f'<meta property="og:image:alt" content="{escape(image_alt(place))}">',
        '<meta name="twitter:card" content="summary_large_image">',
    ]
    marker = f"<title>{SITE_NAME}</title>"
    if marker not in template:
        raise ValueError(f"index.html has no {marker}")
    return template.replace(marker, "\n  ".join(tags), 1)
