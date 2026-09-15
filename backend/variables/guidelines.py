
EOV_DOMAIN = "embarkingonvoyage.com"

# ---------------------------------------------------------------------------
# Node config
# ---------------------------------------------------------------------------
LENGTH_WORDS = {
    "short": "500-600",
    "medium": "700-900",
    "long": "1000-1200",
}

MIN_WORDS_BY_LENGTH = {
    "short": 300,
    "medium": 600,
    "long": 900,
}

MIN_TABLE_ROWS = 5
SEO_SCORE_THRESHOLD = 80
MAX_SEO_FIX_ATTEMPTS = 2
EXTERNAL_LINKS_PER_POST = 5

# ---------------------------------------------------------------------------
# Shared SEO guidance, injected into plan/draft/polish/fix prompts
# ---------------------------------------------------------------------------
SEO_GUIDELINES = """\
This content must be optimized to score 80+ on a Rank Math-style SEO \
analysis. Use the exact FOCUS KEYWORD given to you, verbatim (same wording/ \
casing - do not swap in a synonym or a different grammatical form):

1. The focus keyword must appear in the SEO title, ideally within the first \
   half of the title.
2. The focus keyword must appear in the meta description.
3. The focus keyword must appear naturally within the first 10% of the \
   article's body content (i.e. early in the first section).
4. The focus keyword must appear in at least one subheading (section heading).
5. Keyword density across the full article should land between 0.6% and \
   2.0% of total words - enough to be findable, never stuffed.
6. The SEO title should be 50-60 characters long.
7. The meta description should be 120-160 characters long.
8. The URL slug must be a short, lowercase, hyphenated version of the title \
   that includes the focus keyword.
9. The article must contain at least one outbound link to a reputable \
   external source and at least one internal link to embarkingonvoyage.com.
"""

FORMATTING_EXAMPLE = """\
Example of correctly formatted section content (for a section about "choosing a \
running shoe"):

Picking the right running shoe comes down to matching the shoe to your gait and \
mileage, not just the brand. Most runners fall into one of three categories: \
neutral, overpronator, or supinator. Getting a gait analysis at a specialty running \
store is the fastest way to find out which one you are.

Once you know your gait type, a few features matter more than the rest:

- **Cushioning**: more cushioning reduces impact on long runs but can feel less responsive
- **Drop**: the heel-to-toe height difference, usually 0-12mm
- **Stability**: added support for overpronators, usually a firmer foam wedge

> Runners who replace shoes every 300-500 miles report noticeably fewer overuse \
injuries than those who run shoes into the ground.

Try on shoes later in the day, when your feet are slightly swollen, and always \
walk or jog a few steps in-store before buying.

Example of a section that correctly uses a TABLE because it compares options - note \
it has 5 data rows, the required minimum \
(for a section about "cushioned vs minimalist running shoes"):

Choosing between cushioned and minimalist shoes comes down to how your feet \
currently handle impact, not personal preference alone. The table below lays \
out how the two styles differ on the factors that matter most.

| Factor | Cushioned | Minimalist |
|---|---|---|
| Heel-to-toe drop | 8-12mm | 0-4mm |
| Best for | Long-distance, road running | Short runs, strength-focused training |
| Injury risk if switching too fast | Low | Higher without a gradual transition |
| Typical price range | $120-$180 | $90-$140 |
| Break-in period | Minimal | 2-4 weeks, gradual |

Most runners are better off starting cushioned and transitioning gradually if \
they want to try minimalist shoes.

Example of naturally citing an external source and linking to a relevant EOV \
service with a concrete benefit (use ONLY the URLs you are actually given for \
the real post - these two are illustrative placeholders, not real ones to reuse):

Teams that skip structured gait analysis entirely see far higher return rates \
on running shoes, according to [industry retail research](https://example.com/research). \
If your team is building a fitting tool like this in-house, EOV's \
[AI-Native Digital Product Consulting](https://embarkingonvoyage.com/services/ainative-digital-product-consulting/) \
service specializes in mapping a user journey like this before writing a \
single line of code, which is usually the fastest way to avoid costly rework \
later.
"""
