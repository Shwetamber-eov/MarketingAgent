_VALID_MODELS = {"gemini-3.5-flash-lite", "gemma-4-31b-it", "gemma-3-12b-it"}

# Fixed model used specifically for expanding rough visual ideas into full
# image-generation prompts - kept separate from GOOGLE_MODEL since this is a
# smaller, distinct task from the main writing pipeline.
VISUAL_PROMPT_MODEL = "gemma-4-31b-it"

# Fixed model used for Google Search-grounded link discovery. Confirmed to
# support the built-in `google_search` tool. Kept separate from GOOGLE_MODEL
# for the same reason as VISUAL_PROMPT_MODEL - distinct, smaller task.
SEARCH_MODEL = "gemma-4-31b-it"


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


TOP_KEYWORD_QUERY="""Act as an enterprise B2B SEO strategist and search-intent researcher for EmbarkingOnVoyage (EOV) Digital Solutions, an India-headquartered AI-native digital product engineering and technology consulting company.

Website: https://embarkingonvoyage.com/

First understand EOV's current positioning, services, technologies, case studies, and existing content from the website and reliable current web sources.

EOV's relevant capabilities include, but are not limited to:

* AI-native digital product engineering
* Agentic AI
* AI-native product consulting
* digital product experience / UX
* data engineering
* quality assurance and AI-driven test automation
* enterprise software engineering
* cloud and modern application engineering
* Microsoft / Azure ecosystem
* .NET / .NET Core
* Java
* React
* Angular
* Node.js
* SQL
* microservices
* serverless
* legacy modernization
* intelligent automation
* LLM orchestration
* enterprise digital transformation

Primary target audience:

* CTOs
* CIO/technology executives where relevant
* VPs of Engineering
* VPs of Digital
* Heads of Digital
* Heads of Technology
* Heads of Product / Engineering when commercially relevant

Primary market:
India first, while identifying searches that could also have international enterprise relevance.

Goal:
Identify the search queries, topics, and keyword themes that these decision-makers are most likely to search on Google when:

1. they are trying to understand a technology/problem,
2. they are evaluating approaches, vendors, technologies, or implementation options,
3. they are actively looking for a company/partner to help solve the problem.

We want blog opportunities that can help EOV become visible in Google organic results for future buyer searches. Do NOT optimize for generic high-volume consumer keywords or developer-only informational traffic unless the topic clearly connects to an enterprise buyer problem or technology decision.

Research the CURRENT search landscape. Prioritize recent/rising topics and terminology, but do not invent search-volume numbers. If reliable numerical search volume is not available, use qualitative trend evidence instead.

Generate a broad candidate set first, then rank the best opportunities using:

* business value to EOV
* relevance to EOV services
* likelihood the searcher is a CTO, VP, or Digital/Technology head
* commercial/solution intent
* topical relevance and authority potential
* current trend momentum
* likelihood EOV could realistically rank with a strong authoritative article
* content gap/opportunity
* relevance to India
* potential to lead naturally to an EOV service or consultation
* ability to support a cluster of related articles

Classify each candidate by search intent:

* informational
* commercial investigation
* transactional/vendor selection
* mixed

Also classify the buyer stage:

* awareness
* problem identification
* solution evaluation
* vendor evaluation
* implementation

Important:

* Prefer natural search queries and query phrasing that a real business decision-maker would type.
* Include long-tail queries, comparison queries, “how to choose” queries, implementation queries, cost/ROI questions, architecture questions, migration questions, AI adoption questions, and vendor-selection queries where relevant.
* Include emerging enterprise AI topics such as Agentic AI, AI-native engineering, AI-assisted software development, AI modernization, enterprise AI implementation, AI agents, LLM orchestration, AI governance, AI product engineering, and related topics only where they have genuine business relevance.
* Do not produce a list dominated by obsolete 2024/2025 terminology simply because EOV already has old content.
* Do not fabricate exact search volume, CPC, trend percentages, SERP positions, or keyword difficulty.
* Do not treat developer documentation searches as high-value unless they can attract the intended decision-maker audience.
* Avoid branded EOV searches.
* Avoid keywords where EOV has no plausible service/topic authority.
* Avoid duplicate keyword variations that represent essentially the same search intent.

For every shortlisted keyword/topic, provide:

* keyword
* normalized_topic
* search_intent
* buyer_stage
* target_persona
* business_problem
* why_a_decision_maker_would_search_it
* EOV_relevance
* business_value_score from 0-100
* buyer_intent_score from 0-100
* trend_score from 0-100, based on current evidence rather than invented volume
* ranking_opportunity_score from 0-100
* overall_priority_score from 0-100
* recommended_content_type
* suggested_blog_angle
* related_keyword_cluster
* India_relevance
* international_relevance
* evidence_sources
* source_date or recency where available

Content types may include:

* strategic guide
* buyer's guide
* comparison
* implementation guide
* architecture guide
* ROI/business case
* vendor-selection guide
* trend analysis
* executive guide
* technical explainer for decision-makers

Return the FINAL answer as valid JSON only.

Use this JSON structure:

{
"research_summary": {
"company": "EmbarkingOnVoyage",
"market": "India",
"audiences": ["CTO", "VP", "Digital Head"],
"research_date": "YYYY-MM-DD",
"methodology_note": "..."
},
"top_opportunities": [
{
"rank": 1,
"keyword": "...",
"normalized_topic": "...",
"search_intent": "...",
"buyer_stage": "...",
"target_persona": "...",
"business_problem": "...",
"why_a_decision_maker_would_search_it": "...",
"eov_relevance": "...",
"business_value_score": 0,
"buyer_intent_score": 0,
"trend_score": 0,
"ranking_opportunity_score": 0,
"overall_priority_score": 0,
"recommended_content_type": "...",
"suggested_blog_angle": "...",
"related_keyword_cluster": ["...", "..."],
"india_relevance": "...",
"international_relevance": "...",
"evidence_sources": [
{
"title": "...",
"url": "...",
"reason": "...",
"date": "..."
}
]
}
],
"recommended_top_10": ["keyword 1", "keyword 2", "..."]
}

Return at least 25 qualified opportunities before selecting the top 10. Rank the final opportunities from highest to lowest overall priority.
"""

GATHER_LINKS_QUERY="""Act as a B2B technical content researcher and editorial linking strategist for EmbarkingOnVoyage (EOV) Digital Solutions.

Website:
https://embarkingonvoyage.com/

The purpose of this research is to support a future blog about the topic:

{TOPIC}

Primary audience:

* CTOs
* VPs
* Digital Heads
* Technology leaders
* Product/Engineering leaders where relevant

Your task has TWO separate outputs:

PART A — EXISTING EOV INTERNAL LINKS

Search the EOV website and identify only pages that currently exist and are directly relevant to {TOPIC}.

Use ONLY URLs that you can verify actually exist on https://embarkingonvoyage.com/.

Prioritize:

1. service pages directly related to the topic
2. highly relevant solution pages
3. case studies demonstrating the topic or adjacent capability
4. relevant technology pages
5. relevant existing blog/insight pages
6. relevant about/company/partner pages only when contextually useful

Do NOT:

* invent URLs
* infer URLs from page titles
* create hypothetical pages
* link to pages that do not exist
* return irrelevant internal links merely to increase link count

For every internal link, explain exactly why the page is relevant to the article and where it should naturally be linked.

PART B — AUTHORITATIVE EXTERNAL SOURCES

Find authoritative, trustworthy external sources that can be used to validate factual statements, statistics, technical claims, standards, definitions, research findings, market data, security guidance, architecture recommendations, or other claims likely to appear in a high-quality article about {TOPIC}.

Prioritize sources in this order where applicable:

1. official government or regulatory sources
2. official standards/specification bodies
3. official technology/vendor documentation
4. respected research institutions
5. major consulting/research organizations
6. academic/research publications
7. highly reputable industry publications

Prefer primary sources over secondary summaries.

Avoid:

* low-quality SEO blogs
* content farms
* AI-generated websites
* affiliate sites
* anonymous articles
* duplicate syndicated content
* competitor service pages unless no authoritative primary source exists and the source is genuinely useful
* sources that cannot be verified

For each external source, identify:

* exactly what claim it validates
* why it is authoritative
* where in a typical article about {TOPIC} it would be useful
* suggested anchor text
* whether it is primary or secondary evidence
* source publication/update date when available

IMPORTANT LINKING RULES:

* Every URL must be directly verified.
* Never fabricate or guess URLs.
* Prefer deep links to the exact relevant page instead of homepages.
* Do not recommend a link simply because the domain is authoritative; the page itself must support the article.
* External sources are for factual validation and reader value, not artificial SEO link stuffing.
* Prefer 3-8 highly relevant internal links and 5-10 highly authoritative external sources rather than producing a large number of weak links.
* Identify which source is strongest for each important factual claim.
* Flag sources that are outdated or whose information may change frequently.

Also identify:

* factual claims that should probably be cited in the article
* claims that require current verification before publishing
* claims for which an authoritative source could not be found

Return valid JSON only.

Use this schema:

{
"topic": "{TOPIC}",
"research_date": "YYYY-MM-DD",
"internal_links": [
{
"url": "https://embarkingonvoyage.com/...",
"title": "...",
"page_type": "service|case_study|technology|blog|company|other",
"relevance_score": 0,
"why_relevant": "...",
"recommended_article_section": "...",
"suggested_anchor_text": "...",
"linking_purpose": "service_context|supporting_example|deeper_explanation|case_study|conversion"
}
],
"external_links": [
{
"url": "https://...",
"title": "...",
"publisher": "...",
"source_type": "government|standards|official_documentation|research|academic|consulting|industry_publication|other",
"authority_score": 0,
"primary_or_secondary": "primary|secondary",
"claim_supported": "...",
"recommended_article_section": "...",
"suggested_anchor_text": "...",
"why_authoritative": "...",
"publication_or_update_date": "...",
"current_verification_required": true
}
],
"citation_targets": [
{
"claim_type": "statistic|definition|technical_claim|market_claim|security_claim|regulatory_claim|other",
"claim_to_verify": "...",
"best_source_url": "...",
"citation_reason": "..."
}
],
"linking_notes": {
"internal_linking_summary": "...",
"external_citation_summary": "...",
"missing_evidence": ["..."]
}
}
"""