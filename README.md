# India Labour Law RAG

**A retrieval system over India's labour codes and, more importantly, an honest measurement of whether it actually works. I played with Claude to help with the code and then ran multiple iterations chunking strategies aiming to improve the overall recall@4 number.**

```
Recall@4   63.6%  ──────────────────▶  78.1%
MRR        0.531  ──────────────────▶  0.681
                  Tested 20 configurations
                  Then rejected 7 hypotheses
```

---

## The problem

In November 2025, India replaced 29 separate labour laws with four consolidated codes. Roughly 500 million workers are covered by them. Almost none of those workers can read them.

Here is the answer to "how much notice do I get before being laid off," as the statute puts it:

> *70. No worker employed in any industry who has been in continuous service for not less than one year under an employer shall be retrenched by that employer until— (a) the worker has been given one month's notice in writing indicating the reasons for retrenchment and the period of notice has expired, or the worker has been paid in lieu of such notice, wages for the period of the notice...*

The answer is in there. It's on page 33 of a 78-page PDF, and you had to know the word "retrenchment" to find it.

**So: ask in plain English, get the right passage back.**

---

## Why not just ask ChatGPT?

Because it answers from memory. That memory is unversioned, might be from before these codes existed, might be about a different country entirely, and — the real problem — **you cannot check it.** A confident paragraph about your employment rights with no source is worse than no answer.

The fix is retrieval-augmented generation:

```
   "Can my boss fire me without warning?"
                  │
                  ▼
   RETRIEVE   find the 4 passages most likely
              to contain the answer
                  │
                  ▼
   AUGMENT    paste ONLY those passages
              into the prompt
                  │
                  ▼
   GENERATE   answer using nothing else,
              cite what was used
```

The model never sees the corpus. It sees four passages and a question. **Everything interesting in this project is about choosing those four passages well.**

---

## Act I — The documents fought back

| Code | Source | Pages |
|---|---|---|
| Industrial Relations Code, 2020 | PRS India | 78 |
| Code on Social Security, 2020 | labour.gov.in | 118 |
| Occupational Safety & Health Code, 2020 | dgfasli.gov.in | 86 |

Government portals reorganise, and **four of my first eight URLs returned 404.** The working link for the Social Security Code turned out to be `b0620548445580767b5c0d18c95c26f7.pdf` — a content hash, because the ministry had migrated CMS and renamed every file.

So the first thing I built was a **quality gate**: four automated checks run against every document before it's allowed into the corpus.

```
code                    ch/pg   empty  sections  readable  verdict
────────────────────────────────────────────────────────────────────
wages                     437     0%         0   100.0%    REVIEW x
osh                      3148     0%       212   100.0%    GO
industrial_relations     3437     0%       126   100.0%    GO
social_security          3253     0%       314   100.0%    GO
```

**It immediately earned its keep.** The Code on Wages PDF scored 437 characters per page, detected zero sections, and returned zero hits for every search term including "minimum wage," in the Code on *Wages*. It was a cover document, not the Act. Dropped before it could poison anything.

That's why minimum wage and bonus questions fail here. Known, deliberate gap.

---

## Act II — Building the thing

### Chunking

250 pages won't fit in a prompt. Whole pages don't work either a page holds several unrelated provisions, so most of what we'd send is noise.

So: **1,000-character passages, overlapping by 150.**

The overlap is for a reason. A provision sitting on a boundary would be cut in half and lost from both passages. Repeating the tail means it appears whole somewhere — two chances to be found.

**Result: 1,104 chunks.**

### Embeddings

Each chunk becomes ~768 numbers representing its meaning. Similar meanings land in nearby coordinates. This is what lets *"money I get when I leave"* match a gratuity provision that shares none of those words.

### Retrieval, in three layers

```
  QUESTION
     │
     ├──────────────┬──────────────┐
     ▼              ▼              │
  BM25          BGE                │  two searches that fail
  keyword       semantic           │  in OPPOSITE ways
     │              │              │
     └──────┬───────┘              │
            ▼                      │
        RRF fusion                 │  fuse by RANK, not score
            │                      │  (scores aren't comparable)
            ▼                      │
     40 candidates                 │
            │                      │
     ┌──────┴───────┐              │
     ▼              ▼              │
  ms-marco       BGE               │  two cross-encoders,
  reranker       reranker          │  also fused by rank
     │              │              │
     └──────┬───────┘              │
            ▼                      │
     TOP 4 PASSAGES ◄──────────────┘
```

**Why two search methods.** Embeddings understand meaning but blur exact words, mine confused "fire me" with mine *fire and flooding* provisions. BM25 keyword search would never make that mistake, but fails completely when your wording differs from the statute's. **Opposite failure modes is exactly the condition where combining two methods pays.**

**Why reranking.** The embedder is a *bi-encoder* - it encodes the question and each passage separately, so it's fast, but it never sees them together and can't reason about how they relate. A *cross-encoder* reads the pair and scores relevance properly, but is far too slow to run over 1,104 chunks. So: fast model shortlists 40, slow model reorders them. **Cheap filter, expensive ranker.**

---

## Act III - Had to do it!

Here's what makes RAG dangerous: **if retrieval grabs the wrong passages, the AI still writes a fluent, confident, completely wrong answer.** Nothing looks broken. You'd never know.

So I built a test: 50 questions, spread deliberately across difficulty by *vocabulary distance from the statute's own language*.

| Band | Phrasing | Example |
|---|---|---|
| **easy** | uses the legal term | "What notice period is required before *retrenching* a worker?" |
| **medium** | natural, overlapping words | "How much warning before *laying people off*?" |
| **hard** | pure layperson | "Can my boss *fire* me?" |
| **unanswerable** | not in the corpus | "What's the GST rate on consulting?" |

The easy and hard versions **point at the same provision**. That isolates one variable 'vocabulary', while holding the correct answer constant.

For each question I recorded which passages genuinely contain the answer. That's the **gold** standard. Then:

- **Recall@k** - was a correct passage in the top k? The ceiling on the whole system. If the right text isn't retrieved, the answer cannot be right.
- **MRR** - *where* did it rank? Position 1 scores 1.0, position 4 scores 0.25. Recall can't tell those apart. MRR can.

---

## Two bugs I found in my own methodology

These cost me hours. They're also the most useful thing in this repository.

### Bug 1 - I graded the test with the system's own homework. LOL.

My first answer key was built by asking an LLM to pick the correct passage **from the top 8 results of my own semantic search.**

Read that again. Gold could only ever be something the retriever had already found. Recall@10 came back at a beautiful, meaningless **100%**.

```
   retriever ──▶ gold labels ──▶ graded by
       ▲                              │
       └──────────────────────────────┘
          circular. always 100%.
```

**Fix:** build the candidate pool from *keyword* search first - a method that works on a completely different principle, and can surface passages the embedder never finds. The honest baseline came back at **63.6%**. A worse number and a real one.

### Bug 2 — chunk IDs don't survive re-chunking

Gold was stored as IDs like `social_security_p26_c2` - *the third 1,000-character window on page 26*. That name only means anything under one chunking scheme. Change the chunking and it points at different text, or nothing at all.

Every new chunking strategy would have scored **0%**, and I'd have concluded they were all terrible.

**Fix — anchor matching.** Store a distinctive 120-character *snippet* from each gold passage, and count a hit when the retrieved text *contains* it.

> Instead of *"page 26, paragraph 3 of the paperback,"* you say *"the paragraph containing the phrase **one month's notice in writing**."* That works in any edition.

Verified: 31 of 33 anchors were findable under every chunking variant tested, so the comparison was fair.

---

## Results

```
BASELINE   fixed-size chunking · MiniLM · semantic only
           Recall@4  63.6%   MRR 0.531
           easy 77%  |  medium 64%  |  hard 33%

SHIPPED    fixed-size chunking · BGE · hybrid BM25 · dual reranker
           Recall@4  78.1%   Recall@8  87.5%   MRR 0.681
           easy 92%  |  medium 100%  |  hard 50%
```

### What moved the needle

| Change | Effect |
|---|---|
| **MiniLM to BGE embeddings** | MRR 0.531 to 0.616 — **the single biggest lever** |
| Hybrid BM25 + semantic | medium questions 64% to 71% |
| Cross-encoder reranking | hard questions 33% to 50% |
| **Two rerankers, rank-fused** | Recall@4 75% to 81%, MRR to 0.708 |
| Label audit (4 of 33 corrected) | see below |

**The embedding swap mattered more than every chunking experiment combined** - and I'd have bet against that. It only surfaced because it was measured.

### What I was wrong about

More hypotheses died than survived. That's what evaluation looks like when you're not fooling yourself.

| Hypothesis | Result |
|---|---|
| Section-aware chunking — one chunk per legal provision | **27–58%**, worse in all 6 variants |
| Smaller section chunks (400 chars) | 27.3% — much worse |
| Adding overlap to section chunks | no effect |
| BGE-**large** instead of BGE-base | worse |
| Query expansion — rewrite plain English into legal terms | hurt hard questions 50% to 33% |
| `bge-reranker` alone | best Recall@10 (87.9%), **worst** Recall@1 (24.2%) |
| Multi-query retrieval | matched the winner, at higher cost |
| A third reranker | Recall@4 flat, MRR down |
| Wider candidate pools (60, 80) | **identical results** |

Section-aware chunking is the one that stings. One chunk per legal provision is *obviously* the right idea. It was consistently worse across six variants, and after testing dilution, chunk size, and missing overlap as explanations — none of which held — I stopped and wrote it down as unexplained. Which is more honest than inventing a tidy story.

### Knowing when to stop

Three independent signals said the ceiling was reached, not that I got bored:

1. Candidate pools of 40, 60 and 80 gave **identical** metrics — the pool was already sufficient
2. A third reranker traded Recall@1 for Recall@8 with no net gain
3. Hard questions sat at **exactly 50% in every single configuration tested**

Beyond that point, with 32 questions each worth 3.1 percentage points, "improving" would have meant tuning until two specific questions moved. That's fitting to the test set, not building a better system.

---

## Auditing my own answer key

Five questions never surfaced gold in the top 10 under *any* of 20 configurations. That pattern usually means the labels are wrong, not the retrieval.

| # | Question | Diagnosis | Action |
|---|---|---|---|
| 11 | how long must records be preserved | the codes say *maintain registers* — no retention period exists | reclassified unanswerable |
| 38 | what if I'm hurt and can't work | gold was the Sixth Schedule: actuarial multipliers, not the entitlement | relabelled |
| 42 | hired via an agency, am I protected | gold was about licensing authorities, not worker protection | relabelled |
| 7 | employer PF contribution rate | gold correct, anchor landed awkwardly | added neighbour |
| 37 | can my boss fire me without warning | **gold correct — retrieval genuinely failed** | **left broken** |

**The distinction that matters:** correcting a wrong label repairs the measuring instrument. Tuning parameters against the same 32 questions until the number looks good is overfitting. Different things, and I've said which one I did.

Question 37 stays broken on purpose. It's the honest example of what this system still gets wrong.

---

## What doesn't work

**Plain-English questions fail half the time.** 50%, versus 92% for questions using legal terminology. Nothing I tried closed that gap.

The canonical failure, still live in the repo:

```
Q: "Can my boss fire me without any warning at all?"

   x [0.454] OSH Code p.35 — "...restricting the area that might be
              affected by FIRE or FLOODING in the mine..."
   x [0.451] OSH Code p.35 — "...isolation of the part of the mine..."

   > Should have been: IR Code p.33, section 70 — one month's notice
```

In a general-purpose embedding space, *fire someone* and *fire in a mine* really are close together. Fixing it needs a legal-domain-trained embedder or a proper synonym layer. Both are their own projects.

---

## Limitations

- **32 scored questions.** Each is worth 3.1 points, so differences under ~6 points are noise.
- **The answer key is LLM-labelled**, with 4 entries corrected by hand. Not independently verified.
- **Three of four codes.** No Code on Wages, so minimum wage and bonus questions fail by design.
- **Difficulty labels: one annotator.** No inter-annotator agreement measured.
- **Recall is strict** — a neighbouring passage covering the same provision scores as a miss, so real-world usefulness is somewhat higher than the number suggests.
- **Not legal advice.** 78% retrieval accuracy is nowhere near good enough to rely on for a decision that affects someone's livelihood.

---

## Reproducibility

Every number above came from a **clean clone on a fresh machine**, not from the session where it was built:

```bash
git clone https://github.com/itsimranmd/India-labour-rag.git
cd India-labour-rag
pip install -r requirements.txt

python ingest.py        # 1,104 chunks
python evaluate.py      # Recall@4 78.1%, MRR 0.681
```

That test found a real bug — `requirements.txt` had been written with escaped newlines and pip refused to parse it. Which is the point of running it.

My working session reported 81.2%. The clean clone reports **78.1%**, a one-question difference from library versions. **The clean-clone number is the one quoted throughout**, because it's the one anyone else will get.

Note: the embedding pass takes ~30 seconds on a GPU and **7m33s on CPU**.

---

## What's in here

```
ingest.py            download, strip gazette boilerplate, chunk
retrieve.py          BGE + BM25 hybrid + dual-reranker ensemble
answer.py            grounded generation with citations
evaluate.py          Recall@k, MRR, anchor matching
eval/questions.json  the answer key — the most valuable file here
```

**Stack:** Python · sentence-transformers · BGE-base-en-v1.5 · rank-bm25 · ChromaDB · OpenAI gpt-4o-mini

---

## Questions you might have

**Would a smarter generation model help?**
Barely. If retrieval hands over the wrong passages, a better model writes a better-worded wrong answer. 78.1% is a *retrieval* ceiling. That's why this project is about retrieval and the generation step is 20 lines.

**Why isn't there a live demo?**
The interesting half is retrieval, and it's fully reproducible from the repo in two commands. A hosted demo would need my API key on a public endpoint, and the generation step is the least novel part of the system.

**Is this production-ready?**
No, and the limitations section says why. It demonstrates the architecture and, more to the point, the evaluation method.

**What would you do with another week?**
Fine tune the embedder on legal text, that's where the plain-English gap lives. Add the Code on Wages. Get a second annotator on the answer key. And work out why section-aware chunking lost, because I'm still figuring out too. HMU if you want to know anything beyond this.

---

## On AI assistance

Built with heavy AI assistance for the code. The evaluation design, the failure analysis, and the decisions about what to test and when to stop are mine - including catching the circular answer key, and the call that optimising rank-1 would lift Recall@4, which is where the two-reranker ensemble came from.

Every claim in this README is reproducible from a clean clone. That was the point.
