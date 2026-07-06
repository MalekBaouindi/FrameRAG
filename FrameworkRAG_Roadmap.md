    # FrameworkRAG — Assistant de comparaison d'écosystèmes RAG/LLM

**Pitch en une phrase :** un système RAG + Graph RAG + multi-agent qui ingère la documentation de plusieurs frameworks (LangChain, LlamaIndex, Haystack, CrewAI, AutoGen...) et répond à des questions de comparaison, de compatibilité et de migration entre eux — en citant les sources exactes et en signalant les breaking changes entre versions.

---

## 1. Pourquoi ce projet est fort pour un recrutement AI Engineering

- Tu résous un vrai problème que **toi-même tu vis** en apprenant ces outils (la doc de chaque framework est dispersée, les versions changent vite, comparer coûte du temps).
- Tu démontres la stack complète demandée dans l'annonce CareersBang : RAG classique, Graph RAG, LLMs, embeddings, bases vectorielles, hybrid search, re-ranking, prompt engineering, IA agentique, évaluation, déploiement.
- Le graphe a un **vrai sens sémantique** ici : versions, dépréciations, dépendances entre modules, équivalences fonctionnelles entre frameworks — ce n'est pas un graphe plaqué artificiellement sur du texte plat.
- Narratif d'entretien tout fait : *"j'ai construit un outil RAG pour m'aider à apprendre l'écosystème RAG."*

---

## 2. Portée du projet (scope réaliste)

Ne vise pas *tous* les frameworks RAG existants. Choisis **3 à 4 frameworks** pour garder un scope gérable :

- **LangChain** (le plus documenté, bon pour prototyper l'ingestion)
- **LlamaIndex** (fort sur l'indexation/retrieval, bon contraste avec LangChain)
- **Haystack** (deepset — architecture pipeline différente, bon pour montrer que tu captures des paradigmes différents)
- **CrewAI** ou **AutoGen** (optionnel, pour la partie "agents multi-frameworks" si tu as le temps)

Périmètre des données : documentation officielle (pas le code source complet des libs, ce serait un projet à part) + éventuellement les release notes / changelogs pour la détection de breaking changes.

---

## 3. Stack technique complète

| Couche | Outil retenu | Alternative | Justification |
|---|---|---|---|
| Scraping / ingestion doc | Playwright + `readability-lxml` ou scraping direct des sites doc (souvent statiques, Sphinx/MkDocs) | `firecrawl`, `trafilatura` | Tu maîtrises déjà Playwright depuis ton projet précédent |
| Découpage (chunking) | `langchain-text-splitters` (RecursiveCharacterTextSplitter, ou MarkdownHeaderTextSplitter) | `semantic-chunkers` | La doc est structurée en headers → chunker par structure sémantique, pas juste par taille fixe |
| Embeddings | `BAAI/bge-m3` (multilingue, bon sur code+texte) ou `text-embedding-3-small` (OpenAI, si budget dispo) | `nomic-embed-text` | bge-m3 tourne en local, gratuit, gère bien le mélange texte/code des docs techniques |
| Base vectorielle | **Qdrant** (Docker, hybrid search natif) | ChromaDB, Milvus | Qdrant a le meilleur support hybride dense+sparse out-of-the-box, exactement ce que demande l'annonce |
| Sparse retrieval | BM25 via Qdrant natif, ou `rank_bm25` | Elasticsearch | Nécessaire pour matcher les noms exacts de fonctions/classes (`ChatOpenAI`, `VectorStoreIndex`...) que le dense embedding rate parfois |
| Re-ranking | `BAAI/bge-reranker-v2-m3` (local) | Cohere Rerank API | Re-ranking local = gratuit, pas de dépendance API externe pour cette étape |
| Graphe de connaissances | **Neo4j** (Community Edition, Docker) | ArangoDB | Standard de l'industrie, énorme écosystème, requêtes Cypher lisibles |
| Extraction d'entités/relations pour le graphe | LLM (function calling structuré) + règles heuristiques sur la structure de la doc | `spaCy` + NER custom | Un LLM avec sortie structurée (JSON) extrait bien "Classe X hérite de Y", "Méthode Z dépréciée depuis version N" depuis du texte de doc |
| LLM (génération) | **Mistral** (API, moins cher, bon en FR/EN) ou **DeepSeek** | Qwen, Llama via Groq/Together | Coût maîtrisé, l'annonce cite ces modèles explicitement |
| Orchestration agents | **LangGraph** | CrewAI, AutoGen | Explicitement dans l'annonce, contrôle fin du flux (utile pour un graphe d'agents avec conditions) |
| API backend | **FastAPI** | Flask | Standard demandé, async natif (important pour appels LLM/DB concurrents) |
| Frontend (optionnel) | Réutilise Next.js/TypeScript/Tailwind (tu maîtrises déjà) | Streamlit pour prototyper vite | Streamlit pour la V1 rapide, Next.js si tu veux un vrai produit démontrable |
| Conteneurisation | **Docker + docker-compose** (Qdrant + Neo4j + FastAPI + Redis optionnel) | — | Demandé explicitement, montre que tu sais déployer un stack multi-service |
| Évaluation | **RAGAS** (faithfulness, context precision/recall, answer relevance) | DeepEval, TruLens | RAGAS est le standard actuel, bien documenté, s'intègre facilement à un pipeline LangChain |
| Observabilité / tracing | **LangSmith** (gratuit en tier dev) ou **Langfuse** (self-hosted, open source) | Phoenix (Arize) | Indispensable pour débugger un système agentique multi-étapes — les recruteurs valorisent cette compétence |
| Versioning / CI | Git/GitHub + GitHub Actions basique (lint + tests) | — | Demandé explicitement dans l'annonce |

---

## 4. Architecture du graphe Neo4j (schéma de départ)

```
(Framework {name, version, release_date})
(Module {name, path})
(Concept {name, description})  -- ex: "Chain", "Agent", "Index", "Retriever"
(Fonctionnalité {name})        -- ex: "hybrid_search", "streaming", "memory"

(Framework)-[:CONTIENT]->(Module)
(Module)-[:IMPLEMENTE]->(Concept)
(Concept)-[:EQUIVALENT_A]->(Concept)          -- ex: LangChain "Retriever" == LlamaIndex "QueryEngine"
(Module)-[:DEPEND_DE]->(Module)
(Framework)-[:SUPPORTE]->(Fonctionnalité)
(Version)-[:DEPRECIE]->(Module)
(Version)-[:CASSE_COMPATIBILITE_AVEC]->(Version)
```

C'est la relation `EQUIVALENT_A` entre concepts de frameworks différents qui fait tout l'intérêt du projet — c'est elle qui permet de répondre à *"comment fait-on X dans LlamaIndex si je connais Y dans LangChain"*. C'est aussi la plus dure à extraire automatiquement : prévois de la construire à moitié manuellement au départ (10-20 mappings clés), puis semi-automatiser avec un LLM.

---

## 5. Roadmap (12 semaines, calée sur un été de stage)

### Semaine 1-2 — Fondations & ingestion
- Setup Docker Compose (Qdrant + Neo4j + FastAPI skeleton)
- Scraper Playwright sur la doc de LangChain (commence par un seul framework)
- Chunking structuré (par header Markdown/HTML), stockage des métadonnées (framework, version, url source, section)
- **Livrable :** pipeline d'ingestion qui tourne de bout en bout sur un framework

### Semaine 3-4 — RAG classique fonctionnel
- Embeddings + indexation Qdrant
- Retrieval dense simple + génération avec Mistral/DeepSeek
- Ajoute le sparse (BM25) et combine en hybrid search
- **Livrable :** tu peux poser une question sur LangChain et obtenir une réponse sourcée

### Semaine 5-6 — Extension multi-frameworks + re-ranking
- Répète l'ingestion pour LlamaIndex et Haystack
- Ajoute le re-ranker après retrieval
- Premiers tests de prompt engineering pour forcer le LLM à comparer plutôt que décrire un seul framework
- **Livrable :** questions comparatives simples ("X existe dans LlamaIndex ?") fonctionnent

### Semaine 7-8 — Graph RAG
- Construction manuelle du schéma Neo4j + premiers mappings `EQUIVALENT_A`
- Extraction semi-automatique via LLM structuré (function calling) pour peupler le graphe depuis les chunks de doc
- Pipeline hybride : requête Cypher pour la structure + vector search pour la nuance sémantique
- **Livrable :** questions de type "quels modules dépendent de X" ou "l'équivalent de Y dans Z" fonctionnent via le graphe

### Semaine 9-10 — IA agentique multi-agents
- LangGraph : agent orchestrateur qui route vers (a) recherche vectorielle, (b) requête graphe, (c) les deux
- Agent "extracteur" qui peut enrichir le graphe à la volée si une info manque
- Agent "vérificateur" qui checke que la réponse cite bien des sources réelles (anti-hallucination basique)
- **Livrable :** démo d'un flux agentique complet visible dans LangSmith/Langfuse

### Semaine 11 — Évaluation
- Dataset de 30-50 questions gold (toi-même en connaissant les 3 frameworks, ou semi-généré par LLM puis vérifié)
- RAGAS : mesure faithfulness, context precision/recall
- Itère sur le chunking/prompt en fonction des résultats
- **Livrable :** un rapport chiffré de performance, très valorisé en entretien

### Semaine 12 — Déploiement & polish
- Dockerisation complète, docker-compose up en une commande
- README soigné avec architecture, diagrammes, résultats d'évaluation
- Petit frontend (Streamlit rapide ou Next.js si le temps permet) pour démo live
- **Livrable :** projet présentable en entretien, avec repo GitHub propre

---

## 6. Pièges à anticiper

- **La doc change vite** : versionne tes données scrapées (date + version du framework) pour ne pas te retrouver avec des infos obsolètes en cours de projet.
- **Le mapping `EQUIVALENT_A` est subjectif** : documente ta méthodologie (pourquoi tu considères deux concepts équivalents) — c'est un excellent sujet de discussion en entretien, pas une faiblesse à cacher.
- **Ne sur-ingénierie pas le multi-agent trop tôt** : fais marcher le RAG simple d'abord, l'agentique vient ensuite en surcouche.
- **Budget API** : local pour les embeddings/re-ranking (gratuit), API seulement pour la génération finale (coût maîtrisable même avec beaucoup de tests).
- **Rate limiting sur le scraping** : mets des délais, respecte les robots.txt des sites de doc.

---

## 7. Ce que tu dois apprendre en amont (ordre conseillé)

1. Bases RAG (chunking, embeddings, retrieval) — tu peux commencer directement, c'est le plus intuitif
2. Qdrant (CRUD, hybrid search, filtres par métadonnées)
3. Neo4j + Cypher (requêtes de base : MATCH, CREATE, relations)
4. LangGraph (states, nodes, edges conditionnels — différent de LangChain classique)
5. RAGAS (métriques d'évaluation RAG)
6. Function calling structuré avec un LLM (pour l'extraction d'entités vers le graphe)

Si tu veux, je peux te donner un starter kit concret (docker-compose.yml + script d'ingestion Playwright + schéma Neo4j initial en Cypher) pour attaquer directement la semaine 1.
