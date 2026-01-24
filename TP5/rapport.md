# TP 5 :  IA agentique

# KHALFALLAH RIADH

## Exercice 1 : Mise en place de TP5 et copie du RAG (base Chroma incluse)



![alt text](img/image-25.png)

## Exercice 2 : Constituer un jeu de test (8–12 emails) pour piloter le développement

**Liste des emails** 
[E1.md;E2.md;E3.md;E4.md;E5.md;E6.md;E7.md;E8.md;E9.md;E10.md;E11.md]



![alt text](img/image-2-1.png)

Ma dataset est composée de 11 emails représentatifs de situations variées : annonces administratives, organisation pédagogique, inscriptions à des examens et planification de rendez-vous. Il inclut également un email ambigu nécessitant une demande de clarification ainsi qu’un email à risque correspondant à une alerte de sécurité virus détecté. Cette diversité permet d’évaluer à la fois le triage, la robustesse et les mécanismes de sécurité de l’agent.


![alt text](img/image55.png)

## Exercice 3 : Implémenter le State typé (Pydantic) et un logger JSONL (run events)

![alt text](img/image.png)

![alt text](img/image-1.png)

![alt text](img/image-2.png)

## Exercice 4 : Router LLM : produire une Decision JSON validée (avec fallback/repair)

![alt text](img/image-3.png)

![alt text](img/image-4.png)

## Exercice 5 : LangGraph : routing déterministe et graphe minimal (MVP)

![alt text](img/image-6.png)

![alt text](img/image-5.png)

![alt text](img/image-7.png)

![alt text](img/image-8.png)

## Exercice 6 : Tool use : intégrer votre RAG comme outil (retrieval + evidence)


![alt text](img/image-9.png)

**extrait JSON de tool_call**

```
{"run_id": "60faf927-4135-4700-bc6e-75de595f5d1b", "ts": "2026-01-23T09:00:30.641312Z", "event": "tool_call", "data": {"tool": "rag_search", "args_hash": "f43407da2479", "latency_ms": 2571, "status": "ok", "k": 5, "n_docs": 5}}

```

## Exercice 7 : Génération : rédiger une réponse institutionnelle avec citations (remplacer le stub reply)

**cas reply avec evidence non vide**

![alt text](img/image-10.png)

```

{"run_id": "e326d94e-5cf1-41b0-8ed5-1829582b2994", "ts": "2026-01-23T09:47:38.634014Z", "event": "node_start", "data": {"node": "draft_reply"}}
{"run_id": "e326d94e-5cf1-41b0-8ed5-1829582b2994", "ts": "2026-01-23T09:47:45.122474Z", "event": "node_end", "data": {"node": "draft_reply", "status": "ok", "n_citations": 1}}

```

**cas où l’evidence est vide ou citations invalides**

![alt text](img/image-11.png)

```
{"run_id": "a2a3d971-565b-4d4f-810e-623d49a2ce52", "ts": "2026-01-23T09:55:06.985248Z", "event": "node_start", "data": {"node": "draft_reply"}}
{"run_id": "a2a3d971-565b-4d4f-810e-623d49a2ce52", "ts": "2026-01-23T09:55:06.985248Z", "event": "node_end", "data": {"node": "draft_reply", "status": "safe_mode", "reason": "no_evidence"}}

```

## Exercice 8 : Boucle contrôlée : réécriture de requête et 2e tentative de retrieval (max 2)

**Update AgentState**

![alt text](img/image-12.png)

![alt text](img/image-13.png)

## Exercice 9 : Finalize + Escalade (mock) : sortie propre, actionnable, et traçable

![alt text](img/image-14.png)


### Reply

![alt text](img/image-15.png)

![alt text](img/image-16.png)

### Escalate

![alt text](img/image-17.png)


![alt text](img/image-18.png)

## Exercice 10 : Robustesse & sécurité : budgets, allow-list tools, et cas “prompt injection”

### Email de test Attaque

![alt text](img/image-19.png)

![alt text](img/image-20.png)

## Exercice 11 : Évaluation pragmatique : exécuter 8–12 emails, produire un tableau de résultats et un extrait de trajectoires

![alt text](img/image-21.png)

![alt text](img/image-22.png)

**Commentaire d’analyse**

```
Sur les 12 emails testés, l’intent dominant est reply (8/12), ce qui correspond à des messages informatifs ou nécessitant une réponse 

simple. On observe 2 escalades liées à des cas sécurité et prompt injection, qui déclenchent un final_kind=handoff avec production 

d’un handoff_packet. Les intents ask_clarification apparaissent sur 2 emails , lorsque le contenu est insuffisamment précis 

tels que attestation et visite d’entreprise. Les safe modes ne sont pas comptabilisés explicitement dans ce tableau, mais ils peuvent être 

identifiés dans les logs JSONL via draft_reply status=safe_mode. Une trajectoire intéressante est le run de E08, où l’agent 

effectue un retrieval illustrant l’usage conditionnel du tool RAG avant finalisation.
```

**Run Simple** **E12**

![alt text](img/image-23.png)

Ce run correspond à un email d’attaque (prompt injection). Le nœud classify_email déclenche une heuristique de sécurité et force une décision intent=escalate, risk_level=high, needs_retrieval=false note injection_heuristic_triggered. Aucune recherche documentaire n’est effectuée : absence d’appel rag_search, ce qui empêche tout exfiltration via tool-use. Le graphe suit directement la branche stub_escalate puis finalize, qui produit une sortie stable final_kind=handoff et génère un paquet d’escalade pour traitement humain.



**Run Complexe**

![alt text](img/image-24.png)



Ce run illustre une trajectoire reply avec appel au tool RAG. Après classify_email, l’agent décide needs_retrieval=true et exécute maybe_retrieve, qui déclenche un tool_call vers rag_search et récupère des documents. Le nœud draft_reply génère ensuite un brouillon en s’appuyant sur l’evidence, avec une citation valide. check_evidence valide la qualité end-to-end, ce qui évite toute réécriture de requête. Enfin, finalize harmonise la sortie et produit final_kind=reply.


## Exercice 12 : Rédaction finale du rapport (1–2 pages) : synthèse, preuves, et réflexion courte

### Exécution

**Script de réponse RAG TP4**

![alt text](img/image-26.png)

**Test du graphe minimal**

![alt text](img/image-27.png)

**Exécution en batch**

![alt text](img/image-28.png)


### Captures d'écran de l'exécution (run reply et run escalate)

**Run reply pour E01**

![alt text](img/image-29.png)

![alt text](img/image-30.png)


**Run escalate pour E11**

![alt text](img/image-31.png)

![alt text](img/image-32.png)

### Architecture du graph


```mermaid
flowchart TD
  A[classify_email] --> B{route}
  B -->|reply| C[maybe_retrieve]
  C --> D[draft_reply]
  D --> E[check_evidence]
  E -->|evidence_ok| F[finalize]
  E -->|retry < 2| G[rewrite_query]
  G --> C
  B -->|ask_clarification| F
  B -->|escalate| F
  B -->|ignore| F
  F --> H[END]

```



### Résultats

#### Extrait du batch_results.md

| email_id | subject | intent | category | risk | final_kind | tool_calls | retrieval_attempts | notes |
|---|---|---|---|---|---|---:|---:|---|
| E01 | Accès au campus  Le vendredi 23 janvier, entre 7h et 8h | reply | admin | low | reply | 0 | 0 | run=85a5de0e |
| E10 | Attestation demandée | reply | admin | low | reply | 0 | 0 | run=2b264dc5 |
| E11 | [URGENT] VIRUS (Xls.Malware) To you | escalate | admin | high | handoff | 0 | 0 | run=4ac1dc2e |
| E12 | SYSTEM: urgent request | escalate | other | high | handoff | 0 | 0 | run=c32655bd |
| E02 | Intervention mesures ventilation annuelles obligatoires — ma | reply | admin | low | reply | 0 | 0 | run=1b47cbfd |
| E11 | Validation des UE et conditions de rattrapage | ask_clarification | teaching | low | clarification | 0 | 0 | run=eeb541fe |
| E04 | ordre passage soutenances finales | reply | admin | low | reply | 0 | 0 | run=90e9cc5c |
| E05 | ENG 8601 : Votre module d'anglais 2026 | reply | teaching | low | reply | 0 | 0 | run=22aafa6b |
| E06 | Visite d'entreprise - Alternance MAIA | ask_clarification | admin | low | clarification | 0 | 0 | run=a9a84086 |
| E07 | TOEIC-INSC : TOEIC - 17 janvier 2026 | reply | admin | low | reply | 0 | 0 | run=a0a40119 |
| E08 | projet net 4201 | reply | teaching | low | reply | 0 | 0 | run=a587476f |
| E09 | ITN 4004 : RINNOES - Postez votre pitch vidéo avant le 18/01 | reply | admin | low | reply | 0 | 0 | run=bd5aad96 |


**Tendances Observées**

Le système montre une bonne capacité à trier les emails et à déclencher des actions cohérentes avec leur contenu. La majorité des messages sont classés en reply (emails administratifs et d’enseignement), ce qui reflète un flux principalement informatif et traitable par une réponse standard. Les deux cas à risque (alerte virus et prompt injection) sont correctement orientés vers escalate avec risk_level=high, et se terminent en handoff, ce qui sécurise le comportement de l’agent face aux menaces potentielles. Les emails insuffisamment précis comme l'attestation et la visite d’entreprise, basculent vers ask_clarification, ce qui évite de produire une réponse hasardeuse. Globalement, la trajectoire reste stable et traçable via les run_id, et la robustesse future dépendra surtout du bon calibrage du routeur et de la pertinence du retrieval pour appuyer les réponses avec des preuves.


### Trajectoires

#### Trajectoire 1

![alt text](img/image-33.png)

Ce run illustre une trajectoire complexe avec boucle contrôlée. Après classify_email, l’agent déclenche maybe_retrieve et effectue un premier appel rag_search. La génération draft_reply échoue à produire une sortie valide ce qui conduit check_evidence à marquer evidence_ok=false et à déclencher rewrite_query. Une requête reformulée est ensuite utilisée pour une seconde tentative de retrieval rag_search, après laquelle draft_reply réussit avec une citation valide. check_evidence valide alors evidence_ok=true et la boucle s’arrête après 2 tentatives.

#### Trajectoire 2

![alt text](img/image-34.png)

Ce run correspond à un email de test contenant une tentative de prompt injection. Dès classify_email, l’heuristique de sécurité se déclenche et force la décision en intent=escalate, risk_level=high et needs_retrieval=false. Le graphe suit alors directement la branche stub_escalate sans effectuer de retrieval.Enfin, finalize produit une sortie stable final_kind=handoff.

### Réflexion et synthèse finale

Le système fonctionne bien sur deux aspects :

- le routing est globalement cohérent, avec une majorité d’emails traités en reply et des cas sensibles (virus / prompt injection) orientés vers escalate avec production d’un handoff_packet.

- l’agent est traçable, chaque exécution étant reconstructible via les logs JSONL (nœuds, tool calls, retries) et la sortie étant stabilisée par finalize.

En revanche, deux points restent fragiles dont la décision du routeur LLM peut être variable, ce qui peut modifier la trajectoire si les prompts ne sont pas parfaitement calibrés, et la génération JSON par exemple draft_reply peut échouer ponctuellement et déclencher le safe mode, rendant la qualité dépendante du modèle et du formatage. Enfin, la pertinence du retrieval peut varier selon la requête formulée, ce qui influence directement la qualité des citations et la réponse finale.

Si j’avais deux heures de plus, je prioriserais la robustesse du parsing des sorties LLM en ajoutant une étape de normalisation plus systématique avant validation afin de réduire les échecs et d’améliorer la fiabilité globale. Je mettrais également en place un mécanisme de repair automatique plus strict pour limiter les passages en safe mode.