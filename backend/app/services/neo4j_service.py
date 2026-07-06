from neo4j import GraphDatabase
from app.core.config import settings


class Neo4jService:
    def __init__(self):
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        return self._driver

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def run_query(self, query: str, params: dict | None = None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def create_framework(self, name: str, version: str, release_date: str | None = None):
        return self.run_query(
            """
            MERGE (f:Framework {name: $name, version: $version})
            SET f.release_date = $release_date
            RETURN f
            """,
            {"name": name, "version": version, "release_date": release_date},
        )

    def search_entities(self, query: str, limit: int = 10):
        return self.run_query(
            """
            MATCH (e)
            WHERE e.name CONTAINS $query OR e.description CONTAINS $query
            RETURN e.name AS name, labels(e)[0] AS type, e.description AS description
            LIMIT $limit
            """,
            {"query": query, "limit": limit},
        )

    def get_entity_context(self, entity_name: str, depth: int = 2):
        return self.run_query(
            """
            MATCH path = (e {name: $name})-[*1..$depth]-(related)
            WHERE labels(e)[0] IN ['Component', 'Concept', 'Integration']
            RETURN path
            LIMIT 50
            """,
            {"name": entity_name, "depth": depth},
        )

    def get_related_chunks(self, entity_name: str, limit: int = 10):
        return self.run_query(
            """
            MATCH (e {name: $name})<-[:MENTIONS]-(ch:Chunk)
            RETURN ch.id AS chunk_id, ch.content AS content, ch.url AS url, ch.title AS title
            LIMIT $limit
            """,
            {"name": entity_name, "limit": limit},
        )

    def query_graph(self, entity_terms: list[str], limit: int = 10):
        return self.run_query(
            """
            MATCH (e)
            WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS toLower(term))
            OPTIONAL MATCH (e)<-[:MENTIONS]-(ch:Chunk)
            OPTIONAL MATCH (e)-[r]-(related)
            WHERE related:Component OR related:Concept OR related:Integration
            RETURN e.name AS entity, labels(e)[0] AS type, e.description AS description,
                   collect(DISTINCT ch.content) AS chunks,
                   collect(DISTINCT {name: related.name, type: labels(related)[0], rel_type: type(r)}) AS relations
            LIMIT $limit
            """,
            {"terms": entity_terms, "limit": limit},
        )


neo4j_service = Neo4jService()
