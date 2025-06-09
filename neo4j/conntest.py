import os
import json
import dotenv
from neo4j import GraphDatabase

load_status = dotenv.load_dotenv(".env")
if load_status is False:
    raise RuntimeError('env not loaded')

URI = os.getenv("NEO4J_HOST")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print('connection established')

    records, summary, keys = driver.execute_query(
            #"""MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name""",
            """
            Match (n:User)
            RETURN n LIMIT 25;
            """,
            database_="neo4j",
    )

    for record in records:
        print(record.data())
    
