# test1.py —— Neo4j 连通性冒烟
# 运行（项目根目录）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test1.py
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# 连 system 库，列出所有数据库
with driver.session(database="system") as session:
    result = session.run("SHOW DATABASES")
    for record in result:
        print(dict(record))

driver.close()