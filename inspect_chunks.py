import sqlite3

conn = sqlite3.connect("vector_metadata.db")
cur = conn.cursor()

# Cuántos chunks por namespace
cur.execute("SELECT namespace, COUNT(*) FROM chunks WHERE active=1 GROUP BY namespace")
print("Chunks por namespace:", cur.fetchall())

# Muestra algunos ejemplos para el namespace 'compromidos'
cur.execute(
    "SELECT id, source, substr(content,1,200) FROM chunks WHERE namespace='compromidos' AND active=1 LIMIT 5"
)
for row in cur.fetchall():
    print("ID:", row[0])
    print("Fuente:", row[1])
    print("Extracto:", row[2].replace("\n", " "), "...")
    print("----")
