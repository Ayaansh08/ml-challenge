import json
with open("hive/tasks.json", "r") as f: data = json.load(f)
for t in data["tasks"]:
    if t["id"] == "run-e2e-pipeline":
        t["assignee"] = "god"
    if t["id"] == "doc-reconciliation":
        t["status"] = "blocked"
        t["notes"] = "Blocked: Pam is archived"
with open("hive/tasks.json", "w") as f: json.dump(data, f, indent=4)
