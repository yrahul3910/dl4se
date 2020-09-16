from interpret import ResultsInterpreter


directories = ["1 day", "7 days", "14 days",
               "30 days", "90 days", "180 days", "365 days"]
datasets = ["cloudstack", "hadoop", "cocoon",
            "deeplearning", "hive", "node", "ofbiz", "qpid"]
path = "/Users/ryedida/Desktop/menzies/DL4SE/issue_close_time/nondl/nondl-log/"

for data in datasets:
    for time in directories:
        print(f"{data}-{time}")
        print("=" * len(f"{data}-{time}"))
        ri = ResultsInterpreter(
            [f"{path}{data}-{time}-none", f"{path}{data}-{time}-weighted", f"{path}{data}-{time}-all"])
        ri.compare()
        print()
