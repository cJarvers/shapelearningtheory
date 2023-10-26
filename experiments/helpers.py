
def print_table(test_names, results, cellwidth=15):
    """Print results of test runs as a table on the command line."""
    print("Test results:")
    print("| " + "Network".ljust(cellwidth-1), end="|")
    for testname in test_names:
        print(" " + testname.ljust(cellwidth-1), end="|")
    print("\n" + ("|" + "-" * cellwidth) * (len(test_names)+1) + "|")
    for model, results in results.items():
        print("| " + model.ljust(cellwidth-1), end="")
        for testname, result in results.items():
            r = round(result[0]["test_metric"], ndigits=3)
            print("|" + f"{r}".rjust(cellwidth-2), end="  ")
        print("|")