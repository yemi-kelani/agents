"""Citation-faithfulness eval — spec 11.

Without this the pipeline is unmeasured; with it, every design change has a
number attached. What is measured is not "is the report good" — it is **is each
cited claim actually supported by its cited source**, which is the question the
Liu, Zhang & Liang (2023) audit shows a quality-based eval gets backwards.
"""
