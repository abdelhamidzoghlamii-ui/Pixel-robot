"""Parser check: python3 test_jevlike.py"""
from jevlike import BLANK, DEMOS, TEMPLATE_HELP, parse

state, qs = parse(TEMPLATE_HELP + """
INPUT:
Battery: 9 percent.
The dock is far.

QUESTION: What next?
- charge
- explore
- charge
QUESTION: Tell
someone?
- yes
- no
""")
assert state == "Battery: 9 percent. The dock is far."
assert qs == [("What next?", ["charge", "explore"]), ("Tell someone?", ["yes", "no"])]
for demo in DEMOS:
    parse(demo)
for bad, msg in ((BLANK, "INPUT is empty"), ("INPUT: x\n", "no QUESTION"),
                 ("INPUT: x\nQUESTION: q\n- a\n", "at least 2"), ("INPUT: x\nQUESTION:\n- a\n- b\n", "no instruction"),
                 ("hello\nINPUT: x\n", "before INPUT")):
    try:
        parse(bad)
        raise AssertionError(bad)
    except ValueError as e:
        assert msg in str(e), e
print("ok")
