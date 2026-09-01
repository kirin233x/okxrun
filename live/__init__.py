"""Live execution for the R9 strategy.

The strategy itself lives in the shared ``strategy`` package; this package only
decides how to express its target book as OKX orders, and refuses to express
anything that breaks a configured risk limit.
"""
