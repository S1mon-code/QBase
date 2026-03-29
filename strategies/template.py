"""
QBase Strategy Template — redirects to template_simple.py.

For simple strategies: use template_simple.py (recommended for 90% of cases)
For complex strategies: use template_full.py (pyramiding, multi-TF, etc.)
"""
from strategies.template_simple import SimpleStrategy as TemplateStrategy

__all__ = ['TemplateStrategy']
