"""Make contact-weight failure LOUD. If B1K_CONTACT_WEIGHTS=1 is set and the weights cannot
be used, raise instead of quietly training a uniform arm labelled "weighted" -- the exact
failure that silently invalidated the first launch."""
import re, pathlib, ast
P = pathlib.Path("/root/openpi/src/openpi/training/data_loader.py")
src = P.read_text(); orig = src
# turn each "log the problem then return None" into a hard failure
src = re.sub(
    r'logging\.error\((f?"CONTACT WEIGHTS[^\n]*?)\)\n(\s*)return None',
    r'raise RuntimeError(\1)',
    src,
)
src = re.sub(
    r'logging\.error\(\n(\s*)(f?"CONTACT WEIGHTS[^\n]*?\n\s*)\)\n\s*return None',
    r'raise RuntimeError(\n\1\2)',
    src,
)
P.write_text(src); ast.parse(src)
n_raise = src.count("raise RuntimeError(")
print("fail-fast patch: changed =", src != orig, "| RuntimeError sites =", n_raise)
print("remaining silent 'return None' after a CONTACT error:", "CONTACT WEIGHTS" in src and "return None" in src.split("_TwoTierWeightedSampler")[-1].split("def create_b1k_data_loader")[0])
